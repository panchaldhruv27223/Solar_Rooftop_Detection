import os
import cv2
import sys
import numpy as np
import pandas as pd
import datetime, time
from PIL import Image
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
print(current_dir)

project_root = os.path.abspath(os.path.join(current_dir,".."))
print(project_root)
sys.path.append(project_root)

from Solar_rooftop_Detection_indian_images.get_bounding_box_cordinates_indian import bounding_box
from Solar_rooftop_Detection_indian_images.SAM_logger_indian import sam_logger
from Solar_rooftop_Detection_indian_images.generate_isolated_masks_indian import generate_isolated_mask


import monai
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from segment_anything import build_sam_vit_b

sam_logger.info("SAM Logger initialized")



## now create custom Datset
class my_dataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1024, 1024))
        
        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Load and preprocess the binary mask(1 channel)
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1024,1024))

        ## Convert image and mask to torch tensor 
        image = torch.tensor(image).permute(2,0,1).float()/255.0
        mask = torch.tensor(mask)

        return image, mask
    

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        """
        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in loss to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        """Checks validation loss and determines if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement occurs

        else:
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs...")

        if self.counter >= self.patience:
            print("Early stopping triggered. Training stopped.")
            self.early_stop = True
    


class SAM_MODEL:

    def __init__(self, image_path:str="", mask_path:str="", batch_size:int=2, model_path:str="", lr=1e-3, weight_decay=0, early_patience=10, early_min_delta=0.0001, number_epochs = 30, checkpoint = ""):
        self.images_path = image_path
        self.masks_path = mask_path

        self.images = []

        for i in range(len(os.listdir(self.images_path))):
        
            self.images.append(f"{os.path.join(self.images_path,os.listdir(self.images_path)[i])}")

        print(len(self.images))

        self.masks = []

        for i in range(len(os.listdir(self.masks_path))):
        
            self.masks.append(f"{os.path.join(self.masks_path,os.listdir(self.masks_path)[i])}")

        print(len(self.masks))

        self.batch_size = batch_size
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.num_epochs = number_epochs

        self.checkpoint = checkpoint
        os.makedirs(self.checkpoint, exist_ok=True)

        # Initialize early stopping
        self.patience = early_patience
        self.min_delta = early_min_delta
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)



    def model_train(self, model, data_loader, optimizer, seg_loss, checkpoint):
    
        sam_logger.info("Load The model which is already trained on the inria dataset")

        sam_logger.info("Training Started of SAM model on Indian dataset")

        for epoch in range(self.num_epochs):
            # Train the model
            sam_logger.info(f"Epoch {epoch+1}/{self.num_epochs} started")

            running_loss = 0

            for img, mask in data_loader:
                image_ = img.to(self.device)
                mask_ = mask.to(self.device)
                # sam_logger.debug(f"images size : {image_.shape}, masks size : {mask_.shape}")

                for (image, mask) in zip(image_, mask_):
                    
                    # print(image.shape, mask.shape)

                    ## getting the bounding box coordinates from masks 
                    # print(mask.numpy())
                    ## here we have to pass numpy array
                    # print(mask.numpy().astype(np.uint8))
                    
                    bounding_box_corr = bounding_box(mask.detach().cpu().numpy().astype(np.uint8))
                    # sam_logger.info(f"Bounding Box Coordinates: {len(bounding_box_corr)}")

                    # print(bounding_box_corr)


                    for coordinates in bounding_box_corr:
                        # print(f"points are : {coordinates}")
                        one_isolated_mask = generate_isolated_mask(mask, coordinates)
                        one_isolated_mask = one_isolated_mask.unsqueeze(0)
                        one_isolated_mask = one_isolated_mask/255.0
                        # print(f"Unique values of isolated masks are : ",torch.unique(one_isolated_mask))
                        # print("isolated mask shape : ",one_isolated_mask.shape)
                        
                        # plt.imshow(one_isolated_mask.squeeze(0).numpy(), cmap="gray")
                        # plt.show()
                        
                        
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points = None,
                            boxes = torch.tensor([coordinates]).unsqueeze(0).to(self.device),masks = None)
                        
                        low_res_mask, _ = model.mask_decoder(
                            image_embeddings = model.image_encoder(image.unsqueeze(0)),
                            image_pe = model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings = sparse_embeddings,
                            dense_prompt_embeddings = dense_embeddings,
                            multimask_output = False
                        )

                        upsampled_masks = F.interpolate(low_res_mask, size = (1024, 1024), mode="bilinear", align_corners=False)

                        ## calculate the loss
                        # print("Upsampled mask shape : ",upsampled_masks.shape)
                        # print("Unique values this tensor contain is : ", torch.unique(upsampled_masks))
                        
                        # print(upsampled_masks)
                        loss = seg_loss(upsampled_masks, one_isolated_mask.unsqueeze(0).float())
                        
                        # backward pass (compute gradients of parameters w.r.t. loss)
                        optimizer.zero_grad()

                        loss.backward()
                        
                        # Optimize
                        optimizer.step()
                        
                        if len(bounding_box_corr) > 0:
                            running_loss += loss.item() / len(bounding_box_corr)
                        
                        # print(f"runing loss: {running_loss}")
                        # sam_logger.info(f"Current running loss: {running_loss:.4f}")



                        # to see the output and actual masks
                        # copy_up_sampled = upsampled_masks.clone()

                        # plt.imshow(copy_up_sampled.squeeze(0).squeeze(0).detach().numpy(),cmap="gray")
                        # plt.tight_layout()
                        # plt.show()

            
            # print(f"epoch {epoch}, loss: {running_loss / len(data_loader)}")
            sam_logger.info(f"Epoch {epoch+1} completed. Loss: {running_loss / len(data_loader):.4f}")
            
            self.early_stopping(running_loss, model, epoch)
            if self.early_stopping.early_stop:
                sam_logger.error(f"Early Stoping : Epoch {epoch+1} completed. Loss: {running_loss / len(data_loader):.4f}")
                sam_logger.error(f"TRAINING STOPPED EARLY")
                break  # Stop training if early stopping is triggered


            ## save the model checkpoint every 5 epoch
            
            if (epoch + 1) % 5 == 0:
                current_date = datetime.datetime.now().strftime("%d_%m_%Y")
                os.makedirs(checkpoint, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint, f"check_point_sam_model_epoch_{epoch + 1}_{current_date}learning_rate{self.learning_rate}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                sam_logger.info(f"Model checkpoint saved at {checkpoint_path}")

        
        return model



    def train(self):
        dataset = my_dataset(self.images, self.masks)
        ### Now implement Data Loader
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        model = build_sam_vit_b(self.model_path)
        model.to(self.device)

        optimizer = Adam(model.mask_decoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

        for name, param in model.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)


        # trained_model = self.model_train(model, data_loader, optimizer, seg_loss, self.checkpoint)


        # final_model_checkpoint = os.path.join(self.checkpoint, "Final_Models")
        # os.makedirs(final_model_checkpoint, exist_ok=True)

        # current_date = f"FineTune_model_epoch_{self.num_epochs}_"+datetime.datetime.now().strftime("%d_%m_%Y")+f"learning_rate{self.learning_rate}.pth"
        # final_model_path = os.path.join(final_model_checkpoint, current_date)

        # torch.save(trained_model.state_dict(), final_model_path)



if __name__ == "__main__":
    print("Dhruv Panchal")


    image_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train/images"

    mask_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train/masks"

    batch_size = 2

    model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/SAM_Base_model/sam_vit_b.pth"

    learning_rate=1e-3 
    
    weight_decay=0
    
    early_patience=10
    
    early_min_delta=0.0001 
    
    number_epochs = 30
    
    checkpoint = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/easy_model_scratch_only"


    sam_model = SAM_MODEL(image_path= image_path, mask_path=mask_path, batch_size=batch_size, model_path=model_path, lr=learning_rate, weight_decay=weight_decay, early_patience=early_patience, early_min_delta=early_min_delta, number_epochs = number_epochs, checkpoint = checkpoint)

