import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import albumentations as A
from albumentations.pytorch import ToTensorV2

import itertools
import optuna

import sys
current_dir = os.path.dirname(__file__)
# print(current_dir)

project_root = os.path.abspath(os.path.join(current_dir,"../.."))
# print(project_root)

sys.path.append(project_root)
from Solar_rooftop_Detection_indian_images.accuracy_indian import compute_metrics

# from accuracy_indian import compute_metrics

# Augmentation Of Images and masks
 
def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2()
    ])



# Your function to extract individual masks and bounding boxes
def extract_individual_objects(input_mask):
    # Ensure the input is a binary mask (0 and 255 values)
    _, binary_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_masks = []
    bounding_boxes = []
    for contour in contours:
        object_mask = np.zeros_like(binary_mask)
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([x, y, x + w, y + h])  # [x_min, y_min, x_max, y_max]
        cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
        object_masks.append(object_mask // 255)  # Convert to 0/1 binary
    return object_masks, bounding_boxes



# Define the dataset For MaskRCNN
class RooftopDatasetMaskRCNN(Dataset):
    def __init__(self, root, augmentation=True):
        self.root = root
        self.augmentation = augmentation
        if self.augmentation:
            self.transform = get_training_augmentation()
        
        self.images_dir = os.path.join(root, 'images')
        self.masks_dir = os.path.join(root, 'masks')
        self.imgs = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(('.jpg', '.png'))])
        assert len(self.imgs) == len(self.masks), "Mismatch between images and masks"

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.imgs[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load multi-instance mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Extract individual masks and boxes
        instance_masks, boxes = extract_individual_objects(mask)

        if not instance_masks or not boxes:
            # print(f"No rooftops detected in {img_path}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, 1024, 1024), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Validate consistency
            assert len(boxes) == len(instance_masks), f"Mismatch: {len(boxes)} boxes vs {len(instance_masks)} masks"
            
            # Stack masks and convert to tensor
            masks = np.stack(instance_masks, axis=0)  # Shape: (num_instances, 1024, 1024)
            
            if self.augmentation:
                transformed = self.transform(image=img, masks=list(masks))
                img = transformed["image"]
                masks = transformed["masks"]

                img = img.float()
                
                masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in masks])
                for i in masks:
                    torch.unique(i)

                masks = masks.float()
                img = img / 255.0

            else:
                # print("No augmentation applied.")
                img = T.ToTensor()(img)
                masks = torch.from_numpy(masks).to(dtype=torch.uint8)
                

            # Convert boxes to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Shape: (num_instances, 4)
            
            # Assign labels (all 1 for rooftops)
            labels = torch.ones((len(instance_masks),), dtype=torch.int64)
        
            # Debug info
            # print(f"{img_path}: {len(instance_masks)} rooftops")
        
        
        # print(f"image size : {img.size()} and masks size is : {masks.size()}")
        # print(f"uniques values of images : {torch.unique(img)}")
        # print(f"uniques values of masks : {torch.unique(masks)}")


        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.imgs)
    


# Mask RCNN model.

class maskRCNN:
    def __init__(self, batch_size=4, num_classes = 2, learning_rate = 0.0001, augmentation=True, number_epochs=50, optimizer_name= "adam", weight_decay= 0.0,data_dir :str ="", model_output_path : str="", image_output_path : str="",  model_path : str ="", num_workers=2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_epochs = number_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.root_dir = data_dir
        self.model_path = model_path
        self.model_output_path = model_output_path
        self.image_output_path = image_output_path
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

    def get_model_instance_segmentation(self, num_classes=2):  # Background (0) + Rooftop (1)
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model


    # Training function with enhanced logging
    def train_model(self, model, data_loader, optimizer, device, num_epochs=10):
        print("model training.")
        losses_list = []

        for epoch in range(num_epochs):
            # print(f"Epoch {epoch+1}/{num_epochs}")
            total_loss = 0
            for i, (images, targets) in enumerate(data_loader):
                # print(f"Batch {i+1}: Images and targets loaded")
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Log individual losses
                # print(f"Batch {i+1} losses: { {k: v.item() for k, v in loss_dict.items()} }")
                
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stabilize training
                optimizer.step()

                total_loss += losses.item()
            
            total_loss = total_loss / len(data_loader)
            print(total_loss)
            losses_list.append(total_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}")

        return losses_list
    
        # Load the fine-tuned model

    def load_finetuned_model(self,model_path):
        my_model = torch.load(model_path, weights_only=False)
        my_model.to(self.device)
        my_model.train()
        return my_model
        

    def train(self):

        dataset = RooftopDatasetMaskRCNN(root=self.root_dir, augmentation=self.augmentation)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))

        # iter_data = iter(data_loader)
        # dummmy = next(iter_data)
        # print(dummmy)

        # return

        print("Data is loaded")

        if self.model_path : 
            model = self.load_finetuned_model(self.model_path)
            print("loading pre trian model")
        else:
            print("load model")
            model = self.get_model_instance_segmentation(num_classes=self.num_classes)
            model.to(self.device)
            model.train()

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer")
        
        loss = self.train_model(model, data_loader, optimizer, self.device, num_epochs=self.num_epochs)
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        torch.save(model, self.model_output_path)

        print(f"Model saved !.")
              
              
        epochs = list(range(1, 51))
        maskrcnn_loss = loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, maskrcnn_loss, label='Mask R-CNN Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.image_output_path)
        plt.tight_layout()
        # plt.show()
        
        return sum(loss)/self.num_epochs


    


# Define the dataset class For deeplabv3
class SegmentationDataset(Dataset):
    def __init__(self, root, image_folder="images", mask_folder="masks", is_transform=False, transforms=None):
        self.root = root
        self.transform = transforms
        self.image_folder = os.path.join(root, image_folder)
        self.mask_folder = os.path.join(root, mask_folder)
        self.image_names = sorted(os.listdir(self.image_folder))
        self.mask_names = sorted(os.listdir(self.mask_folder))
        self.is_transform = is_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_names[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_names[idx])
        # image = Image.open(img_path).convert("RGB")

        # mask = Image.open(mask_path).convert("L")  # Grayscale mask

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype(np.uint8)
        image = image/255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.is_transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask/255.0
            mask = mask.unsqueeze(0)

            # print(f"image size : {image.size()} mask size : {mask.size()}")

        else:
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # Mask to tensor (0-1 range)
            # print(f"image size : {image.size()} mask size : {mask.size()}")

        # print(torch.unique(image))
        
        return {"image": image.float(), "mask": mask.float()}
    


## deeplabv3 model
class deepLabV3:

    def __init__(self, data_dir :str , model_output_path : str, image_output_path : str,  model_path : str ="", num_epochs : int = 50, batch_size :int = 2, learning_rate : float = 1e-4, optimizer_name="adam", weight_decay=0 , is_transform=False):
        self.data_dir = data_dir
        self.model_output_path = model_output_path
        self.image_output_path = image_output_path
        self.model_path = model_path
        self.number_epoch = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_transform = is_transform
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay


    def train_model(self, model, dataloader, num_epochs=50, learning_rate = 1e-4):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()  # For binary segmentation
        
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
            
        loss_list = []
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            for sample in dataloader:
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)
                # print(masks.size())
                # print(torch.unique(masks))

                optimizer.zero_grad()
                outputs = model(inputs)["out"]  # Shape: (batch, 1, H, W)
                
                # print(outputs.shape)
                # print(torch.unique(outputs))
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            running_loss = running_loss / len(dataloader.dataset)
            loss_list.append(running_loss)
            print(f"Train Loss: {running_loss:.4f}")
            
        return model, loss_list


    def create_deeplab(self, output_channels=1):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.classifier = DeepLabHead(2048, output_channels)
        return model


    def deeplabv3_train(self):

        if self.is_transform :
            transform = get_training_augmentation()
        else:
            transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()])
        dataset = SegmentationDataset(self.data_dir, is_transform=self.is_transform, transforms= transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # iter_data = iter(dataloader)
        # dummmy = next(iter_data)
        # print(dummmy)
        # image_d, mask_d = dummmy["image"], dummmy["mask"]
        # print(f"images size {image_d.size()} mask size {mask_d.size()}")
        # return   

        
        if self.model_path : 
            print("Loading The Pretrain Model")
            model = torch.load(self.model_path, weights_only=False)
        else:
            print("Define the model Architecture.")
            model = self.create_deeplab(output_channels=1)  
            # print(model)

        trained_model, losses = self.train_model(model, dataloader, num_epochs=self.number_epoch, learning_rate=self.learning_rate)
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        torch.save(trained_model, self.model_output_path)

        epochs = list(range(1,self.number_epoch+1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, label='DeepLabV3 Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.image_output_path)
        plt.tight_layout()
        # plt.show()
        
        return sum(losses)/self.number_epoch




### UNET MODEL  
# Dataset for UNET Model
class RooftopDataset(Dataset):
    def __init__(self, root, image_folder="images", mask_folder="masks", is_transform="" ,transform=None):
        self.root = root
        self.image_dir = os.path.join(self.root, image_folder)
        self.mask_dir = os.path.join(root, mask_folder)
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.image_masksnames = sorted(os.listdir(self.mask_dir))
        self.is_transform = is_transform
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_masksnames[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        image = image/255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        
        if self.is_transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)
            # print("input masks: ",mask.size())
        # print("start from data loader")
        # print(image.shape, image.dtype)
        # print(mask.shape, mask.dtype)
        # print("end from data loader")
        
        else:
            # print("No augmentation")
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
            
        # print("input masks: ",mask.size())
        # print("start from data loader")
        # print(image.size(), image.dtype)
        # print(torch.unique(image))
        # print(mask.size(), mask.dtype)
        # print(torch.unique(mask))
        # print("end from data loader")

        return image.float(), mask.float()


# U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        self.encoder = nn.ModuleList([
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        ])
        
        self.pool = nn.MaxPool2d(2)
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(256, 128),
            conv_block(128, 64),
            conv_block(64, 32),
            conv_block(32, 16)
        ])
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)
        
        for i in range(4):
            x = self.upconv[i](x)
            enc_out = enc_outs[-(i+2)]
            if x.shape[2:] != enc_out.shape[2:]:
                x = nn.functional.interpolate(x, size=enc_out.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, enc_out], dim=1)
            x = self.decoder[i](x)
        
        return torch.sigmoid(self.final_conv(x))
    

class UnetTrain:
    def __init__(self, root_dir, is_transform, learning_rate=1e-4, number_epoch=50, batch_size=4,optimizer_name="adam", weight_decay=0 ,image_output_path="", model_output_path="", model_path=""):
        
        self.root_dir = root_dir
        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.image_output_path = image_output_path
        self.model_output_path = model_output_path
        self.model_path = model_path
        self.is_transform = is_transform
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        
        if is_transform:
            self.transform = get_training_augmentation()
        else:
            self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor()
                ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


    def train(self, model, train_loader, criterion, optimizer, epochs=50):

        losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)  # Ensure correct shape [B, 1, H, W]
                ## masks shape
                # print(masks.size())
                
                optimizer.zero_grad()
                outputs = model(images)
                # outputs = outputs.squeeze()
                # print(torch.unique(outputs))
                # print("output masks shape is : ",outputs.size())
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # print(loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            print(epoch_loss/len(train_loader))
            losses.append(epoch_loss/len(train_loader))
            
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_output_path)        
        print("Model saved!")

        return losses 


    def train_unet(self):
        # Load datasets
        train_dataset = RooftopDataset(self.root_dir, image_folder="images", mask_folder="masks", is_transform = self.is_transform, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # print(train_loader)
        # data_iter = iter(train_loader)
        # print(data_iter)
        # images_dummy, masks_dummy = next(data_iter)
        # print(torch.unique(images_dummy))
        # print(torch.unique(masks_dummy))
        # print(images_dummy.shape, masks_dummy.shape)
        # print(train_loader[0])

        model = UNet()
        
        # # Training Setup
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))
            print("load pretrain model")
        
        print("define new model")
        
        model = model.to(self.device)

        criterion = nn.BCELoss()
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer")
        print("training start.")
        loss = self.train(model, train_loader, criterion= criterion, optimizer=optimizer,epochs=self.number_epoch)
        print("training end.")
        unet_loss = loss

        epochs = list(range(1,51))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, unet_loss, label='UNet Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.image_output_path)
        # plt.show()

        return sum(unet_loss)/self.number_epoch


if __name__ == "__main__":
    
    # param_grid = {
    #     "batch_size": [2, 4],
    #     "learning_rate": [1e-2, 1e-3, 1e-4],
    #     "number_epoch": [30, 50, 80, 100],
    #     "optimizer": ["adam", "sgd"],
    #     "weight_decay": [0, 1e-5, 1e-4],
    # }
        
        
    

    # all_combinations = list(itertools.product(
    #     param_grid["batch_size"],
    #     param_grid["learning_rate"],
    #     param_grid["number_epoch"],
    #     param_grid["optimizer"],
    #     param_grid["weight_decay"]
    # ))
    

    # print("Start UNET")
    
    # data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"

    # for i, (batch_size, lr, epochs, opt_name, wd) in enumerate(all_combinations):
    #     model_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/grid_models/unet_model_{i+1}.pth"
    #     image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/grid_models/unet_loss_plot_{i+1}.png"
        
    #     print(f"\n🔍 Grid Search [{i+1}/{len(all_combinations)}] - "
    #         f"BS={batch_size}, LR={lr}, Epochs={epochs}, Opt={opt_name}, WD={wd}")

    #     unet = UnetTrain(
    #         root_dir=data_dir,
    #         is_transform=False,
    #         learning_rate=lr,
    #         number_epoch=epochs,
    #         batch_size=batch_size,
    #         optimizer_name=opt_name,
    #         weight_decay=wd,
    #         image_output_path=image_output_path,
    #         model_output_path=model_path,
    #         model_path=""
    #     )
        
    #     unet.train_unet()


    # def objective(trial):
    #     data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"

    #     batch_size = trial.suggest_categorical("batch_size", [2,4,8])
    #     learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    #     number_epoch = trial.suggest_categorical("number_epoch", [30, 50])
    #     optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    #     weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    #     model_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_Moderate/unet_without_augmentation_Moderate_plus_Easy_model_1.pth"
    #     image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_Moderate/unet_without_augmentation_Moderate_plus_Easy_model_1.png"

    #     unet = UnetTrain(root_dir=data_dir, is_transform = False, learning_rate=learning_rate,number_epoch=number_epoch,batch_size=batch_size,optimizer_name=optimizer_name, weight_decay=weight_decay, image_output_path=image_output_path, model_output_path=model_output_path, model_path="")
    #     loss = unet.train_unet()
    #     return loss
    

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=20)

    # print("\nBest Hyperparameters:")
    # print(study.best_params)
    
    # learning_rate = 0.0004188691696239928
    # number_epoch = 50
    # batch_size = 2
    # optimizer_name = "adam"
    # weight_decay = 1.0816565013623374e-06
    
    # data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"
    # model_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_finetune/unet_without_augmentation__Easy_model_1.pth"
    # image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_finetune/unet_without_augmentation__Easy_model_1.png"

    # unet = UnetTrain(root_dir=data_dir, is_transform = False, learning_rate=learning_rate,number_epoch=number_epoch,batch_size=batch_size,optimizer_name=optimizer_name, weight_decay=weight_decay, image_output_path=image_output_path, model_output_path=model_output_path, model_path="")
    # loss = unet.train_unet()
    



    print("Start DeeplabV3")


    # def objective(trial):
    #     data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"

    #     batch_size = trial.suggest_categorical("batch_size", [2])
    #     learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    #     number_epoch = trial.suggest_categorical("number_epoch", [30, 50])
    #     optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    #     weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    #     model_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune_optim/deeplabv3_without_augmentation_Easy_model_1.pth"
    #     image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune_optim/deeplabv3_without_augmentation_Easy_model_1.png"

    #     deeplabv3_model = deepLabV3(data_dir=data_dir, model_output_path=model_output_path, image_output_path=image_output_path, model_path="",num_epochs=number_epoch, batch_size=batch_size, learning_rate=learning_rate,  optimizer_name=optimizer_name, weight_decay=weight_decay, is_transform= False)

    #     # start training 
    #     loss = deeplabv3_model.deeplabv3_train()
    
    #     return loss
    

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=20)

    # print("\nBest Hyperparameters:")
    # print(study.best_params)

    data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"
# Best Hyperparameters:
# {'batch_size': 2, 'learning_rate': 0.000983405046744849, 'number_epoch': 50, 'optimizer': 'sgd', 'weight_decay': 7.232961079009517e-05}
    model_path = ""
    learning_rate = 0.000983405046744849
    number_epoch = 50
    batch_size = 2
    optimizer_name = "sgd"
    weight_decay = 7.232961079009517e-05
    is_transform = False
    model_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune/deeplabv3_without_augmentation_Easy_model_1.pth"
    image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune/deeplabv3_without_augmentation_Easy_model_1.png"
    
    
    deeplabv3_model = deepLabV3(data_dir=data_dir, model_output_path=model_output_path, image_output_path=image_output_path, model_path="",num_epochs=number_epoch, batch_size=batch_size, learning_rate=learning_rate,  optimizer_name=optimizer_name, weight_decay=weight_decay, is_transform= False)

    # start training 
    deeplabv3_model.deeplabv3_train()



    # print("Start MaskRCNN")



    # def objective(trial):
    #     data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"
    #     num_classes = 2
    #     augmentation = False
    #     num_workers=2
    #     batch_size = trial.suggest_categorical("batch_size", [2])
    #     learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-3)
    #     number_epoch = trial.suggest_categorical("number_epoch", [30, 50])
    #     optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    #     weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    #     model_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Easy_finetune_optim/maskRCNN_without_augmentation_Easy_model_1.pth"
    #     image_output_path = f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Easy_finetune_optim/maskRCNN_without_augmentation_Easy_model_1.png"

    #     mask_rcnn_model = maskRCNN(batch_size=batch_size, num_classes = num_classes, learning_rate = learning_rate, augmentation=augmentation, number_epochs=number_epoch, optimizer_name= optimizer_name, weight_decay= weight_decay ,data_dir = data_dir, model_output_path=model_output_path, image_output_path=image_output_path, num_workers=num_workers)

    #     loss = mask_rcnn_model.train()
    
    #     return loss
    

    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=20)

    # print("\nBest Hyperparameters:")
    # print(study.best_params)

    # data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/train"

    
    # num_classes = 2
    # augmentation = False
    # num_workers=2
    # model_path = ""
    # learning_rate = study.best_params["learning_rate"]
    # number_epoch = study.best_params["number_epoch"]
    # batch_size = study.best_params["batch_size"]
    # optimizer_name = study.best_params["optimizer"]
    # weight_decay = study.best_params["weight_decay"]
    
    # model_output_path=f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Easy_finetune/maskRCNN_without_augmentation_Easy_model_1.pth"
    # image_output_path=f"/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Easy_finetune/maskRCNN_without_augmentation_Easy_model_1.png"

    # mask_rcnn_model = maskRCNN(batch_size=batch_size, num_classes = num_classes, learning_rate = learning_rate, augmentation=augmentation, number_epochs=number_epoch, optimizer_name= optimizer_name, weight_decay= weight_decay ,data_dir = data_dir, model_output_path=model_output_path, image_output_path=image_output_path, num_workers=num_workers)

    # mask_rcnn_model.train()