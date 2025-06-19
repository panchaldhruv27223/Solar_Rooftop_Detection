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

import sys
current_dir = os.path.dirname(__file__)
# print(current_dir)

project_root = os.path.abspath(os.path.join(current_dir,"../.."))
# print(project_root)

sys.path.append(project_root)
from Solar_rooftop_Detection_indian_images.accuracy_indian import compute_metrics

# from accuracy_indian import compute_metrics



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
class RooftopDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
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
        img = T.ToTensor()(img)  # Shape: (3, 1024, 1024)

        # Load multi-instance mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # print(f"Warning: Mask failed to load for {mask_path}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, 1024, 1024), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
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
                masks = torch.from_numpy(masks).to(dtype=torch.uint8)
                
                # Convert boxes to tensor
                boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Shape: (num_instances, 4)
                
                # Assign labels (all 1 for rooftops)
                labels = torch.ones((len(instance_masks),), dtype=torch.int64)
                
                # Debug info
                # print(f"{img_path}: {len(instance_masks)} rooftops")

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    


# Mask RCNN model.

class maskRCNN:
    def __init__(self, batch_size=4, num_classes = 2, learning_rate = 0.0001, number_epochs=50, data_dir :str ="", model_output_path : str="", image_output_path : str="",  model_path : str ="", num_workers=2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_epochs = number_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.root_dir = data_dir
        self.model_path = model_path
        self.model_output_path = model_output_path
        self.image_output_path = image_output_path


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

        dataset = RooftopDataset(self.root_dir)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=lambda x: tuple(zip(*x)))

        print("Data is loaded")

        if self.model_path : 
            model = self.load_finetuned_model(self.model_path)
            print("loading pre trian model")
        else:
            print("load model")
            model = self.get_model_instance_segmentation(num_classes=self.num_classes)
            model.to(self.device)
            model.train()


        # optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # loss = self.train_model(model, data_loader, optimizer, self.device, num_epochs=self.num_epochs)

        # torch.save(model, self.model_output_path)

        # print(f"Model saved !.")
              
              
        # epochs = list(range(1, 51))
        # maskrcnn_loss = loss
        # plt.figure(figsize=(10, 6))
        # plt.plot(epochs, maskrcnn_loss, label='Mask R-CNN Loss', color='green')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Loss vs Epoch')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(self.image_output_path)
        # plt.tight_layout()
        # plt.show()


    


# Define the dataset class For deeplabv3
class SegmentationDataset(Dataset):
    def __init__(self, root, image_folder="images", mask_folder="masks", transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_folder = os.path.join(root, image_folder)
        self.mask_folder = os.path.join(root, mask_folder)
        self.image_names = sorted(os.listdir(self.image_folder))[:-1]
        self.mask_names = sorted(os.listdir(self.mask_folder))[:-1]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_names[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_names[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        if self.transforms:
            image = self.transforms(image)
            mask = transforms.ToTensor()(mask)  # Mask to tensor (0-1 range)
        return {"image": image, "mask": mask}
    


## deeplabv3 model
class deepLabV3:

    def __init__(self, data_dir :str , model_output_path : str, image_output_path : str,  model_path : str ="", num_epochs : int = 50, batch_size :int = 2, learning_rate : float = 1e-4):
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        self.model_output_path = model_output_path
        self.image_output_path = image_output_path
        self.model_path = model_path
        self.number_epoch = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


    def train_model(self, model, dataloader, num_epochs=50, learning_rate = 1e-4):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()  # For binary segmentation
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_list = []
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            for sample in tqdm(dataloader):
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)["out"]  # Shape: (batch, 1, H, W)
                # print(outputs.shape)
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

        dataset = SegmentationDataset(self.data_dir, transforms= self.transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        
        if self.model_path : 
            print("Loading The Pretrain Model")
            model = torch.load(self.model_path, weights_only=False)
        else:
            print("Define the model Architecture.")
            model = self.create_deeplab(output_channels=1)  
            print(model)

        trained_model, losses = self.train_model(model, dataloader, num_epochs=self.number_epoch, learning_rate=self.learning_rate)

        torch.save(trained_model, self.model_output_path)

        deeplabv3_loss = losses
        epochs = list(range(1,self.number_epoch+1))


        plt.figure(figsize=(10, 6))
        plt.plot(epochs, deeplabv3_loss, label='DeepLabV3 Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.image_output_path)
        plt.tight_layout()
        plt.show()
            









### UNET MODEL 
# Dataset for UNET Model
#  
class RooftopDataset(Dataset):
    def __init__(self, root, image_folder="images", mask_folder="masks", transform=None):
        self.root = root
        self.image_dir = os.path.join(self.root, image_folder)
        self.mask_dir = os.path.join(root, mask_folder)
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.image_masksnames = sorted(os.listdir(self.mask_dir))

        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_masksnames[idx])
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = mask.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).unsqueeze(0)  # Ensure mask shape [1, H, W]
        
        return image, mask.squeeze(0)  # Ensure mask shape [H, W]


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
    def __init__(self, root_dir, learning_rate=1e-4, number_epoch=50, image_output_path="", model_output_path="", model_path=""):
        
        self.root_dir = root_dir
        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.image_output_path = image_output_path
        self.model_output_path = model_output_path
        self.model_path = model_path

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
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            losses.append(epoch_loss/len(train_loader))
        
        torch.save(model.state_dict(), self.model_output_path)        
        print("Model saved!")

        return losses 


    def train_unet(self):
        # Load datasets
        train_dataset = RooftopDataset(self.root_dir, image_folder="images", mask_folder="masks",transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

        model = UNet()
        
        # Training Setup
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))
            print("load pretrain model")
        
        print("define new model")
        
        model = model.to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        loss = self.train(model, train_loader, criterion= criterion, optimizer=optimizer,epochs=self.number_epoch)
        
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
        plt.show()



if __name__ == "__main__":

    print("Done")
    # data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train" 
    # model_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/deeplab_rooftop_full_50_indian_usa.pth"
    # image_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/deeplabv3_loss_vs_epochs_usa.png"
    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/BaseLineModels/deeplabv3/deeplab_rooftop_full_50.pth"
    # num_epochs= 50
    # batch_size = 2
    # learning_rate = 1e-4
    
    
    # deeplabv3_model = deepLabV3(data_dir=data_dir, model_output_path=model_output_path, image_output_path=image_output_path, model_path= model_path,num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

    ## start training 
    # deeplabv3_model.deeplabv3_train()



    # data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train" 
    # model_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/unet_rooftop_full_50_indian_usa.pth"
    # image_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/unet_loss_vs_epochs_usa.png"
    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/unet_rooftop_50_indian_usa.pth"
    # num_epochs= 50
    # batch_size = 2
    # learning_rate = 1e-4

    # unet = UnetTrain(root_dir=data_dir, learning_rate=learning_rate,number_epoch=num_epochs,image_output_path=image_output_path, model_output_path=model_output_path, model_path=model_path)
    # unet.train_unet()




    root_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train"
    batch_size=4
    num_classes = 2
    learning_rate = 0.0001
    number_epochs=50
    model_output_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/maskrcnn_model_scratch.pth"
    image_output_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/loss_image_maskrcnn.png"
    num_workers=2
    model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/BaseLineModels/Maksed_RCNN/mask_rcnn_epoch_50_full.pth"

    mask_rcnn_model = maskRCNN(batch_size=batch_size, num_classes = num_classes, learning_rate = learning_rate, number_epochs=number_epochs, data_dir =root_dir, model_output_path=model_output_path, image_output_path=image_output_path, num_workers=num_workers)

    mask_rcnn_model.train()