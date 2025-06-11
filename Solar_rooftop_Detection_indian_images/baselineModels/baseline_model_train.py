import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from accuracy_indian import compute_metrics
import cv2
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



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
    


# Define the dataset For deeplabv3
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
    

def train_model(model, dataloader, num_epochs=50, learning_rate = 1e-4):
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


def create_deeplab(output_channels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier = DeepLabHead(2048, output_channels)
    return model


def deeplabv3_train(data_dir :str , model_output_path : str, image_output_path : str,  model_path : str ="", num_epochs : int = 50, batch_size :int = 2, learning_rate : float = 1e-4):

    data_dir = data_dir

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(data_dir, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    if model_path : 
        print("Loading The Pretrain Model")
        model = torch.load(model_path, weights_only=False)
    else:
        print("Define the model Architecture.")
        model = create_deeplab(output_channels=1)  
        print(model)

    # trained_model, losses = train_model(model, dataloader, num_epochs=num_epochs, learning_rate=1e-4)

    # torch.save(trained_model, model_output_path)

    # deeplabv3_loss = losses
    # epochs = list(range(1,num_epochs+1))


    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, deeplabv3_loss, label='DeepLabV3 Loss', color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Epoch')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(image_output_path)
    # plt.tight_layout()
    # plt.show()






if __name__ == "__main__":
    print("Done")
    data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/train" 
    model_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/deeplab_rooftop_full_50_indian_usa.pth"
    image_output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/deeplabv3_loss_vs_epochs_usa.png"
    model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/BaseLineModels/deeplabv3/deeplab_rooftop_full_50.pth"
    num_epochs= 50
    batch_size = 2
    learning_rate = 1e-4
    
    deeplabv3_train(data_dir=data_dir, model_output_path=model_output_path, image_output_path=image_output_path, model_path= model_path,num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)