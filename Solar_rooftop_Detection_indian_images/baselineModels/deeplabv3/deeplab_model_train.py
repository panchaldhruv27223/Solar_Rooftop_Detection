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