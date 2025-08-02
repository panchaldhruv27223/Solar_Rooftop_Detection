import os
import sys

current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_dir,"../.."))
sys.path.append(project_root)

# print(f"currnet directory : {current_dir}")
# print(f"parent direcotry : {project_root}")


import cv2
import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from Solar_rooftop_Detection_indian_images.accuracy_indian import compute_metrics
from Solar_rooftop_Detection_indian_images.baselineModels.UNET.Unet_model import UNet

from torch.utils.data import DataLoader, Dataset
# from torchvision.transforms import transforms


from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt




def iou_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-10) if union > 0 else 1.0

def f1_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(y_true * y_pred)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * (precision * recall) / (precision + recall + 1e-10)


## dataset for UNET
class RooftopDatasetUNET(Dataset):
    def __init__(self, root):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir =  os.path.join(root, "masks")
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

        self.transform = transforms.Compose([
                            transforms.ToTensor()
                        ])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = torch.from_numpy(image)
        # image = image.float()
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).unsqueeze(0)  # Ensure mask shape [1, H, W]
        
        return image, mask.squeeze(0), (img_path,mask_path)  # Ensure mask shape [H, W]
    


# dataset for DeepLabV3

class SegmentationDatasetDeeplabV3(Dataset):
    def __init__(self, root, image_folder="images", mask_folder="masks"):
        self.root = root
        self.transforms = transforms.Compose([
                            transforms.Resize((1024, 1024)),
                            transforms.ToTensor(),
                        ])
        self.image_folder = os.path.join(root, image_folder)
        self.mask_folder = os.path.join(root, mask_folder)
        self.image_names = sorted(os.listdir(self.image_folder))
        self.mask_names = sorted(os.listdir(self.mask_folder))

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
        return {"image": image, "mask": mask, "image_name": img_path, "mask_name":mask_path}
    




## dataset Loading for maskrcnn

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


class RooftopDatasetMaskRCNN(Dataset):
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
        # print("mask values : ",np.unique(mask))

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
                # print("mask values : ",np.unique(masks))
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


        return img, target,  (img_path,mask_path) 

    def __len__(self):
        return len(self.imgs)






class test_model:
    
    def __init__(self, model_path, output_path, test_data_dir, avg_accuracy_txt):
        self.model_path = model_path
        self.output_path = output_path
        self.test_data_dir = test_data_dir
        self.avg_accuracy_txt = avg_accuracy_txt

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## check if directory is exits or not 
        # os.makedirs(self.output_path, exist_ok=True)
        # os.makedirs(self.avg_accuracy_txt, exist_ok=True)
        
    

    def evaluate_unet_model(self, model, val_loader):
        model.eval()
        result_data = []
        # iou_scores = []
        with torch.no_grad():
            for images, masks, filenames in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                # print(images.shape)
                # print(masks.shape)
                # print(torch.unique(images))
                outputs = model(images)

                # print(masks.shape)
                # print(torch.unique(masks))
                # print(torch.unique(outputs))
                # print(outputs.shape)
                
                preds = (outputs > 0.5).float()
                # print(torch.unique(preds))
                for i in range(images.size(0)):
                    metrics = compute_metrics(preds[i], masks[i])
                    # print(metrics)
                    result_data.append([filenames[0][i],filenames[1][i]] + metrics)

                # print(result_data)
                # one  = preds[0].squeeze(0).cpu().numpy()
                # print(np.unique(one))
                # plt.imshow(one, cmap="gray")
                # plt.tight_layout()
                # plt.show()
                
                # original = masks[0].squeeze(0).cpu().numpy()
                # plt.imshow(original, cmap="gray")
                # plt.tight_layout()
                # plt.show()

                # metrics = compute_metrics(preds, masks)
                # result_data.append(metrics)
            
        return result_data

    def test_unet_model(self):
        my_model = UNet()
        my_model.to(device=self.device)
        my_model.load_state_dict(torch.load(self.model_path))
        
        # print(my_model)
        
        val_dataset = RooftopDatasetUNET(self.test_data_dir)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        accuracy_result = self.evaluate_unet_model(my_model, val_loader)
        
        # print(accuracy_result)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        metrics_df = pd.DataFrame(accuracy_result, columns=["image_path", "mask_path", "pixel_iou", "pixel_dice", "pixel_accuracy", "pixel_precision", "pixel_recall", "region_iou", "region_dice", "region_precision", "region_recall", "region_success_accuracy"])
        metrics_df.to_csv(self.output_path, index=False)

        avg_metrics = metrics_df.iloc[:,2:].mean()
        # print(avg_metrics)

        os.makedirs(os.path.dirname(self.avg_accuracy_txt), exist_ok=True)

        with open(self.avg_accuracy_txt, "w") as f:
            f.write("Average Metrics:\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

        return 
    





    ### DeeplabV3
    def evaluate_model_deeplabv3(self, model, dataloader, output_path, avg_accuracy_txt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set to evaluation mode

        total_loss = 0.0
        total_iou = 0.0
        total_f1 = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()  # Same loss as training
        num_samples = 0
        all_matrics = []

        with torch.no_grad():  # No gradient computation
            for sample in tqdm(dataloader):
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)
                image_name = sample["image_name"]
                mask_name = sample["mask_name"]
                # print(image_name)
                # print(mask_name)
                
            
                outputs = model(inputs)["out"]  # Shape: (batch, 1, H, W)
                loss = criterion(outputs, masks)
                total_loss += loss.item() * inputs.size(0)

                # Convert logits to binary predictions
                
                preds = torch.sigmoid(outputs) > 0.5  # Threshold at 0.5
                preds_ = preds
                masks_ = masks
                preds = preds.cpu().numpy().astype(np.uint8)
                masks = masks.cpu().numpy().astype(np.uint8)

                # Compute metrics per batch
                for i in range(inputs.size(0)):
                    all_matrics.append([image_name[i],mask_name[i]]+compute_metrics(preds_[i], masks_[i]))
                    # Compute IoU and F1 score
                    total_iou += iou_score(masks[i], preds[i])
                    total_f1 += f1_score(masks[i], preds[i])
                num_samples += inputs.size(0)

        avg_loss = total_loss / num_samples
        avg_iou = total_iou / num_samples
        avg_f1 = total_f1 / num_samples
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df = pd.DataFrame(all_matrics, columns=["image_path", "mask_path", "pixel_iou", "pixel_dice", "pixel_accuracy", "pixel_precision", "pixel_recall","region_iou", "region_dice", "region_precision", "region_recall", "region_success_accuracy"])

        df.to_csv(output_path, index=False)

        avg_metrics = df.iloc[:,2:].mean()
        print(avg_metrics)

        os.makedirs(os.path.dirname(avg_accuracy_txt), exist_ok=True)

        
        with open(self.avg_accuracy_txt, "w") as f:
            f.write("Average Metrics:\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

            f.write(f"avg_loss: {avg_loss:.4f}\n")
            f.write(f"avg_iou: {avg_iou:.4f}\n")
            f.write(f"avg_f1: {avg_f1:.4f}\n")

        return

    def test_deeplabv3_model(self):
        
        dataset = SegmentationDatasetDeeplabV3(self.test_data_dir)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = torch.load(self.model_path, weights_only=False)

        self.evaluate_model_deeplabv3(model, dataloader,  self.output_path, self.avg_accuracy_txt)
        
        
        
        
        

    
    def calculate_maskRcnn_iou(self,mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    

    def calculate_maskRcnn_box_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
        xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0


    def load_finetuned_maskRcnn_model(self, model_path):
        my_model = torch.load(model_path, weights_only=False)
        my_model.to(self.device)
        my_model.eval()
        return my_model
    
    def evaluate_maskRcnn_model(self, model, data_loader, output_csv_path, output_txt_path, iou_threshold=0.5, conf_threshold=0.5):
        """Evaluate a Mask R-CNN model on test data, computing precision, recall, and F1-score for boxes and masks."""
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        # print(output_csv_path)
        # print(output_txt_path)

        
        all_tp_boxes, all_fp_boxes, all_fn_boxes = 0, 0, 0
        all_tp_masks, all_fp_masks, all_fn_masks = 0, 0, 0
        all_metrics = []
        
        
        with torch.no_grad():   
            
            for batch_idx, (images, targets, paths) in enumerate(data_loader):
            
                # print(f"\nProcessing batch {batch_idx + 1}/{len(data_loader)}...")
                # print(f"Number of images in batch: {len(images)}")
                images = [img.to(self.device) for img in images]
                # print(f"Image shape (first in batch): {images[0].shape}")

                # Model inference
                outputs = model(images)  # List of dicts, one per image
                # print(f"Outputs length: {len(outputs)}, Keys: {outputs[0].keys()}")
                # print(f"Output masks shape: {outputs[0]['masks'].shape}")
                # print(f"Output boxes shape: {outputs[0]['boxes'].shape}")

                # print(f"Targets length: {len(targets)}, Keys: {targets[0].keys()}")
                # print(f"Target masks shape: {targets[0]['masks'].shape}")
                # print(f"Target boxes shape: {targets[0]['boxes'].shape}")
                # loss = model(images, targets)
                # print(f"Loss: {loss}")
                # break
                # Process each image in the batch
                # print(len(images))
                # print(len(paths))
            
                for output, target, path in zip(outputs, targets, paths):
                
                    # Ground truth
                    gt_boxes = target["boxes"].cpu().numpy()  # [N_gt, 4]

                    gt_masks = target["masks"].cpu().numpy()  # [N_gt, H, W]

                    gt_labels = target["labels"].cpu().numpy()  # [N_gt]

                    
                    # Predictions
                    pred_boxes = output["boxes"].cpu().numpy()  # [N_pred, 4]
                    pred_scores = output["scores"].cpu().numpy()  # [N_pred]
                    pred_masks = output["masks"].squeeze(1).cpu().numpy()  # [N_pred, 1, H, W] -> [N_pred, H, W]
                    pred_labels = output["labels"].cpu().numpy()  # [N_pred]

                    # Filter predictions by confidence threshold
                    mask = pred_scores >= conf_threshold
                    pred_boxes = pred_boxes[mask]
                    pred_scores = pred_scores[mask]
                    pred_masks = pred_masks[mask]
                    pred_labels = pred_labels[mask]
                    
                    # print(gt_masks.shape)
                    # print(pred_masks.shape)
                

                    # Evaluate bounding boxes
                    matched_gt = set()
                    tp_boxes, fp_boxes = 0, 0
                    for pred_box in pred_boxes:
                        best_iou = 0
                        best_gt_idx = -1
                        for i, gt_box in enumerate(gt_boxes):
                            if i in matched_gt:
                                continue
                            iou = self.calculate_maskRcnn_box_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = i
                        if best_iou >= iou_threshold:
                            tp_boxes += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            fp_boxes += 1
                    fn_boxes = len(gt_boxes) - len(matched_gt)
                    all_tp_boxes += tp_boxes
                    all_fp_boxes += fp_boxes
                    all_fn_boxes += fn_boxes

                    # Evaluate masks and compute metrics
                    matched_gt_masks = set()
                    tp_masks, fp_masks = 0, 0
                    M = []
                    for pred_idx, pred_mask in enumerate(pred_masks):
                        best_iou = 0
                        best_gt_idx = -1
                        for gt_idx, gt_mask in enumerate(gt_masks):
                            if gt_idx in matched_gt_masks:
                                continue
                            iou = self.calculate_maskRcnn_iou(pred_mask, gt_mask)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        if best_iou >= iou_threshold:
                            tp_masks += 1
                            matched_gt_masks.add(best_gt_idx)
                            # Compute metrics for this true positive pair
                            
                            pred_mask_tensor = torch.from_numpy((pred_mask).astype(np.uint8))
                            gt_mask_tensor = torch.from_numpy(gt_mask.astype(np.uint8))

                            # print("predicted mask")
                            # print(pred_mask_tensor.size())
                            # print(torch.unique(pred_mask_tensor))
                            # print("gorund truth mask")
                            # print(gt_mask_tensor.size())
                            # print(torch.unique(gt_mask_tensor))

                            metrics = compute_metrics(pred_mask_tensor, gt_mask_tensor)
                            M.append(metrics)
                        else:
                            fp_masks += 1
                            
                    fn_masks = len(gt_masks) - len(matched_gt_masks)
                    all_tp_masks += tp_masks
                    all_fp_masks += fp_masks
                    all_fn_masks += fn_masks
                    # print(M)
                    M_mean = np.mean(M, axis=0) if M else [0] * 10
                    # print(M_mean)
                    # print(len(M_mean))
                    # print(M_mean.shape)
                    all_metrics.append([path[0], path[1]] + list(M_mean))

        matrix_df = pd.DataFrame(all_metrics, columns=["image_path", "mask_path","pixel_iou", "pixel_dice", "pixel_accuracy", "pixel_precision", "pixel_recall",
                                                    "region_iou", "region_dice", "region_precision", "region_recall", "region_success_accuracy"])     
        # print(matrix_df.head(5))
        matrix_df.to_csv(output_csv_path, index=False) 
        
        # Compute average metrics for true positive mask pairs
        avg_metrics = matrix_df.iloc[:,2:].mean()
        print("Average :")
        print(avg_metrics)
            
            
        # Compute detection metrics for boxes
        precision_boxes = all_tp_boxes / (all_tp_boxes + all_fp_boxes) if (all_tp_boxes + all_fp_boxes) > 0 else 0
        recall_boxes = all_tp_boxes / (all_tp_boxes + all_fn_boxes) if (all_tp_boxes + all_fn_boxes) > 0 else 0
        f1_boxes = 2 * (precision_boxes * recall_boxes) / (precision_boxes + recall_boxes) if (precision_boxes + recall_boxes) > 0 else 0

        # Compute detection metrics for masks
        precision_masks = all_tp_masks / (all_tp_masks + all_fp_masks) if (all_tp_masks + all_fp_masks) > 0 else 0
        recall_masks = all_tp_masks / (all_tp_masks + all_fn_masks) if (all_tp_masks + all_fn_masks) > 0 else 0
        f1_masks = 2 * (precision_masks * recall_masks) / (precision_masks + recall_masks) if (precision_masks + recall_masks) > 0 else 0
        
        
        print(output_txt_path)
        with open(output_txt_path, "w") as f:
            f.write("Average Metrics:\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write(f"precision_boxes: {precision_boxes:.4f}\n")
            f.write(f"recall_boxes: {recall_boxes:.4f}\n")
            f.write(f"f1_boxes: {f1_boxes:.4f}\n")

            f.write(f"precision_masks: {precision_masks:.4f}\n")
            f.write(f"recall_masks: {recall_masks:.4f}\n")
            f.write(f"f1_masks: {f1_masks:.4f}\n")



    def test_mask_rcnn_model(self):

        def custom_collate_fn(batch):
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            paths   = [item[2] for item in batch]
            return images, targets, paths

        test_dataset = RooftopDatasetMaskRCNN(self.test_data_dir)
        
        model = self.load_finetuned_maskRcnn_model(self.model_path)
        # print(model)

        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

        iou_threshold = 0.5
        conf_threshold= 0.5

        # # Evaluate
        self.evaluate_maskRcnn_model(model, data_loader = test_loader, output_csv_path = self.output_path, output_txt_path=self.avg_accuracy_txt, iou_threshold=iou_threshold , conf_threshold=conf_threshold)


if __name__ == "__main__":
    print("Dhruv Panchal")
    

    # UNET
    ## modrate dataset
    test_data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/dense_data/Original/test"


    # ## USA dataset
    # # test_data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Arial_validation_images"


    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/moderate/unet_without_augmentation_moderate_model_1.pth"
    
    model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_finetune/train_dense_unet_without_augmentation_E_70.pth"

    output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_finetune/train_test_dense_unet_without_augmentation_E_70.csv"

    avg_accuracy_txt = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/UNET/Easy_finetune/train_test_dense_unet_without_augmentation_E_70.txt"
    
    model_testing = test_model(model_path, output_path, test_data_dir, avg_accuracy_txt)
    model_testing.test_unet_model()
    
    # print("UNET DONE")


    # DEEPLABv3

    # test_data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final-moderate/original/test"
    
    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/moderate/deeplabv3_without_augmentation_moderate_model_1.pth"
    
    model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune/train_dense_deeplabv3_without_augmentation_E_70.pth"
    
    output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune/train_test_dense_deeplabv3_without_augmentation_E_70.csv"
    
    avg_accuracy_txt = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/deeplabv3/Easy_finetune/train_test_dense_deeplabv3_without_augmentation_E_70.txt"

    model_testing = test_model(model_path, output_path, test_data_dir, avg_accuracy_txt)
    model_testing.test_deeplabv3_model()

    print("DEEPLAB V3 DONE")

    # MASKRcnn

    # test_data_dir = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final_easy_dataset/original/test"
    
    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/moderate/maskRCNN_without_augmentation_moderate_model_1.pth"
    # model_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Easy_finetune/maskRCNN_without_augmentation_Easy_model_1.pth"
    
    # output_path = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Gold_standard/train_Easy_test_GOLD_without_augmentation_full_data.csv"
    
    # avg_accuracy_txt = "/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/baselineModels/maskrcnn/Gold_standard/train_Easy_test_GOLD_without_augmentation_full_data.txt"

    # model_testing = test_model(model_path, output_path, test_data_dir, avg_accuracy_txt)
    # model_testing.test_mask_rcnn_model()
    
    # print("Mask RCNN DONE")