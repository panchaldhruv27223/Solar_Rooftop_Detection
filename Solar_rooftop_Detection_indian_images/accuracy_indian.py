import torch 
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, accuracy_score, precision_score, recall_score

def calculate_mask_accuracy(ground_truth: torch.Tensor, predicted: torch.Tensor):
    ## it take two tensor, 1 : ground_truth mask and 2 : predicted mask 
    
    gt_binary = (ground_truth == 255).to(torch.uint8)
    pred_binary = (predicted == 255).to(torch.uint8)

    # Compute the number of correctly predicted pixels
    correct_pixels = (gt_binary == pred_binary).sum().item()

    # Compute total number of pixels
    total_pixels = ground_truth.numel()

    # Compute accuracy
    accuracy = correct_pixels / total_pixels
    return accuracy


def compute_metrics(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0).to(torch.uint8)
    gt_mask = (gt_mask > 0).to(torch.uint8)

    pred_flat = pred_mask.flatten().cpu().numpy()  
    gt_flat = gt_mask.flatten().cpu().numpy()      
    
    pixel_iou = jaccard_score(gt_flat, pred_flat, average='binary')
    pixel_dice = f1_score(gt_flat, pred_flat, average='binary')
    pixel_accuracy = accuracy_score(gt_flat, pred_flat)
    pixel_precision = precision_score(gt_flat, pred_flat, zero_division=1)
    pixel_recall = recall_score(gt_flat, pred_flat, zero_division=1)

    # Region-level metrics (single region)
    intersection = torch.sum(pred_mask & gt_mask).item()  # Overlapping pixels
    pred_area = torch.sum(pred_mask).item()              # Predicted region area
    gt_area = torch.sum(gt_mask).item()                  # Ground truth region area
    union = pred_area + gt_area - intersection           # Union of the regions

    # Handle edge cases (e.g., empty masks)
    region_iou = intersection / union if union > 0 else 1.0
    region_dice = 2 * intersection / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 1.0
    region_precision = intersection / pred_area if pred_area > 0 else 1.0
    region_recall = intersection / gt_area if gt_area > 0 else 1.0
    region_success_accuracy = 1.0 if region_iou > 0.5 else 0.0  # Binary detection success (IoU > 0.5)

    # Return all metrics
    return [float(pixel_iou), pixel_dice, pixel_accuracy, pixel_precision, pixel_recall, 
            region_iou, region_dice, region_precision, region_recall, region_success_accuracy]

if __name__ == "__main__":
    pred_mask = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]]) 
    gt_mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])   
    pred_mask = torch.tensor(pred_mask)
    gt_mask = torch.tensor(gt_mask)
    
    metrics = compute_metrics(pred_mask, gt_mask)
    print(metrics)