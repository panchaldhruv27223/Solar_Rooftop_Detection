import torch 
import numpy as np


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

