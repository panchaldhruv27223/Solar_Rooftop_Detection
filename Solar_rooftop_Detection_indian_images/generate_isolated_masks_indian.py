import torch
import numpy as np
import cv2
import os

def generate_isolated_mask(mask, box):
    """"
    input = "mask" and "coordinates of bounding box"
    output = "Isolated mask.
    """
    x1, y1, x2, y2 = box
    # print(x1, y1, x2, y2)

    isloated_mask = torch.zeros_like(mask)
    
    # print(mask[y1: y2, x1:x2].shape)
    
    isloated_mask[y1: y2, x1:x2] = mask[y1: y2, x1:x2]
    
    # print(np.unique(isloated_mask.numpy()))
    
    # cv2.imshow("new_mask",isloated_mask.numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print(isloated_mask.shape)

    return isloated_mask

def extract_fullsize_instance_masks_and_bboxes(instance_mask, target_size=(1024, 1024)):
    """
    Given an instance mask, extract binary masks (same size) and their bounding boxes.
    
    Args:
        instance_mask (np.ndarray): 2D array with 0 as background and unique IDs for objects.
        target_size (tuple): The output binary mask shape (default 1024x1024)
    
    Returns:
        masks_tensor (np.ndarray): [N, H, W] binary mask tensor for N objects
        bboxes (List[List[int]]): list of bounding boxes [x_min, y_min, x_max, y_max]
    """
    H, W = target_size
    object_ids = np.unique(instance_mask)
    object_ids = object_ids[object_ids != 0]  # Exclude background

    num_objects = len(object_ids)
    masks_tensor = np.zeros((num_objects, H, W), dtype=np.uint8)
    bboxes = []

    for idx, obj_id in enumerate(object_ids):
        # Binary mask
        mask = (instance_mask == obj_id).astype(np.uint8)

        # Store full-sized binary mask
        masks_tensor[idx] = mask

        # Get bounding box
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            bboxes.append([0, 0, 0, 0])  # fallback
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bboxes.append((x_min, y_min, x_max, y_max))

    return masks_tensor, bboxes



if __name__ == "__main__":
    print("Hello from Dhruv Panchal.")