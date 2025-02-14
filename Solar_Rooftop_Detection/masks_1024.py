import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import os


def get_1024_masks(patch_file_path, counter, patch_mask_size, target_mask_path):
    mask_patch_image =  cv2.imread(patch_file_path, cv2.IMREAD_UNCHANGED)
    print(mask_patch_image.shape)
    print(counter)

    patch_masks = []
    patch_size = patch_mask_size

    for i in range(0, mask_patch_image.shape[0], patch_size):
        for j in range(0, mask_patch_image.shape[1], patch_size):
            patch = mask_patch_image[i:i+patch_size, j:j+patch_size]
            
            if (patch.shape[0] == patch_mask_size and patch.shape[1] == patch_mask_size):
                patch_masks.append(patch)

    # print(len(patch_masks))
    patch_masks = np.array(patch_masks)

    for i in range(len(patch_masks)):
        img = patch_masks[i]
        file_name = target_mask_path+"/"+str(counter)+".jpg"
        cv2.imwrite(file_name, img)
        counter += 1
        
    return counter
        

if __name__ == "__main__":
    
    original_tiff_mask_files_path = "/home/student/Documents/Arpit_sir/SOLAR/CodeFiles/dhruv/AerialImageDataset/train/gt"
    target_mask_path = "/home/student/Documents/Arpit_sir/SOLAR/CodeFiles/dhruv_git/Solar_Rooftop_Detection/Arial_images_1024_1024/masks"
    
    # print(os.listdir(original_tiff_files_path))
    counter = 0
    pictures_list = os.listdir(original_tiff_mask_files_path)
    pictures_list.sort()
    print(len(pictures_list))
    for i in pictures_list:
        tiff_image_path = original_tiff_mask_files_path+"/"+i
        counter = get_1024_masks(tiff_image_path, counter, 1024, target_mask_path)

    print("Task is Done")