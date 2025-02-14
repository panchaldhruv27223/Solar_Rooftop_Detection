import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import os 


def get_1024_images(patch_file_path, counter, patch_image_size, target_image_path):
    img = tiff.imread(patch_file_path)
    # (img.shape)
    patch_images = []
    pat_image_size = patch_image_size
        
    for i in range(0, img.shape[0], pat_image_size):
        for j in range(0, img.shape[1], pat_image_size):
            patch = img[i:i+pat_image_size, j:j+pat_image_size, :]
            
            if (len(patch) == patch_image_size and len(patch[1]) == patch_image_size):
                patch_images.append(patch)

    # print(len(patch_images))
    patch_images = np.array(patch_images)
    

    for i in range(len(patch_images)):
        img = patch_images[i]
        file_name = target_image_path+"/"+str(counter)+".jpg"
        counter += 1
        cv2.imwrite(file_name, img)
        
    return counter


if __name__ == "__main__":
    original_tiff_files_path = "/home/student/Documents/Arpit_sir/SOLAR/CodeFiles/dhruv/AerialImageDataset/train/images"
    target_image_path = "/home/student/Documents/Arpit_sir/SOLAR/CodeFiles/dhruv_git/Solar_Rooftop_Detection/Arial_images_1024_1024/images"
    
    # print(os.listdir(original_tiff_files_path))
    counter = 0
    pictures_list = os.listdir(original_tiff_files_path)
    pictures_list.sort()
    for i in (pictures_list):
        tiff_image_path = original_tiff_files_path+"/"+i
        counter = get_1024_images(tiff_image_path, counter, 1024, target_image_path)

    print("Task is Done")