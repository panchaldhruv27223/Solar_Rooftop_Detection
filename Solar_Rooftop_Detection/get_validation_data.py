import os
import shutil
import random


def create_validation_data_folder(data_images_dir, data_masks_dir, val_images_dir, val_masks_dir):
    
    # Create validation directories if they don't exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(data_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Calculate number of validation images (20%)
    total_images = len(image_files)
    val_size = int(total_images * 0.2)

    # Randomly select indices
    random_indices = random.sample(range(total_images), val_size)

    # Move files based on random indices
    for idx in random_indices:
        # Get the image file name
        image_file = image_files[idx]
        
        # Construct full paths for source and destination
        src_image_path = os.path.join(data_images_dir, image_file)
        dst_image_path = os.path.join(val_images_dir, image_file)
        
        # Get the mask file name
        src_mask_path = os.path.join(data_masks_dir, image_file)
        dst_mask_path = os.path.join(val_masks_dir, image_file)
        
        # Move image, mask
        
        if os.path.exists(src_mask_path):
            shutil.move(src_image_path, dst_image_path)
        else:
            print(f"Warning: Mask {image_file} not found in {data_images_dir}")
       
       
        if os.path.exists(src_mask_path):
            shutil.move(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: Mask {image_file} not found in {data_masks_dir}")

    print(f"Completed! Moved {val_size} image-mask pairs to validation folder.")
    print(f"Training set remaining: {total_images - val_size} pairs.")
    
    
if __name__ == "__main__":
    
    # Define paths
    
    data_images_dir = '/home/selc-a4-sr2/Solar_Rooftop_Detection/Arial_images_1024_1024/images'
    data_masks_dir = '/home/selc-a4-sr2/Solar_Rooftop_Detection/Arial_images_1024_1024/masks'
    val_images_dir = '/home/selc-a4-sr2/Solar_Rooftop_Detection/Arial_validation_images/images'
    val_masks_dir = '/home/selc-a4-sr2/Solar_Rooftop_Detection/Arial_validation_images/masks'
    create_validation_data_folder(data_images_dir, data_masks_dir, val_images_dir, val_masks_dir)