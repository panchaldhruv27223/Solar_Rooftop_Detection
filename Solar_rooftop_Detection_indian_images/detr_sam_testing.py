import os 
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from accuracy_indian import compute_metrics
from generate_isolated_masks_indian import generate_isolated_mask
from rfdetr import RFDETRBase

class DETRSAM_Evaluator:
    def __init__(self, detr_model_path, sam_model_path, 
                 data_folder, mask_folder, 
                 output_csv_path, output_txt_path,
                 sam_model_type="vit_b",
                 detr_threshold=0.5):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.output_csv_path = output_csv_path
        self.output_txt_path = output_txt_path
        self.detr_threshold = detr_threshold

        # Load DETR
        self.detr_model = self.load_detr(detr_model_path)

        # Load SAM
        sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_txt_path), exist_ok=True)

        self.results_data = []

    def load_detr(self, checkpoint_path):
        model_wrapper = RFDETRBase()
        model_wrapper.model.reinitialize_detection_head(num_classes=2)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        pytorch_model = model_wrapper.model.model
        pytorch_model.load_state_dict(checkpoint['model'])
        pytorch_model.to(self.device)
        pytorch_model.eval()

        return model_wrapper

    def run(self):
        image_list = sorted(os.listdir(self.data_folder))
        mask_list = sorted(os.listdir(self.mask_folder))

        for img_name, mask_name in zip(image_list, mask_list):
            print(f"Processing {img_name}...")

            img_path = os.path.join(self.data_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)

            image = cv2.imread(img_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = torch.from_numpy(gt_mask).to(self.device)

            # DETR Prediction
            detections = self.detr_model.predict(image, threshold=self.detr_threshold)
            boxes = detections.xyxy

            self.sam_predictor.set_image(image)

            if len(boxes) == 0:
                print(f"No detections for {img_name}")
                continue

            H, W = (1024, 1024)
            
            final_pred_mask = torch.zeros((H, W), dtype=torch.int, device=self.device)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]))
                pred_mask = masks[0].astype("int")
                pred_mask = torch.from_numpy(pred_mask).to(self.device)

                pred_mask_cropped = pred_mask[y1:y2, x1:x2]

                final_pred_mask[y1:y2, x1:x2] = torch.logical_or(
                    final_pred_mask[y1:y2, x1:x2], pred_mask_cropped
                )

            metrics_list = compute_metrics(final_pred_mask, gt_mask)

            metric_keys = [
                "pixel_iou", "pixel_dice", "pixel_accuracy", "pixel_precision", "pixel_recall",
                "region_iou", "region_dice", "region_precision", "region_recall", "region_success_accuracy"
            ]
            metrics = dict(zip(metric_keys, metrics_list))

            result = {
                "image_path": img_path,
                "mask_path": mask_path,
                "num_boxes": len(boxes),
                **metrics
            }

            self.results_data.append(result)

        self.save_results()

    def save_results(self):
        df = pd.DataFrame(self.results_data)
        df.to_csv(self.output_csv_path, index=False)
        print(f"Saved CSV results to {self.output_csv_path}")

        with open(self.output_txt_path, "w") as f:
            f.write("\n\n==== Metrics Mean ====\n")
            f.write(df.mean(numeric_only=True).to_string())
            f.write("\n\n==== Metrics Std ====\n")
            f.write(df.std(numeric_only=True).to_string())
        print(f"Saved summary TXT to {self.output_txt_path}")


if __name__ == "__main__":
    
    evaluator = DETRSAM_Evaluator(
        detr_model_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/indian_moderate_weights/detr_moderate.pth",
        
        sam_model_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/only_moderate_SAM_scratch_with_aug_on all data_version_2/Final_Model/SAM_final_04_07_2025.pth",
        
        data_folder="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/test/images",
        
        mask_folder="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/gandhinagar_dataset/test/masks",
        
        output_csv_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/DETR_sam_moderate/DETR_only_moderate_SAM_scratch_with_aug_on_all_data_version_2.csv",
        
        output_txt_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/DETR_sam_moderate/DETR_only_moderate_SAM_scratch_with_aug_on_all_data_version_2.txt"

    )
    
    evaluator.run()