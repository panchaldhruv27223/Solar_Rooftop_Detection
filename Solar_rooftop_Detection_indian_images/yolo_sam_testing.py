import torch
import cv2
import pandas as pd
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
import os 
from accuracy_indian import compute_metrics
from generate_isolated_masks_indian import generate_isolated_mask

class YOLOSAM_Evaluator:
    def __init__(self, yolo_model_path, sam_model_path, 
                 data_folder, mask_folder, 
                 output_csv_path, output_txt_path,
                 sam_model_type="vit_b",
                 yolo_conf=0.5):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.output_csv_path = output_csv_path
        self.output_txt_path = output_txt_path
        self.yolo_conf = yolo_conf

        # Load YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # Load SAM
        sam = sam_model_registry[sam_model_type](checkpoint=sam_model_path)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_txt_path), exist_ok=True)

        self.results_data = []

    def run(self):
        image_list = sorted(os.listdir(self.data_folder))
        mask_list = sorted(os.listdir(self.mask_folder))

        for img_name, mask_name in zip(image_list, mask_list):
            # print(f"Processing {img_name}...")

            img_path = os.path.join(self.data_folder, img_name)
            mask_path = os.path.join(self.mask_folder, mask_name)

            image = cv2.imread(img_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = torch.from_numpy(gt_mask).to(self.device)

            # YOLO Prediction
            yolo_results = self.yolo_model.predict(img_path, conf=self.yolo_conf, device=self.device)
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

            self.sam_predictor.set_image(image)

            if len(boxes) == 0:
                print(f"No detections for {img_name}")
                continue
            
            H, W = (1024,1024)
            final_pred_mask = torch.zeros((H, W), dtype=torch.int, device=self.device)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # isolated_mask = generate_isolated_mask(gt_mask, [x1, y1, x2, y2])

                masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]))
                pred_mask = masks[0]
                pred_mask = pred_mask.astype("int")
                pred_mask = torch.from_numpy(pred_mask).to(self.device)
                # print(pred_mask.size())
                # print(torch.unique(pred_mask))
                # print(torch.unique(isolated_mask))
                # print(isolated_mask.size())
                
                masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]))
                pred_mask = masks[0].astype("int")
                pred_mask = torch.from_numpy(pred_mask).to(self.device)
                pred_mask_cropped = pred_mask[y1:y2, x1:x2]

                # Resize SAM mask to match the box region in full image
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

    evaluator = YOLOSAM_Evaluator(

        yolo_model_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/indian_moderate_weights/yolov10_moderate.pt",

        sam_model_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/Moderate_SAM_without_aug_on_all_data_version_1/Final_Model/SAM_final_09_07_2025.pth",

        data_folder="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final-moderate/original/test/images",

        mask_folder="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/final-moderate/original/test/masks",

        output_csv_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/yolo_sam_moderate/yolov10_Moderate_SAM_without_aug_on_all_data_version_1.csv",

        output_txt_path="/home/dhruv/Documents/DHRUV_SOLAR_ROOFTOP/solar_github/Solar_Rooftop_Detection/Solar_rooftop_Detection_indian_images/yolo_sam_moderate/yolov10_Moderate_SAM_without_aug_on_all_data_version_1.txt"
)

    evaluator.run()