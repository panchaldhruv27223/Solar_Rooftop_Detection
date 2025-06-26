# ☀️ High-Precision Rooftop Detection in Urban Aerial Imagery

This repository implements a **hybrid detection–segmentation pipeline** for high-precision rooftop detection from aerial imagery. It integrates **state-of-the-art object detection models** (YOLOv10/11/12, DETR) with the **Segment Anything Model (SAM)** to guide segmentation. Additionally, baseline semantic and instance segmentation models (U-Net, DeepLabV3+, Mask R-CNN) are benchmarked for comparison.

---

## 📌 Project Highlights

- 🔄 **Two-stage hybrid pipeline**: Detection followed by region-guided segmentation.
- 🎯 **Prompt-based segmentation**: Bounding boxes from detectors used as SAM prompts.
- 🇮🇳 **Indian rooftop datasets**: Custom-labeled Gandhinagar dataset (600+ images) and upcoming dense-region Ahmedabad dataset.
- 📊 **Comprehensive comparison**: Benchmarking across rooftops, roads, trees, and solar panels using 4 public datasets.

---

## 🏗️ Pipeline Architecture

### 🔹 Baseline Pipeline

Standalone segmentation models trained on aerial imagery:
- `U-Net`
- `DeepLabV3+`
- `Mask R-CNN`
- `Segment Anything Model (SAM)` (zero-shot and fine-tuned)

📁 Directory: `BaseLineModels/`

---

### 🔹 Hybrid Pipeline

**Proposed two-stage approach**:
1. **Detection**: YOLOv10/11/12 or DETR localizes rooftops.
2. **Prompted Segmentation**: SAM generates masks using bounding box prompts.


📁 Directory: `Solar_Rooftop_Detection/`

---

## 🌍 Dataset Coverage

- **Inria Aerial Image Labeling Dataset** (Rooftops)
- **Zenodo Solar Panel Dataset**
- **Massachusetts Roads Dataset**
- **ISPRS Potsdam Dataset** (Trees)

### 🇮🇳 India-Specific Contribution

- ✅ **Gandhinagar Dataset**: 600+ manually annotated rooftop images.
- ✅ All models trained and tested on this dataset.
- 🔜 **Ahmedabad Dataset**: Collected dense urban images—results coming soon.

📁 Directory: `Solar_rooftop_Detection_indian_images/`

---

## 📊 Model Performance (Rooftops Example)

| Model               | Pixel IoU | Dice Score | Region Accuracy |
|---------------------|-----------|------------|------------------|
| YOLOv12 + SAM       | 0.780     | 0.863      | 0.965            |
| DETR + SAM          | 0.745     | 0.840      | 0.953            |
| U-Net               | 0.633     | 0.774      | 1.000            |
| DeepLabV3           | 0.648     | 0.747      | 0.911            |
| Mask R-CNN          | 0.017     | 0.019      | 0.021            |

📁 Full results: `accuracy.txt`, `RooftopAccuracy.txt`

---



## 📄 Reports

- 📘 `Report_Version_1.pdf`: Methodology, architecture, early experiments, 4 urban segmentation tasks, detailed benchmarks, and pipeline diagrams

---

## ✅ Key Contributions

- 🧱 Designed & implemented a modular rooftop segmentation pipeline
- ✍️ Annotated 600+ Indian rooftops (Gandhinagar)
- 📦 Integrated 4 detectors (YOLOv10–12, DETR) and 4 segmenters (U-Net, DeepLabV3, Mask R-CNN, SAM)
- 🧠 Leveraged SAM for zero-shot and prompt-based segmentation
- 🧪 Conducted detailed evaluation across four public datasets and Indian rooftops

---

Stay tuned for results on the **Ahmedabad dataset**, focusing on dense urban morphology.

Developed by Dhruv Panchal And Ritwik Agrawal. For questions or collaboration, reach out via GitHub.

> For setup, usage, training, and visualization instructions — refer to the upcoming usage section or contact the authors.

---
