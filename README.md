# UAV Illegal Parking Detection System (Taipei)

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)
![SAHI](https://img.shields.io/badge/SAHI-Vision-yellow.svg)

An AI-powered computer vision and photogrammetry pipeline engineered to automatically detect illegally parked vehicles from high-altitude 4K drone imagery (UAV). This project solves the domain-gap and small-object detection problems inherent in drone aerial photography and fully integrates with GIS/Photogrammetry software for 3D spatial mapping.

## 📌 Project Summary
Detecting vehicles from drone footage is notoriously difficult due to extreme altitudes where objects shrink to mere pixels, causing standard object-detection models to fail or hallucinate. 

This pipeline addresses these challenges by combining the **YOLOv8 Extra Large (`yolov8x`) VisDrone** model with **Slicing Aided Hyper Inference (SAHI)**. By slicing 4K drone imagery into overlapping patches, the system maintains high resolution to identify vehicles down to 3 pixels. The pipeline then computes spatial intersections between detected vehicles and computer-vision-segmented red curbs (illegal zones) to automatically flag violations. Finally, the system extracts precise pixel coordinates and exports them to Agisoft Metashape to project the violations onto a 3D orthomosaic Map.

## 🚀 Key Features
- **Small Object Detection (SAHI):** Slices high-resolution images (`512x512` chunks with `25%` overlap) before running inference, bypassing the typical YOLO downsampling compression.
- **Domain-Specific AI:** Utilizes `mshamrai/yolov8x-visdrone`, a specialized YOLOv8 Extra Large model fine-tuned specifically on drone data, avoiding ground-level domain gap issues.
- **Targeted Filtering:** Explicitly tracks `car`, `truck`, and `van` classes while rejecting motorcycles, pedestrians, and noise.
- **Red Curb Segmentation:** Robust OpenCV HSV color segmentation combined with geometric contour analysis isolates painted red curbs.
- **Automated Violation Logic:** Computes Intersection over Union (IoU) between bounding boxes and dilated red zones to instantly flag illegal parking.
- **Photogrammetry Integration:** Automatically exports illegal vehicle `(X, Y)` center-pixel coordinates into a standardized CSV for 1-click Marker Import into Agisoft Metashape.

## 💻 Environment & Requirements
The pipeline was developed and tested in the following environment:
- **OS**: Ubuntu Linux
- **Python**: `3.13.12`
- **CUDA**: `11.8` (NVIDIA compilation tools)
- **Frameworks**: PyTorch, Ultralytics, SAHI, OpenCV

### Setup
```bash
python -m venv UAV_VENV
source UAV_VENV/bin/activate
pip install ultralytics sahi opencv-python torch torchvision
```

## 🛠 Usage
Run the main detection pipeline on a specific drone image:
```bash
./UAV_VENV/bin/python yolov8_detection.py <image_name>.jpg
```

**Outputs:**
1. `output/final_result.png`: An annotated image displaying green bounding boxes for legal vehicles, red boxes for illegal vehicles, highlighted red curbs, and a dynamic statistics legend.
2. `output/metashape_markers.csv`: A CSV file containing exact coordinate markers ready to be imported into Agisoft Metashape.

## 📚 Citations & Acknowledgements
This pipeline relies heavily on the open-source community. We acknowledge the following tools, datasets, and algorithms:

1. **Ultralytics YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). *YOLO by Ultralytics* (Version 8.0.0). [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. **SAHI (Slicing Aided Hyper Inference)**: Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *IEEE International Conference on Image Processing (ICIP)*. [https://github.com/obss/sahi](https://github.com/obss/sahi)
3. **VisDrone Dataset**: Zhu, P., Wen, L., Du, D., et al. (2021). Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
4. **HuggingFace Hub (Model Weights)**: The Extra Large VisDrone fine-tuned weights were provided via HuggingFace Hub by `mshamrai`.
5. **Agisoft Metashape**: Used as the primary photogrammetry suite for 3D orthomosaic generation and spatial mapping.

---
*Developed for UAV Image Processing Final Project.*
