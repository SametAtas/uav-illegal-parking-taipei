# UAV Illegal Parking Detection System (Taipei)

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)
![SAHI](https://img.shields.io/badge/SAHI-Vision-yellow.svg)

A computer vision pipeline developed for detecting illegally parked vehicles from high-altitude 4K drone imagery (UAV). This project integrates object detection with Photogrammetry software for 3D spatial mapping.

## System Output

![Final Detection Result](assets/final_result.png)
> **Dataset Citation**: The raw drone imagery used in this pipeline represents the NTNU Campus. These aerial images were provided by the course lecturer for this academic project.

## Project Summary
Detecting vehicles from high-altitude drone footage is challenging because objects appear very small (often under 10 pixels), causing standard object-detection models to miss targets or produce false positives. 

This pipeline addresses this by combining a YOLOv8 Extra Large (`yolov8x`) model fine-tuned on the VisDrone dataset with Slicing Aided Hyper Inference (SAHI). By slicing 4K drone imagery into overlapping patches, the system maintains original image resolution to identify small vehicles. The pipeline then computes spatial intersections between detected vehicles and computer-vision-segmented red curbs (illegal zones) to flag violations. The system extracts pixel coordinates and exports them to a standardized CSV format for downstream 3D mapping.

## Key Features
- **Small Object Detection (SAHI):** Slices high-resolution images (`512x512` chunks with `25%` overlap) before running inference, avoiding resolution loss from standard YOLO downsampling.
- **Drone-Specific Weights:** Utilizes `mshamrai/yolov8x-visdrone`, a YOLOv8 Extra Large model fine-tuned on aerial drone data to reduce the domain gap.
- **Class Filtering:** Explicitly tracks `car`, `truck`, and `van` classes while filtering out motorcycles and pedestrians.
- **Red Curb Segmentation:** Uses OpenCV HSV color segmentation and geometric contour analysis to isolate painted red curbs.
- **Violation Logic:** Computes Intersection over Union (IoU) between vehicle bounding boxes and dilated red zones to identify illegal parking.
- **Photogrammetry Integration:** Exports illegal vehicle `(X, Y)` center-pixel coordinates to a standardized CSV for 3D marker generation.

## Environment & Requirements
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

## Usage
Run the main detection pipeline on a specific drone image:
```bash
./UAV_VENV/bin/python yolov8_detection.py <image_name>.jpg
```

**Outputs:**
1. `output/final_result.png`: An annotated image displaying green bounding boxes for legal vehicles, red boxes for illegal vehicles, highlighted red curbs, and a dynamic statistics legend.
2. `output/metashape_markers.csv`: A CSV file containing exact coordinate markers ready for photogrammetry software.

## Citations & Acknowledgements
This pipeline relies heavily on the open-source community. We acknowledge the following tools, datasets, and algorithms:

1. **Ultralytics YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). *YOLO by Ultralytics* (Version 8.0.0). [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. **SAHI (Slicing Aided Hyper Inference)**: Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *IEEE International Conference on Image Processing (ICIP)*. [https://github.com/obss/sahi](https://github.com/obss/sahi)
3. **VisDrone Dataset**: Zhu, P., Wen, L., Du, D., et al. (2021). Detection and tracking meet drones challenge. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
4. **HuggingFace Hub (Model Weights)**: The Extra Large VisDrone fine-tuned weights were provided via HuggingFace Hub by `mshamrai`.
