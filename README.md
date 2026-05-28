# Taipei Illegal Parking Detection with UAV

> **An AI-powered computer vision pipeline for automated illegal parking detection from drone imagery in Taipei.**

A sleek system designed to detect illegally parked vehicles... The pipeline leverages state-of-the-art computer vision to identify red curb lines (紅線) and flags vehicles that are parked in these restricted zones.

## Features
- **Vehicle Detection:** Utilizes YOLOv8 (Medium/VisDrone) for high-accuracy bounding box detection of cars, trucks, buses, and motorcycles from aerial views.
- **Red Line Detection:** Advanced HSV-based color segmentation combined with geometric contour analysis (aspect ratio, true thickness) to robustly detect painted red curbs.
- **Illegal Parking Logic:** Automatically calculates Intersection over Union (IoU) between vehicle bounding boxes and red-line masks. Flags vehicles as `ILLEGAL` if they overlap with restricted zones.
- **Sleek Visualization:** Generates beautiful bounding box overlays with clear legal/illegal color coding.

## Repository Structure
- `yolov8_detection.py` - Main pipeline for detecting vehicles and overlaying results.
- `red_line_detector.py` - Core logic for segmenting and filtering red lines.
- `train_yolo.py` - Script for fine-tuning YOLOv8 on specific UAV datasets.
- `train_red_line_seg.py` - Setup for training an upcoming semantic segmentation model for even better road extraction.

## Future Work
- Replacing the HSV color filter with a dedicated YOLOv8-seg Semantic Segmentation model to perfectly separate roads from buildings, eliminating all structural false positives.
