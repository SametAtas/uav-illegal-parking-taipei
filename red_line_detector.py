import cv2
import numpy as np
from ultralytics import YOLO

def compute_overlap(car_bbox, red_mask):
    """
    Computes the percentage of the car bounding box that overlaps with red curb pixels.
    car_bbox: [x1, y1, x2, y2]
    red_mask: binary mask of red curbs (255 for red line, 0 otherwise)
    Returns: overlap percentage (0.0 to 1.0)
    """
    x1, y1, x2, y2 = [int(v) for v in car_bbox]
    car_area = (x2 - x1) * (y2 - y1)
    if car_area == 0:
        return 0.0
        
    # Crop the red mask to the car's bounding box
    crop = red_mask[y1:y2, x1:x2]
    overlap_pixels = cv2.countNonZero(crop)
    
    return overlap_pixels / car_area

def detect_red_lines(image_bgr, model_path="runs/segment/runs/segment/red_line_seg/weights/best.pt"):
    """
    Uses a YOLOv8 instance segmentation model to detect red curb lines.
    Returns: binary mask (red_mask) and a list of contours.
    """
    # Load model and run inference
    seg_model = YOLO(model_path)
    seg_results = seg_model(image_bgr, conf=0.25, verbose=False)
    
    h, w = image_bgr.shape[:2]
    red_mask = np.zeros((h, w), dtype=np.uint8)

    # Convert segmentation coordinates into an OpenCV mask
    if len(seg_results) > 0 and seg_results[0].masks is not None:
        for xy in seg_results[0].masks.xy:
            polygon = np.array(xy, dtype=np.int32)
            cv2.fillPoly(red_mask, [polygon], 255)

    # Convert the mask back to standard OpenCV contours
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return red_mask, red_contours
