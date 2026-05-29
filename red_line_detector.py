import os
import cv2
import numpy as np
from ultralytics import YOLO

_SEG_MODEL = None
_SEG_MODEL_RESOLVED = False  # True once we've decided whether a model is available

def _load_seg_model(model_path):
    """Load and cache the red-line seg model. Returns None if the .pt is missing."""
    global _SEG_MODEL, _SEG_MODEL_RESOLVED
    if _SEG_MODEL_RESOLVED:
        return _SEG_MODEL
    _SEG_MODEL_RESOLVED = True
    if not os.path.isfile(model_path):
        print(f"  [Note] Seg weights not found at {model_path} — using HSV-only red-line detection.")
        return None
    _SEG_MODEL = YOLO(model_path)
    return _SEG_MODEL

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

def detect_red_lines_hsv(image_bgr):
    """
    HSV-based red curb detection as a fallback method.
    Red in HSV is typically 0-10 or 170-180 in H channel.
    Returns: binary mask and contours
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Detect red colors in HSV space
    # Red range 1: 0-10
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    # Red range 2: 170-180
    red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    
    # Combine both red ranges
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    min_area = 100
    red_contours = [c for c in red_contours if cv2.contourArea(c) > min_area]
    
    return red_mask, red_contours

def detect_red_lines(image_bgr, model_path="runs/segment/runs/segment/red_line_seg/weights/best.pt"):
    """
    Hybrid red curb detection: YOLO segmentation + HSV fallback.
    Uses YOLO as primary method, and HSV as fallback if YOLO finds nothing.
    Returns: binary mask (red_mask) and a list of contours.
    """
    seg_model = _load_seg_model(model_path)
    if seg_model is None:
        return detect_red_lines_hsv(image_bgr)

    seg_results = seg_model(image_bgr, conf=0.05, verbose=False)

    h, w = image_bgr.shape[:2]
    red_mask = np.zeros((h, w), dtype=np.uint8)

    yolo_found = False
    if len(seg_results) > 0 and seg_results[0].masks is not None:
        for xy in seg_results[0].masks.xy:
            polygon = np.array(xy, dtype=np.int32)
            cv2.fillPoly(red_mask, [polygon], 255)
        yolo_found = True

    if yolo_found:
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        print("  [Note] YOLO found no red lines, falling back to HSV color detection...")
        red_mask, red_contours = detect_red_lines_hsv(image_bgr)

    return red_mask, red_contours
