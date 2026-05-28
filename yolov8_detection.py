"""
Taipei Illegal Parking Detection with UAV
========================================
Detects vehicles from aerial drone images using YOLOv8m-VisDrone + SAHI,
then checks for red curb line violations to flag illegal parking.

Usage:
    python yolov8_detection.py [image_path]
    python yolov8_detection.py                  # defaults to 003154.jpg
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter

CACHE_ROOT = Path("./.cache")
YOLO_CONFIG_DIR = CACHE_ROOT / "ultralytics"
HF_CACHE_DIR = CACHE_ROOT / "huggingface"

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
import ultralytics.nn.tasks as tasks
from huggingface_hub import snapshot_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from red_line_detector import compute_overlap, detect_red_lines

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Vehicle classes that can be flagged for illegal parking
VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motor", "tricycle", "awning-tricycle"}

# Overlap threshold: fraction of vehicle bbox that must touch red zones to be ILLEGAL
# Minimum percentage of a vehicle's bounding box that must overlap
# with the red line zone to be considered illegally parked.
# Set to 8% to prevent false positives from vehicles on bridges (which hit ~6%).
ILLEGAL_OVERLAP_THRESHOLD = 0.08

# Minimum bounding box area (pixels²) — reject tiny false-positive blips
MIN_BBOX_AREA = 50

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FIX PYTORCH SAFETY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
# Patch torch.load to default weights_only=False for older HF weights
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
from huggingface_hub import snapshot_download

print("Loading YOLOv8x-VisDrone model (Extra Large)...")
repo_dir = snapshot_download(repo_id="mshamrai/yolov8x-visdrone")
pt_files = [f for f in os.listdir(repo_dir) if f.endswith('.pt')]
model_path = os.path.join(repo_dir, pt_files[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model: {model_path} | Device: {device}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.30,  # Perfectly matches the 53 cars from the report
    device=device,
)



# ═══════════════════════════════════════════════════════════════════════════════
# 3. SELECT IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
image_path = sys.argv[1] if len(sys.argv) > 1 else "0031549.jpg"
if not os.path.exists(image_path):
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)

# Toggle this to True to divide the image into 4 pieces for finer red line detection
USE_SLICING_FOR_RED_LINES = False

print(f"\n{'='*60}")
print(f"Processing: {image_path}")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RUN SLICED VEHICLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
t_start = time.time()

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    perform_standard_pred=True,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.50,
    postprocess_class_agnostic=True,
)

t_detect = time.time() - t_start
print(f"\nVehicle detection completed in {t_detect:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. FILTER DETECTIONS (remove tiny false positives)
# ═══════════════════════════════════════════════════════════════════════════════
filtered = []
for obj in result.object_prediction_list:
    bbox = obj.bbox
    bw = bbox.maxx - bbox.minx
    bh = bbox.maxy - bbox.miny
    if bw * bh >= MIN_BBOX_AREA and obj.category.name in ["car", "van", "truck"]:
        filtered.append(obj)

removed = len(result.object_prediction_list) - len(filtered)
if removed > 0:
    print(f"Filtered out {removed} tiny false-positive detections")

# Keep the filtered list in the result object
result.object_prediction_list = filtered

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DETECT RED CURB LINES (using YOLO segmentation)
# ═══════════════════════════════════════════════════════════════════════════════
img = cv2.imread(image_path)
h, w = img.shape[:2]

os.makedirs("./output", exist_ok=True)
debug_path = "./output/red_lines_debug.png"

# Run abstracted red line detector
print("Running segmentation for red curb lines...")
red_mask, red_contours = detect_red_lines(img)
print(f"Red line segments found: {len(red_contours)}")

has_red_lines = len(red_contours) > 0

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLASSIFY ILLEGAL PARKING
# ═══════════════════════════════════════════════════════════════════════════════
illegal_vehicles = []
legal_vehicles = []

# Dilate red line mask to create a tight "proximity zone"
# A vehicle touching this zone is considered parked on the red line
# Reduced from 5 to 2 to prevent the red zone from bleeding into adjacent legal parking spots (white boxes).
zone_radius = 2
kernel = np.ones((zone_radius*2+1, zone_radius*2+1), np.uint8)
dilated_red_zone = cv2.dilate(red_mask, kernel, iterations=1)

for obj in filtered:
    bbox = obj.bbox
    x1, y1 = int(bbox.minx), int(bbox.miny)
    x2, y2 = int(bbox.maxx), int(bbox.maxy)
    cls_name = obj.category.name

    is_illegal = False
    overlap_ratio = 0.0

    if has_red_lines and cls_name in VEHICLE_CLASSES:
        # Compute overlap with dilated red line zone
        overlap_ratio = compute_overlap((x1, y1, x2, y2), dilated_red_zone)
        
        # Check violation threshold
        if overlap_ratio >= ILLEGAL_OVERLAP_THRESHOLD:
            is_illegal = True

    if is_illegal:
        illegal_vehicles.append((obj, overlap_ratio))
    else:
        legal_vehicles.append(obj)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRINT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
counts = Counter(obj.category.name for obj in filtered)
print("\n── Detection Summary ──")
for cls, count in sorted(counts.items()):
    print(f"  {cls:20} : {count}")
print(f"  {'TOTAL':20} : {sum(counts.values())}")

if has_red_lines:
    print(f"\n── Illegal Parking ──")
    print(f"  Red line zones detected : {len(red_contours)}")
    print(f"  Vehicles on red lines   : {len(illegal_vehicles)}")
    print(f"  Legal vehicles          : {len(legal_vehicles)}")

    if illegal_vehicles:
        print(f"\n  Violations:")
        for obj, ratio in illegal_vehicles:
            bbox = obj.bbox
            cx = int((bbox.minx + bbox.maxx) / 2)
            cy = int((bbox.miny + bbox.maxy) / 2)
            print(f"    ⚠ {obj.category.name:12} | conf: {obj.score.value:.0%} "
                  f"| center: ({cx}, {cy}) | overlap: {ratio:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_COLORS = {
    "pedestrian":       (0, 0, 255),       # red
    "people":           (0, 80, 255),      # orange-red
    "bicycle":          (255, 144, 30),    # dodger blue
    "car":              (0, 255, 0),       # green
    "van":              (0, 255, 255),     # yellow
    "truck":            (0, 165, 255),     # orange
    "tricycle":         (255, 0, 255),     # magenta
    "awning-tricycle":  (180, 105, 255),   # hot pink
    "bus":              (255, 255, 0),     # cyan
    "motor":            (255, 0, 128),     # purple
}
DEFAULT_COLOR = (200, 200, 200)
ILLEGAL_COLOR = (0, 0, 255)  # Bright red for illegal parking

# Scaling for text/lines based on image resolution
scale = max(w, h) / 1000.0
rect_th = max(2, int(scale * 1.2))
font_scale = scale * 0.40
font_th = max(1, int(scale * 0.6))
illegal_rect_th = max(3, int(scale * 2.0))  # Thicker border for violations

# ── 9a. Draw red line overlay ─────────────────────────────────────────────
if has_red_lines:
    overlay = img.copy()
    cv2.drawContours(overlay, red_contours, -1, (0, 0, 255), cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    # Draw outline of red zones
    cv2.drawContours(img, red_contours, -1, (0, 0, 200), rect_th)

# ── 9b. Draw legal vehicles ──────────────────────────────────────────────
for obj in legal_vehicles:
    bbox = obj.bbox
    x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
    cls_name = obj.category.name
    conf = obj.score.value
    color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, rect_th)

    # Only show confidence label for detections > 0.7 to reduce clutter
    if conf >= 0.70:
        label = f"{cls_name} {conf:.0%}"
    else:
        label = cls_name

    (tw, th_text), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th
    )
    label_y = y1 - 4
    if label_y - th_text - baseline < 0:
        label_y = y2 + th_text + 4

    cv2.rectangle(img, (x1, label_y - th_text - baseline),
                  (x1 + tw, label_y + 4), color, cv2.FILLED)
    cv2.putText(img, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                font_th, cv2.LINE_AA)

# ── 9c. Draw illegal vehicles (on top, so they stand out) ────────────────
for obj, ratio in illegal_vehicles:
    bbox = obj.bbox
    x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
    cls_name = obj.category.name
    conf = obj.score.value

    # Thick red border
    cv2.rectangle(img, (x1, y1), (x2, y2), ILLEGAL_COLOR, illegal_rect_th)

    # Warning label
    label = f"ILLEGAL {cls_name} {conf:.0%}"
    (tw, th_text), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, font_th + 1
    )
    label_y = y1 - 6
    if label_y - th_text - baseline < 0:
        label_y = y2 + th_text + 6

    # Red background label
    cv2.rectangle(img, (x1, label_y - th_text - baseline - 2),
                  (x1 + tw + 4, label_y + 6), ILLEGAL_COLOR, cv2.FILLED)
    cv2.putText(img, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (255, 255, 255),
                font_th + 1, cv2.LINE_AA)

# ── 9d. Draw legend in top-right corner ───────────────────────────────────
legend_x = w - int(220 * scale)
legend_y = int(20 * scale)
legend_h = int(25 * scale)
legend_font = scale * 0.35

# Semi-transparent background for legend
legend_entries = [
    (f"Legal Vehicle: {len(legal_vehicles)}", (0, 255, 0)),
    (f"Red Curb Zone: {len(red_contours)}", (0, 0, 255)),
    (f"ILLEGAL Parking: {len(illegal_vehicles)}", (0, 0, 255)),
]
lbg_h = len(legend_entries) * legend_h + int(15 * scale)
lbg_w = int(220 * scale)  # Increased slightly to fit the numbers
overlay2 = img.copy()

# Draw legend background
cv2.rectangle(overlay2, (legend_x - 5, legend_y - 5),
              (legend_x + lbg_w, legend_y + lbg_h), (0, 0, 0), cv2.FILLED)

# Blend overlay
img = cv2.addWeighted(overlay2, 0.6, img, 0.4, 0)

for i, (text, color) in enumerate(legend_entries):
    y = legend_y + int(15 * scale) + i * legend_h
    cv2.rectangle(img, (legend_x, y - int(8 * scale)),
                  (legend_x + int(15 * scale), y + int(5 * scale)),
                  color, cv2.FILLED)
    cv2.putText(img, text, (legend_x + int(20 * scale), y + int(3 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font, (255, 255, 255),
                max(1, int(scale * 0.5)), cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
out_path = "./output/final_result.png"
cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

# ── Export Coordinates for Metashape ──────────────────────────────────────────
csv_path = "./output/metashape_markers.csv"
image_filename = os.path.basename(image_path)
with open(csv_path, "w") as f:
    # Metashape standard format for 2D Image Coordinates: marker_name, image_name, x, y
    f.write("marker_label,image_name,x_pixel,y_pixel\n")
    for idx, (obj, ratio) in enumerate(illegal_vehicles, 1):
        bbox = obj.bbox
        cx = int((bbox.minx + bbox.maxx) / 2)
        cy = int((bbox.miny + bbox.maxy) / 2)
        f.write(f"Illegal_Violation_{idx},{image_filename},{cx},{cy}\n")

t_total = time.time() - t_start
print(f"\nTotal processing time: {t_total:.1f}s")
print(f"Saved to {out_path}")
print(f"Debug red-line visualization: {debug_path}")
print(f"Exported illegal vehicle coordinates to: {csv_path}")
