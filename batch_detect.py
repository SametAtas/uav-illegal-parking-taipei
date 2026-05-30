"""
Batch UAV illegal-parking detection.

Loads the SAHI/YOLOv8x-VisDrone model once, then iterates over every JPG in the
given directory. Per-image: writes an annotated PNG; across the whole run:
writes one combined CSV of illegal-vehicle markers ready for Metashape.

Usage:
    python batch_detect.py <image_dir> [output_dir]

Defaults: image_dir = D:/UAV/C, output_dir = ./output/batch
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

CACHE_ROOT = Path("./.cache")
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
(CACHE_ROOT / "ultralytics").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "huggingface").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from red_line_detector import compute_overlap, detect_red_lines

VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motor", "tricycle", "awning-tricycle"}
ILLEGAL_OVERLAP_THRESHOLD = 0.08
MIN_BBOX_AREA = 50

CLASS_COLORS = {
    "pedestrian": (0, 0, 255), "people": (0, 80, 255), "bicycle": (255, 144, 30),
    "car": (0, 255, 0), "van": (0, 255, 255), "truck": (0, 165, 255),
    "tricycle": (255, 0, 255), "awning-tricycle": (180, 105, 255),
    "bus": (255, 255, 0), "motor": (255, 0, 128),
}
ILLEGAL_COLOR = (0, 0, 255)
DEFAULT_COLOR = (200, 200, 200)


_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load


def load_model():
    print("Loading YOLOv8x-VisDrone model...")
    repo_dir = snapshot_download(repo_id="mshamrai/yolov8x-visdrone")
    pt = next(f for f in os.listdir(repo_dir) if f.endswith(".pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  weights: {pt}  |  device: {device}")
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=os.path.join(repo_dir, pt),
        confidence_threshold=0.30,
        device=device,
    )


def process_one(image_path, model, out_dir):
    # Use np.fromfile to handle non-ASCII paths if any
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]

    result = get_sliced_prediction(
        image_path, model,
        slice_height=512, slice_width=512,
        overlap_height_ratio=0.25, overlap_width_ratio=0.25,
        perform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.50,
        postprocess_class_agnostic=True,
        verbose=0,
    )

    filtered = [
        o for o in result.object_prediction_list
        if (o.bbox.maxx - o.bbox.minx) * (o.bbox.maxy - o.bbox.miny) >= MIN_BBOX_AREA
        and o.category.name in ("car", "van", "truck")
    ]

    red_mask, red_contours = detect_red_lines(img)
    has_red = len(red_contours) > 0

    zone_radius = 2
    k = np.ones((zone_radius * 2 + 1, zone_radius * 2 + 1), np.uint8)
    dilated = cv2.dilate(red_mask, k, iterations=1) if has_red else red_mask

    legal, illegal = [], []
    detections_for_json = []
    for o in filtered:
        x1, y1 = int(o.bbox.minx), int(o.bbox.miny)
        x2, y2 = int(o.bbox.maxx), int(o.bbox.maxy)
        ratio = compute_overlap((x1, y1, x2, y2), dilated) if has_red and o.category.name in VEHICLE_CLASSES else 0.0
        is_illegal = ratio >= ILLEGAL_OVERLAP_THRESHOLD
        (illegal if is_illegal else legal).append((o, ratio))
        detections_for_json.append({
            "class": o.category.name,
            "confidence": round(float(o.score.value), 4),
            "bbox_xyxy": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "area_px": (x2 - x1) * (y2 - y1),
            "illegal": is_illegal,
            "red_overlap_ratio": round(float(ratio), 4),
        })

    scale = max(w, h) / 1000.0
    rect_th = max(2, int(scale * 1.2))
    font_scale = scale * 0.40
    font_th = max(1, int(scale * 0.6))
    illegal_rect_th = max(3, int(scale * 2.0))

    if has_red:
        overlay = img.copy()
        cv2.drawContours(overlay, red_contours, -1, (0, 0, 255), cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        cv2.drawContours(img, red_contours, -1, (0, 0, 200), rect_th)

    for o, _ in legal:
        x1, y1 = int(o.bbox.minx), int(o.bbox.miny)
        x2, y2 = int(o.bbox.maxx), int(o.bbox.maxy)
        color = CLASS_COLORS.get(o.category.name, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, rect_th)
        label = f"{o.category.name} {o.score.value:.0%}" if o.score.value >= 0.70 else o.category.name
        (tw, th_t), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)
        ly = y1 - 4 if y1 - 4 - th_t - bl >= 0 else y2 + th_t + 4
        cv2.rectangle(img, (x1, ly - th_t - bl), (x1 + tw, ly + 4), color, cv2.FILLED)
        cv2.putText(img, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_th, cv2.LINE_AA)

    for o, _ in illegal:
        x1, y1 = int(o.bbox.minx), int(o.bbox.miny)
        x2, y2 = int(o.bbox.maxx), int(o.bbox.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), ILLEGAL_COLOR, illegal_rect_th)
        label = f"ILLEGAL {o.category.name} {o.score.value:.0%}"
        (tw, th_t), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, font_th + 1)
        ly = y1 - 6 if y1 - 6 - th_t - bl >= 0 else y2 + th_t + 6
        cv2.rectangle(img, (x1, ly - th_t - bl - 2), (x1 + tw + 4, ly + 6), ILLEGAL_COLOR, cv2.FILLED)
        cv2.putText(img, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (255, 255, 255), font_th + 1, cv2.LINE_AA)

    legend_x = w - int(220 * scale)
    legend_y = int(20 * scale)
    legend_h = int(25 * scale)
    entries = [
        (f"Legal Vehicle: {len(legal)}", (0, 255, 0)),
        (f"Red Curb Zone: {len(red_contours)}", (0, 0, 255)),
        (f"ILLEGAL Parking: {len(illegal)}", (0, 0, 255)),
    ]
    lbg_h = len(entries) * legend_h + int(15 * scale)
    lbg_w = int(220 * scale)
    ov2 = img.copy()
    cv2.rectangle(ov2, (legend_x - 5, legend_y - 5), (legend_x + lbg_w, legend_y + lbg_h), (0, 0, 0), cv2.FILLED)
    img = cv2.addWeighted(ov2, 0.6, img, 0.4, 0)
    for i, (txt, color) in enumerate(entries):
        y = legend_y + int(15 * scale) + i * legend_h
        cv2.rectangle(img, (legend_x, y - int(8 * scale)), (legend_x + int(15 * scale), y + int(5 * scale)), color, cv2.FILLED)
        cv2.putText(img, txt, (legend_x + int(20 * scale), y + int(3 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.35, (255, 255, 255), max(1, int(scale * 0.5)), cv2.LINE_AA)

    stem = Path(image_path).stem
    out_png = os.path.join(out_dir, f"{stem}_result.png")
    cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(out_png)

    per_image_json = {
        "image": os.path.basename(image_path),
        "width": w,
        "height": h,
        "red_zones": len(red_contours),
        "total_vehicles": len(filtered),
        "legal_count": len(legal),
        "illegal_count": len(illegal),
        "detections": detections_for_json,
    }
    out_json = os.path.join(out_dir, f"{stem}_detections.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(per_image_json, f, indent=2, ensure_ascii=False)

    return {
        "image": os.path.basename(image_path),
        "total_vehicles": len(filtered),
        "red_zones": len(red_contours),
        "legal": len(legal),
        "illegal": illegal,  # list of (obj, ratio)
        "out_png": out_png,
        "json_summary": per_image_json,
    }


def main():
    image_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\UAV\C"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./output/batch"
    os.makedirs(out_dir, exist_ok=True)

    seen = set()
    images = []
    for p in sorted(list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.JPG"))):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            images.append(p)
    if not images:
        print(f"No JPG images in {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Outputs will go to {out_dir}\n")

    model = load_model()

    combined_csv = os.path.join(out_dir, "all_illegal_markers.csv")
    summary_csv = os.path.join(out_dir, "summary.csv")
    combined_json = os.path.join(out_dir, "all_detections.json")
    csv_f = open(combined_csv, "w", encoding="utf-8")
    sum_f = open(summary_csv, "w", encoding="utf-8")
    csv_f.write("marker_label,image_name,class,confidence,x_pixel,y_pixel\n")
    sum_f.write("image,total_vehicles,red_zones,legal,illegal,seconds\n")

    grand_total_illegal = 0
    grand_total_vehicles = 0
    all_image_records = []
    t0_all = time.time()

    for idx, img_path in enumerate(images, 1):
        t0 = time.time()
        try:
            res = process_one(str(img_path), model, out_dir)
        except Exception as e:
            print(f"[{idx:>3}/{len(images)}] {img_path.name}  ERROR: {e}")
            continue
        if res is None:
            print(f"[{idx:>3}/{len(images)}] {img_path.name}  unreadable, skipping")
            continue
        dt = time.time() - t0

        for j, (o, _) in enumerate(res["illegal"], 1):
            cx = int((o.bbox.minx + o.bbox.maxx) / 2)
            cy = int((o.bbox.miny + o.bbox.maxy) / 2)
            csv_f.write(f"{Path(res['image']).stem}_violation_{j},{res['image']},{o.category.name},{float(o.score.value):.4f},{cx},{cy}\n")

        sum_f.write(f"{res['image']},{res['total_vehicles']},{res['red_zones']},{res['legal']},{len(res['illegal'])},{dt:.1f}\n")
        csv_f.flush()
        sum_f.flush()
        all_image_records.append(res["json_summary"])

        grand_total_illegal += len(res["illegal"])
        grand_total_vehicles += res["total_vehicles"]
        print(f"[{idx:>3}/{len(images)}] {res['image']:20} vehicles={res['total_vehicles']:>3}  red_zones={res['red_zones']:>2}  illegal={len(res['illegal']):>2}  ({dt:>4.1f}s)")

    csv_f.close()
    sum_f.close()

    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": "mshamrai/yolov8x-visdrone",
                "confidence_threshold": 0.30,
                "illegal_overlap_threshold": ILLEGAL_OVERLAP_THRESHOLD,
                "min_bbox_area_px": MIN_BBOX_AREA,
                "slice": {"height": 512, "width": 512, "overlap": 0.25},
            },
            "summary": {
                "images_processed": len(all_image_records),
                "total_vehicles": grand_total_vehicles,
                "total_illegal": grand_total_illegal,
                "total_seconds": round(time.time() - t0_all, 1),
            },
            "images": all_image_records,
        }, f, indent=2, ensure_ascii=False)

    total_dt = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"Done in {total_dt/60:.1f} min  ({total_dt:.0f}s)")
    print(f"  Images processed       : {len(images)}")
    print(f"  Total vehicles         : {grand_total_vehicles}")
    print(f"  Total illegal markers  : {grand_total_illegal}")
    print(f"  Combined CSV           : {combined_csv}")
    print(f"  Per-image summary CSV  : {summary_csv}")
    print(f"  Combined JSON          : {combined_json}")
    print(f"  Per-image JSON         : {out_dir}\\<image>_detections.json")
    print(f"  Annotated PNGs         : {out_dir}\\<image>_result.png")


if __name__ == "__main__":
    main()
