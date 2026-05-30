"""
convert_pixel_to_gps.py
=======================
Non-Metashape pixel → GPS reconstruction for illegal-parking detections.

Mirrors the logic of the original Metashape plugin of the same name:
  1. Per image, back-project each YOLO bbox center to world coordinates.
  2. Cluster the same vehicle across images by class + horizontal distance.
  3. Filter spurious clusters (here: minimum-observations instead of Z-sigma).
  4. Emit per-vehicle CSV with WGS84 lat/lon and TWD97 X/Y.

Key differences from the Metashape version:
  * No depth map → flat-ground assumption, Z = drone RelativeAltitude.
  * No bundle-adjusted extrinsics → DJI EXIF (GPS + gimbal angles) per image,
    via the collinearity equation (see ``collinearity_math.py``).
  * Reprojection-into-other-camera check is omitted: without BA the second
    camera's extrinsics aren't accurate enough for that check to add signal.
  * Z-score altitude filter is replaced with a ``min_observations`` filter:
    a real vehicle is usually seen in 2+ consecutive frames during a flight.

Usage (standalone CLI, reads batch_detect.py output):
    python convert_pixel_to_gps.py [markers.csv] [image_dir] [out.csv]

Defaults:
    markers.csv  = ./output/batch/all_illegal_markers.csv
    image_dir    = D:/UAV/C
    out.csv      = ./output/batch/illegal_vehicles_gps.csv
"""

import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer

# Calibrated camera intrinsics for the original author's DJI camera
# (see collinearity_math.py). Override these if you've recalibrated.
DEFAULT_INTRINSICS = {
    "f":  2873.64,
    "cx": 15.89,
    "cy": -6.54,
    "img_w": 4000,
    "img_h": 3000,
}

DJI_XMP_TAGS = {
    "lat":          "GpsLatitude",
    "lon":          "GpsLongitude",
    "altitude":     "RelativeAltitude",
    "gimbal_pitch": "GimbalPitchDegree",
    "gimbal_roll":  "GimbalRollDegree",
    "gimbal_yaw":   "GimbalYawDegree",
}

_WGS84_TO_TWD97 = Transformer.from_crs("epsg:4326", "epsg:3826", always_xy=True)
_TWD97_TO_WGS84 = Transformer.from_crs("epsg:3826", "epsg:4326", always_xy=True)


# ─── EXIF ────────────────────────────────────────────────────────────────────
def read_dji_exif(image_path):
    """Pull GPS + gimbal angles from DJI XMP. Returns None if any tag missing."""
    with open(image_path, "rb") as f:
        # XMP block lives in the first ~64 KB; 200 KB is a safe over-read.
        raw = f.read(200_000).decode("latin-1", errors="ignore")
    out = {}
    for key, tag in DJI_XMP_TAGS.items():
        m = re.search(fr'drone-dji:{tag}="([^"]+)"', raw)
        if m is None:
            return None
        out[key] = float(m.group(1))
    return out


# ─── Collinearity back-projection ────────────────────────────────────────────
def pixel_to_twd97(px, py, exif, intr):
    """
    Single-image, flat-ground back-projection: pixel (px, py) → TWD97 (X, Y).

    DJI → photogrammetric convention mapping (matches collinearity_math.py):
      omega = GimbalRollDegree              (roll about flight axis)
      phi   = GimbalPitchDegree + 90        (0 = nadir; DJI uses -90 for nadir)
      kappa = -GimbalYawDegree              (heading → photogrammetric yaw)
      Z_offset = -RelativeAltitude          (camera above ground = negative Z)
    """
    f, cx, cy = intr["f"], intr["cx"], intr["cy"]
    img_w, img_h = intr["img_w"], intr["img_h"]

    omega = math.radians(exif["gimbal_roll"])
    phi   = math.radians(exif["gimbal_pitch"] + 90.0)
    kappa = math.radians(-exif["gimbal_yaw"])
    delta_Z = -abs(exif["altitude"])

    X_O, Y_O = _WGS84_TO_TWD97.transform(exif["lon"], exif["lat"])

    M_omega = np.array([[1, 0, 0],
                        [0, math.cos(omega), math.sin(omega)],
                        [0, -math.sin(omega), math.cos(omega)]])
    M_phi = np.array([[math.cos(phi), 0, -math.sin(phi)],
                      [0, 1, 0],
                      [math.sin(phi), 0, math.cos(phi)]])
    M_kappa = np.array([[math.cos(kappa), math.sin(kappa), 0],
                        [-math.sin(kappa), math.cos(kappa), 0],
                        [0, 0, 1]])
    M = M_kappa @ M_phi @ M_omega

    x_a = px - (img_w / 2.0)
    y_a = (img_h / 2.0) - py  # image y-axis flips for photogrammetric frame

    num_X = M[0, 0] * (x_a - cx) + M[1, 0] * (y_a - cy) + M[2, 0] * (-f)
    num_Y = M[0, 1] * (x_a - cx) + M[1, 1] * (y_a - cy) + M[2, 1] * (-f)
    den   = M[0, 2] * (x_a - cx) + M[1, 2] * (y_a - cy) + M[2, 2] * (-f)

    X = X_O + delta_Z * (num_X / den)
    Y = Y_O + delta_Z * (num_Y / den)
    return X, Y


def twd97_to_wgs84(x, y):
    lon, lat = _TWD97_TO_WGS84.transform(x, y)
    return lat, lon


# ─── Cross-image clustering ──────────────────────────────────────────────────
def cluster_detections(detections, search_dist_m=10.0):
    """
    Greedy merge: same class + horizontal distance < threshold → same vehicle.
    Mirrors Metashape script's class-gated, distance-thresholded clustering.
    Cluster centroid is an online running mean.
    """
    clusters = []
    for det in detections:
        merged = False
        for c in clusters:
            if c["cls"] != det["cls"]:
                continue
            if math.hypot(c["x_twd97"] - det["x_twd97"],
                          c["y_twd97"] - det["y_twd97"]) > search_dist_m:
                continue
            n = len(c["observations"])
            c["x_twd97"] = (c["x_twd97"] * n + det["x_twd97"]) / (n + 1)
            c["y_twd97"] = (c["y_twd97"] * n + det["y_twd97"]) / (n + 1)
            c["observations"].append(det)
            merged = True
            break
        if not merged:
            clusters.append({
                "cls": det["cls"],
                "x_twd97": det["x_twd97"],
                "y_twd97": det["y_twd97"],
                "observations": [det],
            })
    for c in clusters:
        c["lat"], c["lon"] = twd97_to_wgs84(c["x_twd97"], c["y_twd97"])
    return clusters


# ─── End-to-end ──────────────────────────────────────────────────────────────
def convert(detections_per_image, image_dir, intrinsics=None,
            search_dist_m=10.0, min_observations=1):
    """
    detections_per_image: {image_filename: [(cls, x_pixel, y_pixel), ...]}
    Returns the filtered cluster list.
    """
    intr = intrinsics or DEFAULT_INTRINSICS

    all_dets = []
    skipped = []
    for img_name, vehicles in detections_per_image.items():
        path = os.path.join(image_dir, img_name)
        exif = read_dji_exif(path)
        if exif is None:
            skipped.append(img_name)
            continue
        for cls, px, py in vehicles:
            X, Y = pixel_to_twd97(px, py, exif, intr)
            all_dets.append({
                "cls": cls, "image": img_name,
                "x_pixel": px, "y_pixel": py,
                "x_twd97": X, "y_twd97": Y,
            })

    clusters = cluster_detections(all_dets, search_dist_m=search_dist_m)
    kept = [c for c in clusters if len(c["observations"]) >= min_observations]

    if skipped:
        print(f"[warn] skipped {len(skipped)} images missing DJI EXIF "
              f"(e.g. {skipped[:3]}{'...' if len(skipped) > 3 else ''})")
    print(f"Raw detections: {len(all_dets)}  |  "
          f"clusters: {len(clusters)}  |  kept (>={min_observations} sightings): {len(kept)}")
    return kept


def write_csv(clusters, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["marker_label", "class", "latitude", "longitude",
                    "twd97_x", "twd97_y", "num_sightings", "source_images"])
        for i, c in enumerate(clusters, 1):
            src = ";".join(sorted({o["image"] for o in c["observations"]}))
            w.writerow([
                f"vehicle_{i:04d}", c["cls"],
                f"{c['lat']:.7f}", f"{c['lon']:.7f}",
                f"{c['x_twd97']:.2f}", f"{c['y_twd97']:.2f}",
                len(c["observations"]), src,
            ])


# ─── CLI ─────────────────────────────────────────────────────────────────────
def cli():
    csv_in    = sys.argv[1] if len(sys.argv) > 1 else "./output/batch/all_illegal_markers.csv"
    image_dir = sys.argv[2] if len(sys.argv) > 2 else r"D:\UAV\C"
    csv_out   = sys.argv[3] if len(sys.argv) > 3 else "./output/batch/illegal_vehicles_gps.csv"

    detections_per_image = {}
    with open(csv_in, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row.get("class") or "vehicle"
            detections_per_image.setdefault(row["image_name"], []).append(
                (cls, int(row["x_pixel"]), int(row["y_pixel"]))
            )
    print(f"Loaded {sum(len(v) for v in detections_per_image.values())} "
          f"detections across {len(detections_per_image)} images")

    clusters = convert(detections_per_image, image_dir,
                       search_dist_m=10.0, min_observations=1)
    write_csv(clusters, csv_out)
    print(f"Wrote {len(clusters)} unique vehicles → {csv_out}")


if __name__ == "__main__":
    cli()
