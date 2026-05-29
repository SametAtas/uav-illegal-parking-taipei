"""
visualize_map.py
================
Interactive folium map for sanity-checking the whole pipeline.

Layers shown:
  - Drone flight path: polyline through every photo's GPS, dots at each capture.
  - Unique illegal-vehicle markers: one circle per cluster from
    convert_pixel_to_gps.py, color/size by sighting count.
  - OpenStreetMap + Esri satellite tiles (toggle in top-right).

Usage:
    python visualize_map.py [vehicles.csv] [image_dir] [out.html]

Defaults:
    vehicles.csv = ./output/batch_v2/illegal_vehicles_gps.csv
    image_dir    = D:/UAV/C
    out.html     = ./output/batch_v2/map.html
"""

import csv
import os
import sys
from pathlib import Path

import folium

from convert_pixel_to_gps import read_dji_exif


def load_vehicles(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "label":     row["marker_label"],
                "cls":       row["class"],
                "lat":       float(row["latitude"]),
                "lon":       float(row["longitude"]),
                "twd97_x":   float(row["twd97_x"]),
                "twd97_y":   float(row["twd97_y"]),
                "sightings": int(row["num_sightings"]),
                "sources":   row["source_images"].split(";"),
            })
    return rows


def load_flight(image_dir):
    """Read drone EXIF from every JPG in image_dir, ordered by filename."""
    points = []
    for p in sorted(Path(image_dir).glob("*.JPG")):
        e = read_dji_exif(str(p))
        if e is None:
            continue
        points.append({
            "image":    p.name,
            "lat":      e["lat"],
            "lon":      e["lon"],
            "altitude": e["altitude"],
            "yaw":      e["gimbal_yaw"],
            "pitch":    e["gimbal_pitch"],
        })
    return points


def build_map(vehicles, flight, out_html):
    all_lats = [v["lat"] for v in vehicles] + [f["lat"] for f in flight]
    all_lons = [v["lon"] for v in vehicles] + [f["lon"] for f in flight]
    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]

    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap", control_scale=True)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
        max_zoom=22,
    ).add_to(m)

    # ─── Drone flight path layer ────────────────────────────────────────────
    fg_flight = folium.FeatureGroup(name=f"Flight path ({len(flight)} photos)", show=True)
    if flight:
        coords = [(f["lat"], f["lon"]) for f in flight]
        folium.PolyLine(coords, color="#1f77b4", weight=2, opacity=0.55,
                        tooltip="Drone flight path").add_to(fg_flight)
        for f in flight:
            folium.CircleMarker(
                location=(f["lat"], f["lon"]),
                radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{f['image']}</b><br>"
                    f"alt: {f['altitude']:.1f} m<br>"
                    f"yaw: {f['yaw']:.1f}&deg;<br>"
                    f"pitch: {f['pitch']:.1f}&deg;",
                    max_width=200),
            ).add_to(fg_flight)
    fg_flight.add_to(m)

    # ─── Vehicle markers, split by confidence (sighting count) ─────────────
    bins = [("High (>=3 sightings)", "#d62728", 3, 999),
            ("Medium (2 sightings)", "#ff7f0e", 2, 2),
            ("Low (1 sighting)",     "#7f7f7f", 1, 1)]
    for label, color, lo, hi in bins:
        fg = folium.FeatureGroup(name=label, show=True)
        for v in vehicles:
            if not (lo <= v["sightings"] <= hi):
                continue
            radius = min(11, 4 + v["sightings"] // 2)
            srcs = v["sources"]
            src_html = "<br>".join(srcs[:8]) + (f"<br>...(+{len(srcs)-8} more)" if len(srcs) > 8 else "")
            popup = (
                f"<b>{v['label']}</b><br>"
                f"class: {v['cls']}<br>"
                f"sightings: <b>{v['sightings']}</b><br>"
                f"lat: {v['lat']:.7f}<br>"
                f"lon: {v['lon']:.7f}<br>"
                f"TWD97: {v['twd97_x']:.1f}, {v['twd97_y']:.1f}<br>"
                f"<small>sources:<br>{src_html}</small>"
            )
            folium.CircleMarker(
                location=(v["lat"], v["lon"]),
                radius=radius, color=color, weight=1,
                fill=True, fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(popup, max_width=320),
                tooltip=f"{v['label']} ({v['sightings']}x)",
            ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ─── Static legend / header ─────────────────────────────────────────────
    n_hi = sum(1 for v in vehicles if v["sightings"] >= 3)
    n_md = sum(1 for v in vehicles if v["sightings"] == 2)
    n_lo = sum(1 for v in vehicles if v["sightings"] == 1)
    legend = f"""
    <div style="position: fixed; bottom: 16px; left: 16px; background: rgba(255,255,255,0.95);
                padding: 10px 14px; border: 1px solid #888; border-radius: 6px;
                z-index: 9999; font-family: sans-serif; font-size: 12px; line-height: 1.5;">
      <b style="font-size: 13px;">UAV Illegal Parking — NTNU</b><br>
      <span style="color:#d62728;">&#9679;</span> high confidence (&ge;3 sightings): {n_hi}<br>
      <span style="color:#ff7f0e;">&#9679;</span> medium (2 sightings): {n_md}<br>
      <span style="color:#7f7f7f;">&#9679;</span> low (1 sighting, possibly FP): {n_lo}<br>
      <span style="color:#1f77b4;">&#11044;</span> drone flight path ({len(flight)} photos)<br>
      <hr style="margin:6px 0;">
      <b>Total unique vehicles: {len(vehicles)}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    m.save(out_html)
    return out_html


def main():
    csv_path  = sys.argv[1] if len(sys.argv) > 1 else "./output/batch_v2/illegal_vehicles_gps.csv"
    image_dir = sys.argv[2] if len(sys.argv) > 2 else r"D:\UAV\C"
    out_html  = sys.argv[3] if len(sys.argv) > 3 else "./output/batch_v2/map.html"

    vehicles = load_vehicles(csv_path)
    print(f"Loaded {len(vehicles)} unique vehicles from {csv_path}")
    flight = load_flight(image_dir)
    print(f"Loaded {len(flight)} flight points from {image_dir}")

    out = build_map(vehicles, flight, out_html)
    print(f"\nMap saved: {out}")
    print(f"Open in browser:  file://{Path(out).absolute()}")


if __name__ == "__main__":
    main()
