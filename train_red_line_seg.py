"""
Train a YOLOv8-seg model to detect red curb lines in UAV images.
Uses a small dataset annotated in Roboflow with polygon masks.
"""
import os
from ultralytics import YOLO

def main():
    # Use the nano segmentation model as base (fast, good for small datasets)
    model = YOLO("yolov8n-seg.pt")

    # Dataset path (absolute to avoid path issues)
    data_yaml = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "red_line_dataset", "data.yaml"
    ))

    print(f"Dataset config: {data_yaml}")
    print("Starting training...")

    results = model.train(
        data=data_yaml,
        epochs=100,         # More epochs for small dataset
        imgsz=640,          # Matches Roboflow preprocessing
        batch=8,            # Small batch for small dataset
        patience=20,        # Early stopping if no improvement for 20 epochs
        device=0,           # Use GPU
        project="runs/segment",
        name="red_line_seg",
        exist_ok=True,
        # Augmentation (complement Roboflow augmentations)
        hsv_h=0.0,          # No hue shift (preserve red color)
        hsv_s=0.3,          # Saturation variation (faded vs bright lines)
        hsv_v=0.2,          # Value/brightness variation
        flipud=0.5,         # Vertical flip
        fliplr=0.5,         # Horizontal flip
        mosaic=0.5,         # Mosaic augmentation
        scale=0.3,          # Scale variation
        translate=0.1,      # Translation
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
