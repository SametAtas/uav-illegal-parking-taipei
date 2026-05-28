import os
import argparse
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NTU CARPK Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--name", type=str, default="yolov8m_carpk", help="Name of the training run")
    args = parser.parse_args()

    print("Initializing YOLOv8 training...")
    
    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Training device: {device.upper()}")
    
    epochs = args.epochs
    if device == "cpu" and epochs > 3:
        print("\n⚠️ WARNING: You are training on a CPU!")
        print("Training on 6,000+ images on a CPU could take days or weeks.")
        print("We are setting epochs=3 just to verify the pipeline works.")
        epochs = 3

    # Load the pre-trained VisDrone model as our starting point
    model_path = "yolov8m-visdrone.pt"
    
    if not os.path.exists(model_path):
        # Fallback to standard yolov8m if the local cache path fails
        model_path = "yolov8m.pt"
        
    model = YOLO(model_path)

    # Path to the dataset configuration file we just fixed
    dataset_yaml = "CARPK-1/extracted/CarPK/CarPK/CarPK.yaml"

    print(f"\nStarting Training (epochs={epochs}, imgsz=1024, batch=2, device={device})...")
    # Start training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=1024,
        device=device,
        batch=2, # Lower batch size to prevent out-of-memory errors
        project="output/carpk_training",
        name=args.name
    )
    
    print("\nTraining Complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
