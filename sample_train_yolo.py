import os
import torch
from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 training...")
    
    # Hardcode to GPU 1 for the sample run
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"Detected training device: {device.upper()}")
    
    if device == "cpu":
        print("\n⚠️ WARNING: You are training on a CPU!")
        print("Training on 6,000+ images on a CPU could take days or weeks.")
        print("We are setting epochs=3 just to verify the pipeline works.")
        epochs = 3
    else:
        print("GPU detected! Running a quick 5 epoch sample run.")
        epochs = 5

    # Load the pre-trained VisDrone model as our starting point
    model_path = "yolov8m-visdrone.pt"
    
    if not os.path.exists(model_path):
        # Fallback to standard yolov8m if the local cache path fails
        model_path = "yolov8m.pt"
        
    model = YOLO(model_path)

    # Path to the dataset configuration file we just fixed
    dataset_yaml = "CARPK-1/extracted/CarPK/CarPK/CarPK.yaml"

    print("\nStarting Training...")
    # Start training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=8, # Lower batch size to prevent out-of-memory errors
        project="output/carpk_training_sample",
        name="yolov8m_carpk_sample"
    )
    
    print("\nTraining Complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
