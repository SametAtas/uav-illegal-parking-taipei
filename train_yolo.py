import os
import torch
from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 training...")
    
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected training device: {device.upper()}")
    
    if device == "cpu":
        print("\n⚠️ WARNING: You are training on a CPU!")
        print("Training on 6,000+ images on a CPU could take days or weeks.")
        print("We are setting epochs=3 just to verify the pipeline works.")
        epochs = 3
    else:
        print("GPU detected! Training will be much faster.")
        epochs = 20 # Adjust this higher (e.g., 50 or 100) for a fully trained production model

    # Load the pre-trained VisDrone model as our starting point
    model_path = "yolov8m-visdrone.pt"
    
    if not os.path.exists(model_path):
        # Fallback to standard yolov8m if the local cache path fails
        model_path = "yolov8m.pt"
        
    model = YOLO(model_path)

    # Path to the dataset configuration file we just fixed
    dataset_yaml = "CarPK.yaml"

    print("\nStarting Training...")
    # Start training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        batch=8, # Lower batch size to prevent out-of-memory errors
        project="output/carpk_training",
        name="yolov8m_carpk"
    )
    
    print("\nTraining Complete!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
