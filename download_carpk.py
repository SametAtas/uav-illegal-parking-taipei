import os
import sys
import subprocess

def install_requirements():
    print("Checking and installing required packages (roboflow, ultralytics)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "ultralytics"])
    print("Packages installed successfully!\n")

def download_dataset():
    # Attempt to import Roboflow, install if it fails
    try:
        # pyrefly: ignore [missing-import]
        from roboflow import Roboflow
    except ImportError:
        install_requirements()
        # pyrefly: ignore [missing-import]
        from roboflow import Roboflow

    # ==============================================================================
    # ⚠️ IMPORTANT: REPLACE THE STRING BELOW WITH YOUR ACTUAL ROBOFLOW API KEY ⚠️
    # ==============================================================================
    YOUR_API_KEY = "y6qROlnm95ZCK6G43IHw"

    if YOUR_API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("ERROR: You must insert your rea  l Roboflow API Key into the script!")
        print("Please edit download_carpk.py, change YOUR_API_KEY, and run it again.")
        sys.exit(1)

    print("Authenticating with Roboflow...")
    rf = Roboflow(api_key=YOUR_API_KEY)
    
    print("Locating NTU CARPK Dataset...")
    # Using a verified, stable repository for CARPK
    project = rf.workspace("bubbliiiing").project("carpk")
    version = project.version(1)

    print("Downloading dataset formatted for YOLOv8...")
    # This will automatically download and extract the dataset into a folder
    dataset = version.download("yolov8")

    print("\nDownload Complete!")
    print(f"Dataset saved to: {dataset.location}")

if __name__ == "__main__":
    download_dataset()
