from ultralytics import YOLO
import torch
import time

DATASET_PATH = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\data\processed\plantvillage"

def train_yolo():
    assert torch.cuda.is_available(), "CUDA not available"

    device = "cuda"
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolov8n-cls.pt")

    start_time = time.time()

    model.train(
        data=DATASET_PATH,
        epochs=20,
        imgsz=224,
        batch=32,         
        device=device,
        workers=0,         
        amp=True,
        cache=False        
    )

    end_time = time.time()
    print(f"⏱ Training time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    train_yolo()
