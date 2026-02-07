from ultralytics import YOLO
import torch
import time
import os


# CONFIG

MODEL_PATH = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\runs\classify\train4\weights\best.pt"
IMAGE_PATH = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\data\processed\plantvillage\val\Apple___Apple_scab\0ea78733-9404-4536-8793-a108c66269b3___FREC_Scab 3145.JPG"

NUM_RUNS = 10  # average over multiple runs


def benchmark(device: str):
    print(f"\n Benchmarking on {device.upper()}")

    model = YOLO(MODEL_PATH)
    model.to(device)

    # Warm-up (important for GPU)
    for _ in range(3):
        model(IMAGE_PATH)

    times = []

    for _ in range(NUM_RUNS):
        start = time.time()
        model(IMAGE_PATH)
        end = time.time()
        times.append(end - start)

    avg_time_ms = (sum(times) / len(times)) * 1000
    print(f"Average inference time on {device.upper()}: {avg_time_ms:.2f} ms")

    return avg_time_ms


if __name__ == "__main__":
    assert os.path.exists(IMAGE_PATH), "Test image not found"

    cpu_time = benchmark("cpu")

    if torch.cuda.is_available():
        gpu_time = benchmark("cuda")

        speedup = cpu_time / gpu_time
        print(f"\n GPU Speedup: {speedup:.2f}× faster than CPU")
    else:
        print("CUDA not available")
