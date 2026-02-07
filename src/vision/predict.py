from ultralytics import YOLO
import os

# Path to trained YOLO classification model
MODEL_PATH = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\runs\classify\train4\weights\best.pt"

model = YOLO(MODEL_PATH)

def detect_disease(image_path: str):
    """
    Detects crop disease from an image and returns label + confidence
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image_path)

    # Extract top prediction
    top_class_index = results[0].probs.top1
    confidence = results[0].probs.top1conf.item()
    disease_label = model.names[top_class_index]

    return {
        "disease_label": disease_label,
        "confidence_score": round(confidence, 4)
    }


if __name__ == "__main__":
    # Use any real image from your dataset
    test_image = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\data\processed\plantvillage\val\Apple___Apple_scab\0ea78733-9404-4536-8793-a108c66269b3___FREC_Scab 3145.JPG"

    output = detect_disease(test_image)
    print("Disease Detection Result:")
    print(output)
