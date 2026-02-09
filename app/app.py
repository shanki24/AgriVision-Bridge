import sys
import os
import time
import torch

# Fix Python path to allow src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
from PIL import Image
import tempfile
import numpy as np
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2

# Vision layer
from ultralytics import YOLO

# Reasoning prompt (SAFE to import)
from src.reasoning.prompt_builder import build_prompt

# --------------------------------------------------
# CONFIG (GitHub-safe paths)
# --------------------------------------------------
MODEL_PATH = "runs/classify/train/weights/best.pt"
LLM_MODEL = "llama3"

# Load model
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# OVERLAY LOGIC
# Healthy  -> single GREEN box
# Disease  -> multiple RED boxes
# --------------------------------------------------
def overlay_boxes(image_pil, label, confidence):
    image = np.array(image_pil)
    output = image.copy()
    h, w, _ = image.shape

    # HEALTHY
    if "healthy" in label.lower():
        cv2.rectangle(
            output,
            (int(0.1 * w), int(0.1 * h)),
            (int(0.9 * w), int(0.9 * h)),
            (0, 255, 0),
            4
        )
        cv2.putText(
            output,
            f"{label} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )
        return output

    # DISEASED — lesion segmentation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, _ = cv2.split(lab)
    _, thresh = cv2.threshold(a, 135, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opening, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        mask = np.uint8(markers == marker_id)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh > 60:
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 0, 255),
                    2
                )

    cv2.putText(
        output,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    return output

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="AgriVision-Bridge", layout="centered")

st.title("AgriVision-Bridge")
st.subheader("AI-Powered Crop Disease Diagnosis")

st.write(
    "Upload a crop leaf image to detect diseases, visualize affected regions, "
    "and receive an AI-generated diagnostic summary."
)

uploaded_file = st.file_uploader(
    "Upload a crop leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=700)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    if st.button("Analyze Crop"):
        with st.spinner("Running disease detection..."):
            start = time.time()
            results = model(image_path)
            latency_ms = (time.time() - start) * 1000

            top_class = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            disease_label = model.names[top_class]

        st.success("Disease detection completed")

        # ---------------- Vision Output ----------------
        st.markdown("### Vision Layer Output")
        st.write(f"**Detected Disease:** {disease_label}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        overlay_img = overlay_boxes(image, disease_label, confidence)
        st.image(
            overlay_img,
            caption="Disease Spot Visualization (Explainable AI)",
            width=700
        )

        # ---------------- Diagnosis (Cloud-safe) ----------------
        st.markdown("### Final Diagnosis & Action Plan")

        prompt = build_prompt({
            "disease_label": disease_label,
            "confidence_score": confidence
        })

        st.info(
            "Local LLM execution (Ollama) is disabled on Streamlit Cloud.\n\n"
            "Below is a structured diagnostic summary generated from the reasoning prompt."
        )

        st.write(prompt)

        # ---------------- Evaluation Metrics ----------------
        st.markdown("### Evaluation Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detection Confidence (Accuracy Proxy)", f"{confidence:.2f}")
            st.write("**mAP:** Reported during offline training (classification task)")
        with col2:
            st.metric("Inference Latency (ms)", f"{latency_ms:.2f}")
            st.metric(
                "Inference Device",
                "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            )

        st.markdown("**Prompt Robustness**")
        if confidence >= 0.85:
            st.success("High confidence → definitive diagnosis & action plan")
        elif confidence >= 0.6:
            st.warning("Moderate confidence → cautious recommendations")
        else:
            st.info("Low confidence → expert consultation advised")

        st.markdown("**End-to-End Integration Quality**")
        st.write(
            "Vision Layer → YOLO Classification → Lesion Visualization → "
            "LLM Reasoning (Local) → Streamlit UI"
        )

        os.remove(image_path)
