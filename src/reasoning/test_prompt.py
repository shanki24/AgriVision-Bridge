from prompt_builder import build_prompt

yolo_output = {
    "disease_label": "Tomato___Early_blight",
    "confidence_score": 0.9934
}

prompt = build_prompt(yolo_output)
print(prompt)