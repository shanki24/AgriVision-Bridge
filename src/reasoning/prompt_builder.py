def build_prompt(yolo_output: dict) -> str:
    """
    Builds a confidence-aware prompt for LLM reasoning
    """

    disease_label = yolo_output["disease_label"]
    confidence = yolo_output["confidence_score"]

    # Split crop and disease
    if "___" in disease_label:
        crop, disease = disease_label.split("___")
    else:
        crop = "Unknown crop"
        disease = disease_label

    # Confidence logic
    if confidence >= 0.85:
        confidence_level = "high"
        guidance = (
            "The detection confidence is high. "
            "Provide a clear diagnosis and recommended treatment steps."
        )

    elif confidence >= 0.60:
        confidence_level = "moderate"
        guidance = (
            "The detection confidence is moderate. "
            "Explain possible disease causes and suggest monitoring and preventive actions."
        )

    else:
        confidence_level = "low"
        guidance = (
            "The detection confidence is low. "
            "Do NOT give a definitive diagnosis. "
            "Suggest capturing clearer images or consulting an agricultural expert."
        )

    prompt = f"""
You are an agricultural expert AI assistant.

A computer vision system analyzed an image of a crop leaf and produced the following results:

Crop: {crop}
Detected Condition: {disease}
Confidence Score: {confidence:.2f} ({confidence_level} confidence)

Instructions:
{guidance}

Tasks:
1. Explain the detected condition in simple terms.
2. Recommend next steps appropriate for the confidence level.
3. Use farmer-friendly language.
4. Avoid making strong claims when confidence is low.

Response Guidelines:
- Keep the explanation clear and practical.
- Focus on actionable advice.
"""

    return prompt.strip()