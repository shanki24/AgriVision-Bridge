from prompt_builder import build_prompt
from llm_engine import run_local_llm

# Simulate low-confidence YOLO output
yolo_output = {
    "disease_label": "Tomato___Leaf_Mold",
    "confidence_score": 0.42
}

prompt = build_prompt(yolo_output)
response = run_local_llm(prompt, model_name="llama3")

print("\n🧠 Low-Confidence Diagnostic Report:\n")
print(response)