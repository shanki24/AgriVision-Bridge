import subprocess

def run_local_llm(prompt: str, model_name: str = "llama3") -> str:
    """
    Runs a local LLM via Ollama and returns the response text.
    """

    process = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )

    output, error = process.communicate(prompt)

    # Ollama writes logs to stderr even on success
    if not output.strip():
        raise RuntimeError(f"Ollama returned no output.\nStderr:\n{error}")

    return output.strip()


if __name__ == "__main__":
    test_prompt = """
    A crop image was analyzed.

    Crop: Tomato
    Disease: Early blight
    Confidence: 0.92

    Explain the disease and suggest treatment.
    """

    response = run_local_llm(test_prompt, model_name="llama3")
    print("\n🧠 LLM Response:\n")
    print(response)