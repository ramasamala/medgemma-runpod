"""
MedGemma 4B RunPod Serverless Handler.
Accepts a base64-encoded medical image + prompt, returns clinical analysis text.
"""

import runpod
import torch
import base64
import io
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"

print(f"[medgemma] Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"[medgemma] Model loaded on {model.device}")

DEFAULT_PROMPT = (
    "You are an expert medical imaging analyst. "
    "Analyze this medical image in detail. Describe the imaging modality, "
    "anatomical region, any notable findings, abnormalities, or observations. "
    "Provide a structured assessment."
)


def handler(job):
    try:
        inp = job["input"]
        image_b64 = inp.get("image_base64")
        prompt = inp.get("prompt", DEFAULT_PROMPT)
        max_tokens = inp.get("max_tokens", 1024)

        if not image_b64:
            return {"error": "image_base64 is required"}

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Build chat messages for MedGemma
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
            )
        inference_ms = int((time.time() - t0) * 1000)

        # Decode only the generated tokens (skip the input)
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = processor.decode(generated, skip_special_tokens=True)

        return {
            "text": text,
            "inference_time_ms": inference_ms,
            "model": MODEL_ID,
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
