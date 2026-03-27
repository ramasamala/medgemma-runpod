FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    "transformers>=4.52.0" \
    accelerate \
    Pillow \
    sentencepiece \
    protobuf

# HuggingFace token for gated model access (MedGemma requires license acceptance)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download model weights at build time to avoid cold-start downloads
RUN python -c "\
from huggingface_hub import login; \
import os; \
login(token=os.environ['HF_TOKEN']); \
from transformers import AutoProcessor, AutoModelForImageTextToText; \
AutoProcessor.from_pretrained('google/medgemma-4b-it'); \
AutoModelForImageTextToText.from_pretrained('google/medgemma-4b-it', torch_dtype='auto')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
