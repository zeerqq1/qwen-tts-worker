# Qwen3-TTS serverless worker.
#
# The model is baked into the image on purpose. The alternative — a RunPod
# network volume — costs money every month whether or not anything runs, and
# leaves state behind that has to be cleaned up. A 4GB layer inside the image
# is cached on each worker after its first pull and costs nothing between jobs.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # expandable segments keeps allocator fragmentation from eating the
    # headroom the batch sizer is counting on
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    QWEN_MODEL_DIR=/models/qwen3-tts-1.7b-base \
    QWEN_COMPILE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the weights in. --local-dir-use-symlinks=False writes real files so the
# layer is self-contained and no HF cache lookup happens at runtime.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', \
                  local_dir='/models/qwen3-tts-1.7b-base', \
                  local_dir_use_symlinks=False)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
