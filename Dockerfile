# Qwen3-TTS serverless worker.
#
# The model is baked into the image on purpose. The alternative — a RunPod
# network volume — costs money every month whether or not anything runs, and
# leaves state behind that has to be cleaned up. A 4GB layer inside the image
# is cached on each worker after its first pull and costs nothing between jobs.
#
# Base image is the official PyTorch build pinned to the SAME torch version the
# studio runs locally (2.11.0 + cu128), so the cloud behaves like the machine
# this was tested on. The -devel variant carries the full CUDA toolkit, which
# torch.compile needs to generate kernels — that compile step is where most of
# the cloud speedup comes from, so it is not worth risking a smaller image.

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # The base image ships a PEP 668 "externally managed" Python, so pip
    # refuses to install system-wide without this. There is no OS package
    # manager to defer to inside a single-purpose container, and a venv would
    # only hide the interpreter that already carries the CUDA torch build.
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    # Keeps allocator fragmentation from eating the headroom the batch sizer
    # counts on — the same setting the local studio uses.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    QWEN_MODEL_DIR=/models/qwen3-tts-1.7b-base \
    QWEN_COMPILE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# --extra-index-url matters: if any dependency re-resolves torch, pip must pick
# the CUDA wheel. Without it a CPU build can silently replace the GPU one and
# the worker runs at a fraction of the speed with no visible error.
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /app/requirements.txt && \
    python -c "import torch; assert torch.version.cuda, 'torch lost CUDA support'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Bake the weights in, so a cold worker never waits on HuggingFace.
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/models/qwen3-tts-1.7b-base'); \
print('model at', p)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
