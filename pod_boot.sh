#!/usr/bin/env bash
# Bring a freshly rented Pod up to serving state.
#
# Runs on a stock PyTorch image rather than a custom one: the Serverless build
# lives in RunPod's internal registry, and relying on a Pod being able to pull
# from there is an assumption this setup does not need to make. Everything here
# comes from public sources.
set -euo pipefail

REPO="${POD_REPO:-https://github.com/zeerqq1/qwen-tts-worker}"
DIR=/workspace/worker

echo "[boot] системные пакеты"
apt-get update -qq
apt-get install -y -qq --no-install-recommends git libsndfile1 ffmpeg curl

echo "[boot] код из $REPO"
rm -rf "$DIR"
git clone -q --depth 1 "$REPO" "$DIR"
cd "$DIR"

echo "[boot] python-зависимости"
# --extra-index-url matters: if anything re-resolves torch, pip must pick the
# CUDA wheel. A CPU build would run many times slower with no visible error.
python -m pip install -q --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-pod.txt
python -c "import torch; assert torch.version.cuda, 'torch потерял CUDA'; \
print('[boot] torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[boot] запуск сервера"
exec python -u pod_server.py
