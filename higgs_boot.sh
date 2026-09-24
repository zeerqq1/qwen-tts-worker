#!/usr/bin/env bash
# Bring a rented Pod up serving Higgs TTS 3 behind the studio's own wrapper.
#
# Same shape as pod_boot.sh, and for the same reasons: everything it prints goes
# to /workspace/boot.log, that file is served on the pod's port until the real
# server takes over (RunPod exposes no container logs through its API), every
# network step retries, and a hard deadline deletes the pod through the RunPod
# API if the boot never finishes — exiting would only make RunPod restart the
# container and keep billing.
#
# Layout on the pod:
#   :8001  sgl-omni serve  (SGLang-Omni, the model)
#   :8000  higgs_server.py (the studio's contract: /health /boot.log /generate)
set -uo pipefail

mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
PY=$(command -v python3 || command -v python)
"$PY" -m http.server "${POD_PORT:-8000}" --bind 0.0.0.0 --directory /workspace >/dev/null 2>&1 &
STATUS_PID=$!
echo "[boot] $(date -u +%H:%M:%S) старт · $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi недоступен') · воркер ${POD_WORKER_REVISION:-main}"

REPO_TAR="${POD_REPO_TAR:-https://codeload.github.com/zeerqq1/qwen-tts-worker/tar.gz/refs/heads/main}"
DIR=/workspace/worker
MODEL_ID="${HIGGS_MODEL_ID:-bosonai/higgs-tts-3-4b}"
UP_PORT="${HIGGS_UPSTREAM_PORT:-8001}"
REFS_DIR="${HIGGS_REFS_DIR:-/workspace/refs}"
PIP="$PY -m pip install -q --no-cache-dir --break-system-packages"

log() { echo "[boot] $*" >&2; }

MARKS=/workspace/boot_times.txt
mark() { mkdir -p /workspace; echo "$1 $(date +%s)" >> "$MARKS"; }
mark script_start

# The two long steps (image pull, weights) print nothing while they run, and a
# quiet pod is a pod the studio writes off. Heartbeat until the hand-over.
( while :; do sleep 20; echo "[boot] ...идёт загрузка"; done ) &
HEARTBEAT_PID=$!

self_terminate() {
    "$PY" - "$1" <<'PY'
import os, sys, urllib.request
pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
print("[boot] " + sys.argv[1] + " — снимаю под", file=sys.stderr, flush=True)
if pod and key:
    for method, url in (("DELETE", "https://rest.runpod.io/v1/pods/" + pod),
                        ("POST", "https://rest.runpod.io/v1/pods/" + pod + "/stop")):
        try:
            req = urllib.request.Request(url, method=method, headers={"Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=30) as r:
                print("[boot] " + method + " -> " + str(r.status), file=sys.stderr, flush=True)
            break
        except Exception as e:
            print("[boot] " + method + " не удался: " + str(e), file=sys.stderr, flush=True)
PY
}
( sleep "${POD_LOAD_TIMEOUT_SEC:-1800}"; self_terminate "загрузка не уложилась в ${POD_LOAD_TIMEOUT_SEC:-1800} с"; ) &
DEADLINE_PID=$!

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "$n" -ge 3 ]; then log "ПРОВАЛ после 3 попыток: $*"; return 1; fi
        log "попытка $n не удалась, повтор: $*"
        sleep 3
    done
    return 0
}

# ── 1. The studio's own code (wrapper + shared audio shaping). ──
fetch_code() {
    "$PY" - "$REPO_TAR" "$DIR" <<'PY'
import io, os, shutil, sys, tarfile, urllib.request
url, dest = sys.argv[1], sys.argv[2]
blob = urllib.request.urlopen(url, timeout=90).read()
tmp = dest + ".new"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
    tar.extractall(tmp)
inner = os.path.join(tmp, os.listdir(tmp)[0])
shutil.rmtree(dest, ignore_errors=True)
shutil.move(inner, dest)
shutil.rmtree(tmp, ignore_errors=True)
print("[boot] код: " + ", ".join(sorted(os.listdir(dest))[:8]), file=sys.stderr)
PY
}
retry fetch_code || exit 1
mark code_done
cd "$DIR" || exit 1
mkdir -p "$REFS_DIR"

# ── 2. Weights, in the background: the single biggest item starts first. ──
# The SGLang image is Debian-based, and some of what the wrapper needs (blinker,
# behind flask) is installed by apt. pip then refuses: "cannot uninstall, it is
# a distutils installed project". --ignore-installed leaves the system copy
# alone and puts ours beside it, which is exactly right inside a container.
install_deps() {
    $PIP huggingface_hub hf_transfer flask requests soundfile numpy && return 0
    log "pip споткнулся о системный пакет — повторяю с --ignore-installed"
    $PIP --ignore-installed huggingface_hub hf_transfer flask requests soundfile numpy
}
log "ставлю загрузчик"
retry install_deps || exit 1
mark pip_hf_done
export HF_HUB_ENABLE_HF_TRANSFER=1
log "тяну веса $MODEL_ID (фоном)"
(
    for attempt in 1 2 3; do
        "$PY" - "$MODEL_ID" <<'PY' && exit 0
import sys
from huggingface_hub import snapshot_download
print("[boot] веса в", snapshot_download(sys.argv[1], max_workers=8), file=sys.stderr)
PY
        echo "[boot] веса: попытка $attempt не удалась" >&2
        sleep 5
    done
    exit 1
) &
MODEL_PID=$!
mark download_started

# ── 3. The server binary. The SGLang image ships it, but not always on PATH:
#       the published recipe builds it into a uv venv inside the container. ──
if ! command -v sgl-omni >/dev/null 2>&1; then
    for v in /sgl-workspace/sglang-omni/.venv /sgl-workspace/.venv /opt/sglang-omni/.venv /workspace/.venv /root/.venv; do
        if [ -x "$v/bin/sgl-omni" ]; then export PATH="$v/bin:$PATH"; log "sgl-omni найден в $v"; break; fi
    done
fi
if ! command -v sgl-omni >/dev/null 2>&1; then
    log "sgl-omni не найден в образе — ставлю из git (это долго)"
    retry git clone --depth 1 https://github.com/sgl-project/sglang-omni /opt/sglang-omni \
        && retry $PIP -e /opt/sglang-omni || { log "ПРОВАЛ: не удалось поставить sglang-omni"; kill $MODEL_PID 2>/dev/null; exit 1; }
fi
log "sgl-omni: $(command -v sgl-omni)"
mark pip_done

log "жду веса"
if ! wait $MODEL_PID; then
    log "ПРОВАЛ: веса не скачались"
    exit 1
fi
mark weights_done

# ── 4. The model on the private port, the wrapper on the public one. ──
log "поднимаю SGLang-Omni на :$UP_PORT"
sgl-omni serve --model-path "$MODEL_ID" --host 127.0.0.1 --port "$UP_PORT" \
    --allowed-local-media-path "$REFS_DIR" &
SGL_PID=$!

for i in $(seq 1 240); do
    if "$PY" -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:${UP_PORT}/v1/models', timeout=5)" 2>/dev/null; then
        log "SGLang отвечает"
        break
    fi
    if ! kill -0 "$SGL_PID" 2>/dev/null; then
        log "ПРОВАЛ: sgl-omni завершился на старте"
        exit 1
    fi
    sleep 5
done
mark model_ready

# Hand the public port over: the placeholder must let go before the wrapper binds.
kill "$HEARTBEAT_PID" 2>/dev/null; wait "$HEARTBEAT_PID" 2>/dev/null
kill "$DEADLINE_PID" 2>/dev/null; wait "$DEADLINE_PID" 2>/dev/null
kill "$STATUS_PID" 2>/dev/null; wait "$STATUS_PID" 2>/dev/null
for i in 1 2 3 4 5; do
    "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "${POD_PORT:-8000}" || break
    log "порт ещё занят раздатчиком, жду"; sleep 1
done

log "поднимаю обёртку на :${POD_PORT:-8000}"
exec "$PY" -u higgs_server.py
