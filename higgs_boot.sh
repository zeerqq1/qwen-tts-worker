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
# Why this image and not lmsysorg/sglang-omni:dev: that one is 17GB, and on
# community hosts three machines in a row spent twenty minutes pulling it and
# never finished. The stack is on PyPI with exact pins (sglang-omni 0.1.6 ->
# torch 2.13.0, sglang 0.5.19, flashinfer cu13), and PyTorch publishes a 3GB
# image with precisely that torch. So: small pull, deterministic pip, weights
# from Hugging Face, nothing to discover inside somebody else's container.
#
# Layout on the pod:
#   :8001  sgl-omni serve  (SGLang-Omni, the model)
#   :8000  higgs_server.py (the studio's contract: /health /boot.log /generate)
set -uo pipefail

mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
# The PyTorch images keep their interpreter in conda; fall back to whatever is
# on PATH. Chosen once, used for everything: pip, the server, the wrapper.
PY=""
for cand in /opt/conda/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    [ -x "$cand" ] && { PY="$cand"; break; }
done
[ -n "$PY" ] || PY=$(command -v python3 || command -v python)
"$PY" -m http.server "${POD_PORT:-8000}" --bind 0.0.0.0 --directory /workspace >/dev/null 2>&1 &
STATUS_PID=$!
echo "[boot] $(date -u +%H:%M:%S) старт · $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi недоступен') · воркер ${POD_WORKER_REVISION:-main} · python $("$PY" -c 'import sys; print(sys.version.split()[0])')"

REPO_TAR="${POD_REPO_TAR:-https://codeload.github.com/zeerqq1/qwen-tts-worker/tar.gz/refs/heads/main}"
DIR=/workspace/worker
MODEL_ID="${HIGGS_MODEL_ID:-bosonai/higgs-tts-3-4b}"
# A commit of the weights, not a branch: a push to the repository must not
# change the voice between two chapters of one project. Empty = whatever main is.
MODEL_REV="${HIGGS_MODEL_REVISION:-}"
MODEL_PATH_FILE=/workspace/model_path
SGL_OMNI_VERSION="${HIGGS_SGL_OMNI_VERSION:-0.1.6}"
UP_PORT="${HIGGS_UPSTREAM_PORT:-8001}"
REFS_DIR="${HIGGS_REFS_DIR:-/workspace/refs}"
# A wheel cache on the container disk, not --no-cache-dir: on a slow host a
# multi-gigabyte install that breaks on one wheel must not start over. pip
# also retries each request ten times with a generous read timeout, so a
# stalling mirror is waited out instead of failed.
export PIP_CACHE_DIR=/workspace/pipcache
PIP="$PY -m pip install -q --break-system-packages --retries 10 --timeout 120"

log() { echo "[boot] $*" >&2; }

MARKS=/workspace/boot_times.txt
mark() { mkdir -p /workspace; echo "$1 $(date +%s)" >> "$MARKS"; }
mark script_start

# The long steps print nothing while they run, and a quiet pod is a pod the
# studio writes off. Heartbeat until the hand-over.
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
# A boot that fails for good must not merely exit: RunPod would restart the
# container into the same failure and bill for every lap. Say why, end the pod.
MODEL_PID=0
die() { log "ПРОВАЛ: $*"; kill "$MODEL_PID" 2>/dev/null; self_terminate "$*"; exit 1; }

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "$n" -ge 3 ]; then log "не вышло после 3 попыток: $*"; return 1; fi
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
retry fetch_code || die "код воркера не скачался"
mark code_done
cd "$DIR" || die "нет папки воркера"
mkdir -p "$REFS_DIR"

# ── 2. Weights, in the background: the single biggest item starts first. ──
retry $PIP -U huggingface_hub || die "huggingface_hub не встал"
"$PY" -c "import hf_transfer" 2>/dev/null || $PIP hf_transfer >/dev/null 2>&1 || true
if "$PY" -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    log "веса качает hf_transfer"
else
    log "hf_transfer не встал — веса качает обычный загрузчик"
fi
mark pip_hf_done
log "тяну веса $MODEL_ID${MODEL_REV:+ @ ${MODEL_REV:0:8}} (фоном)"
(
    for attempt in 1 2 3; do
        "$PY" - "$MODEL_ID" "$MODEL_REV" "$MODEL_PATH_FILE" <<'PY' && exit 0
import sys
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1], sys.argv[2] or None, sys.argv[3]
path = snapshot_download(repo, revision=rev, max_workers=8)
open(out, "w").write(path)
print("[boot] веса в", path, file=sys.stderr)
PY
        echo "[boot] веса: попытка $attempt не удалась" >&2
        sleep 5
    done
    exit 1
) &
MODEL_PID=$!
mark download_started

# ── 3. The server stack, from PyPI, pinned. torch in this image already matches
#       sglang-omni's pin, so pip resolves the rest without touching it. ──
echo
log "torch в образе: $("$PY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || echo 'нет') · python $PY"
log "ставлю sglang-omni==$SGL_OMNI_VERSION и flask"
retry $PIP "sglang-omni==$SGL_OMNI_VERSION" flask requests soundfile \
    || die "sglang-omni==$SGL_OMNI_VERSION не установился (см. лог выше)"

# Whatever the server still asks for, by name: the module comes out of the
# ModuleNotFoundError, one package at a time, never a resolving reinstall.
pip_name() {
    case "$1" in
        zmq) echo pyzmq ;; PIL) echo pillow ;; cv2) echo opencv-python-headless ;;
        yaml) echo pyyaml ;; sklearn) echo scikit-learn ;; attr) echo attrs ;;
        google) echo protobuf ;; *) echo "$1" ;;
    esac
}
probe_imports() { "$PY" -c "import sglang_omni.models.higgs_tts, sglang_omni.cli" 2>&1; }
for _ in 1 2 3 4 5 6 7 8; do
    out=$(probe_imports) || true
    miss=$(printf '%s\n' "$out" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
    [ -z "$miss" ] && break
    pkg=$(pip_name "${miss%%.*}")
    log "серверу не хватает модуля $miss — ставлю $pkg"
    $PIP "$pkg" || $PIP --ignore-installed "$pkg" || break
done
if ! "$PY" -c "import sglang_omni.models.higgs_tts, sglang_omni.cli" 2>/dev/null; then
    log "плагин Higgs не импортируется:"; probe_imports | tail -8
    die "sglang_omni не импортируется"
fi
SGL_CMD="sgl-omni"; command -v sgl-omni >/dev/null 2>&1 || SGL_CMD="$PY -m sglang_omni.cli"
log "сервер: $SGL_CMD · sglang $("$PY" -c 'import importlib.metadata as m; print(m.version("sglang"))' 2>/dev/null || echo '?') · torch $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
mark pip_done

log "жду веса"
wait $MODEL_PID || die "веса не скачались"
MODEL_DIR=$(cat "$MODEL_PATH_FILE" 2>/dev/null)
[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] || die "снимок весов без config.json: '$MODEL_DIR'"
mark weights_done

# ── 4. The model on the private port, the wrapper on the public one. ──
log "поднимаю SGLang-Omni на :$UP_PORT"
$SGL_CMD serve --model-path "$MODEL_DIR" --host 127.0.0.1 --port "$UP_PORT" \
    --allowed-local-media-path "$REFS_DIR" &
SGL_PID=$!
SGL_TRIES=0
for i in $(seq 1 240); do
    if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${UP_PORT}/v1/models', timeout=5)" 2>/dev/null; then
        log "SGLang отвечает"
        break
    fi
    if ! kill -0 "$SGL_PID" 2>/dev/null; then
        miss=$(tail -80 /workspace/boot.log | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | tail -1)
        SGL_TRIES=$((SGL_TRIES + 1))
        if [ -n "$miss" ] && [ "$SGL_TRIES" -le 3 ]; then
            pkg=$(pip_name "${miss%%.*}")
            log "sgl-omni упал без модуля $miss — ставлю $pkg и пробую снова"
            $PIP "$pkg" || $PIP --ignore-installed "$pkg"
            $SGL_CMD serve --model-path "$MODEL_DIR" --host 127.0.0.1 --port "$UP_PORT" \
                --allowed-local-media-path "$REFS_DIR" &
            SGL_PID=$!
            continue
        fi
        die "sgl-omni завершился на старте"
    fi
    sleep 5
done
"$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${UP_PORT}/v1/models', timeout=5)" 2>/dev/null \
    || die "SGLang не ответил за 20 минут"
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
