#!/usr/bin/env bash
# Bring a freshly rented Pod up to serving state.
#
# Boot used to be a chain: apt-get, then git clone, then pip install, then a
# 4.2GB model download, then load. Five and a half minutes of it, and each link
# was both minutes of billing on every pod in the batch and its own chance to
# fail — roughly one pod in five never reached serving at all.
#
# So: nothing waits for anything it does not actually need. The model download
# and the Python packages come down at the same time, apt is gone from the
# critical path entirely (git is replaced by a stdlib tarball fetch, and
# libsndfile ships inside the soundfile wheel), and every network step retries
# instead of killing the pod on one bad response.
#
# The pod runs the same engine.py as Serverless, so the audio is identical.
set -uo pipefail

# Everything this script prints goes to a file, and that file is served on the
# pod's port until the real server takes over. RunPod exposes no container
# logs through its API, so before this a pod that never came up was a pod
# that never explained itself — and roughly one in five did not come up.
mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
# Started directly, not inside "cd && ..." — that would background a
# subshell, $! would name the subshell, and killing it later would leave
# the python child holding the port the real server needs.
python -m http.server "${POD_PORT:-8000}" --bind 0.0.0.0 --directory /workspace >/dev/null 2>&1 &
STATUS_PID=$!
echo "[boot] $(date -u +%H:%M:%S) старт · $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi недоступен')"

REPO_TAR="${POD_REPO_TAR:-https://codeload.github.com/zeerqq1/qwen-tts-worker/tar.gz/refs/heads/main}"
DIR=/workspace/worker
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
MODEL_DIR="${QWEN_MODEL_DIR:-/workspace/qwen3-tts-1.7b-base}"
# With a value, not by name. "export QWEN_MODEL_DIR" exported nothing: the path
# had gone into MODEL_DIR, and bash does not put an unset variable into a
# child's environment at all. engine.py then read "" and fell back to the repo
# id, so every pod downloaded the same 4.2GB a SECOND time inside
# from_pretrained — anonymously, and without the retries the snapshot below
# has. That second download is what left four pods out of five serving /health
# with model_loaded=false.
export QWEN_MODEL_DIR="$MODEL_DIR"
export QWEN_MODEL_ID="$MODEL_ID"
PIP="python -m pip install -q --no-cache-dir --break-system-packages"

log() { echo "[boot] $*" >&2; }

# The length of this log is the only progress signal the studio has, and the
# two longest steps — pip -q and snapshot_download — print nothing at all while
# they run. A minute of that on a slow host is indistinguishable from a wedged
# boot, and gets the machine destroyed for being healthy but quiet. Killed
# before the hand-over, so that a pod which stops working stops printing too.
( while :; do sleep 20; echo "[boot] ...идёт загрузка"; done ) &
HEARTBEAT_PID=$!

# Where the minutes actually go. Written as epoch seconds and served by
# /health, so boot can be measured instead of guessed at — including the
# image pull, which happens before this script exists and is only visible
# as the gap between renting the pod and the first mark here.
MARKS=/workspace/boot_times.txt
mark() { mkdir -p /workspace; echo "$1 $(date +%s)" >> "$MARKS"; }
mark script_start

# Hard deadline for the whole boot. A boot wedged on its image pull, pip or
# weights used to run forever if the studio was gone — exiting does not help,
# RunPod restarts the container — so the pod deletes itself through the API
# (RunPod injects RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY). Cancelled
# just before the real server takes over; pod_server has its own clocks.
self_terminate() {
    python - "$1" <<'PY'
import os, sys, time, urllib.request
pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
print("[boot] " + sys.argv[1] + " — снимаю под", file=sys.stderr, flush=True)
if pod and key:
    for method, url in (("DELETE", f"https://rest.runpod.io/v1/pods/{pod}"),
                        ("POST", f"https://rest.runpod.io/v1/pods/{pod}/stop")):
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

# Retry the things that touch the network. A single transient failure used to
# cost the whole pod; three tries cost seconds.
retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        if [ "$n" -ge 3 ]; then
            log "ПРОВАЛ после 3 попыток: $*"
            return 1
        fi
        log "попытка $n не удалась, повтор: $*"
        sleep 3
    done
    return 0
}

# ── 1. Code. urllib instead of git, so apt is not on the critical path. ──
fetch_code() {
    python - "$REPO_TAR" "$DIR" <<'PY'
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

# ── 2. Weights, in the background. The single biggest item, so it starts first
#       and everything else happens while it downloads. hf_transfer opens
#       several connections instead of one. ──
log "ставлю загрузчик"
retry $PIP huggingface_hub hf_transfer || exit 1
mark pip_hf_done

export HF_HUB_ENABLE_HF_TRANSFER=1
log "тяну веса $MODEL_ID -> $MODEL_DIR (фоном)"
(
    for attempt in 1 2 3; do
        python - "$MODEL_ID" "$MODEL_DIR" <<'PY' && exit 0
import os, sys
from huggingface_hub import snapshot_download
# The studio sends the exact snapshot it runs locally; without it a new push
# to the repo would change the pods' voice while the local cache kept the old.
rev = os.environ.get("QWEN_MODEL_REVISION") or None
snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=8, revision=rev)
PY
        echo "[boot] веса: попытка $attempt не удалась" >&2
        sleep 5
    done
    exit 1
) &
MODEL_PID=$!
mark download_started

# ── 3. Python packages, at the same time as the download above. torch and
#       torchaudio come from the base image and are deliberately not listed:
#       if pip re-resolves torch it can put a CPU build over the CUDA one and
#       the pod then runs many times slower with nothing to notice. ──
log "python-зависимости"
retry $PIP --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements-pod.txt || { kill $MODEL_PID 2>/dev/null; exit 1; }

python - <<'PY' || exit 1
import torch, importlib.metadata as m
assert torch.version.cuda, "torch потерял CUDA — под работал бы на процессоре"
print("[boot] torch", torch.__version__, "cuda", torch.version.cuda,
      "qwen-tts", m.version("qwen-tts"), "transformers", m.version("transformers"))
PY

# soundfile normally carries libsndfile inside its wheel; apt is only touched
# on the rare image where it does not, and only then.
if ! python -c "import soundfile" 2>/dev/null; then
    log "soundfile без библиотеки — доставляю libsndfile1 через apt"
    apt-get update -qq && apt-get install -y -qq --no-install-recommends libsndfile1
fi

# ── 4. Both halves have to be there before the model can load. ──
mark pip_done
log "жду веса"
if ! wait $MODEL_PID; then
    log "ПРОВАЛ: веса не скачались"
    exit 1
fi
mark weights_done
log "всё на месте, поднимаю сервер"

# Hand the port over to the real server, and make sure it is actually free:
# a port still held by the placeholder would keep this pod "not ready"
# forever while billing.
kill "$HEARTBEAT_PID" 2>/dev/null; wait "$HEARTBEAT_PID" 2>/dev/null
kill "$DEADLINE_PID" 2>/dev/null; wait "$DEADLINE_PID" 2>/dev/null
kill "$STATUS_PID" 2>/dev/null; wait "$STATUS_PID" 2>/dev/null
for i in 1 2 3 4 5; do
    # connect_ex is 0 when something still answers on the port. Exit 0 then
    # (keep waiting); exit 1 once nothing answers, which is what breaks.
    python -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "${POD_PORT:-8000}" || break
    log "порт ещё занят раздатчиком, жду"; sleep 1
done
exec python -u pod_server.py
