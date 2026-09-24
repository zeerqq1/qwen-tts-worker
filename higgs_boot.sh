#!/usr/bin/env bash
# Bring a rented Pod up serving Higgs TTS 3 behind the studio's own wrapper.
#
# Same shape as pod_boot.sh, and for the same reasons: everything it prints goes
# to /workspace/boot.log, that file is served on the pod's port until the real
# server takes over (RunPod exposes no container logs through its API), every
# network step retries, and a boot that cannot finish ends the pod — or, when
# the pod cannot end itself, goes quiet and lets the studio read why and do it.
#
# The image is lmsysorg/sglang-omni pinned by digest. It is 18GB, and the
# alternative was tried at length on 24.09.2026: a 3GB PyTorch image with the
# stack installed from PyPI. That route drifted at every layer — sgl-deep-ep
# asserting a CUDA toolkit on import, huggingface_hub 2.0 landing over
# transformers' pin on a restart, torch losing its GPU once the stack was in —
# and a slow host spent as long on pip as it would have on the pull. The big
# image is one immutable thing that has already served this model.
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
echo
echo "[boot] $(date -u +%H:%M:%S) старт · $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi недоступен') · воркер ${POD_WORKER_REVISION:-main}"

REPO_TAR="${POD_REPO_TAR:-https://codeload.github.com/zeerqq1/qwen-tts-worker/tar.gz/refs/heads/main}"
DIR=/workspace/worker
MODEL_ID="${HIGGS_MODEL_ID:-bosonai/higgs-tts-3-4b}"
# A commit of the weights, not a branch: a push to the repository must not
# change the voice between two chapters of one project. Empty = whatever main is.
MODEL_REV="${HIGGS_MODEL_REVISION:-}"
MODEL_PATH_FILE=/workspace/model_path
UP_PORT="${HIGGS_UPSTREAM_PORT:-8001}"
REFS_DIR="${HIGGS_REFS_DIR:-/workspace/refs}"
# A wheel cache on the container disk and patient retries: on a slow host a
# download that breaks must not start over.
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
    # The worker's own helper once the code is on disk; before that, the same
    # chain inline: REST, then the GraphQL mutations the pod-scoped key might
    # be accepted by. Measured 24.09.2026: the injected key got 403 from all
    # of them — so the caller must not count on this working.
    if [ -f "$DIR/pod_lifecycle.py" ]; then
        (cd "$DIR" && "$PY" -c "import os, sys, pod_lifecycle; print('[boot] ' + sys.argv[1] + ' — снимаю под', file=sys.stderr); sys.exit(0 if pod_lifecycle._end_pod(os.environ.get('RUNPOD_POD_ID',''), os.environ.get('RUNPOD_API_KEY','')) else 1)" "$1") && return 0
        return 1
    fi
    "$PY" - "$1" <<'PY'
import json, os, sys, urllib.request
pod, key = os.environ.get("RUNPOD_POD_ID", ""), os.environ.get("RUNPOD_API_KEY", "")
print("[boot] " + sys.argv[1] + " — снимаю под", file=sys.stderr, flush=True)
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
attempts = [("DELETE", "https://rest.runpod.io/v1/pods/" + pod, None),
            ("POST", "https://api.runpod.io/graphql", {"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}),
            ("POST", "https://rest.runpod.io/v1/pods/" + pod + "/stop", None),
            ("POST", "https://api.runpod.io/graphql", {"query": 'mutation { podStop(input: {podId: "%s"}) { id } }' % pod})]
ok = False
if pod and key:
    for method, url, body in attempts:
        try:
            req = urllib.request.Request(url, method=method, headers=H, data=json.dumps(body).encode() if body else None)
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")[:200]
                if body and '"errors"' in text:
                    raise RuntimeError(text)
                print("[boot] " + method + " " + url[-24:] + " -> " + str(r.status) + " " + text, file=sys.stderr, flush=True)
            ok = True
            break
        except Exception as e:
            print("[boot] " + method + " " + url[-24:] + " не удался: " + str(e), file=sys.stderr, flush=True)
sys.exit(0 if ok else 1)
PY
}
( sleep "${POD_LOAD_TIMEOUT_SEC:-1800}"; self_terminate "загрузка не уложилась в ${POD_LOAD_TIMEOUT_SEC:-1800} с"; ) &
DEADLINE_PID=$!

# A boot that fails for good must not simply exit: RunPod restarts the
# container into the same failure and bills for every lap, and the restarts
# bury the reason under fresh output. Try to end the pod; if the pod cannot
# (the injected key is refused — measured), stop the heartbeat and wait in
# silence. The studio reads a log that has stopped growing within three
# minutes, prints its tail as the reason, and destroys the machine.
MODEL_PID=0
die() {
    log "ПРОВАЛ: $*"
    kill "$MODEL_PID" 2>/dev/null
    kill "$HEARTBEAT_PID" 2>/dev/null; wait "$HEARTBEAT_PID" 2>/dev/null
    kill "$DEADLINE_PID" 2>/dev/null; wait "$DEADLINE_PID" 2>/dev/null
    if self_terminate "$*"; then
        sleep 30
    fi
    log "под не смог снять себя — молчу и жду, пока студия снимет его по логу"
    while :; do sleep 3600; done
}

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

# ── 1b. The interpreter that actually has the model stack in it. ──
#
# Measured on this image: the environment is the system /usr/bin/python3 and
# there is no venv — but the lookup stays, cheap and honest, in case a later
# digest moves it. Whichever interpreter imports sglang is the one everything
# after this point uses.
for cand in $(ls -d /sgl-workspace/*/.venv/bin/python /sgl-workspace/.venv/bin/python \
                   /opt/*/.venv/bin/python /opt/venv/bin/python /root/.venv/bin/python \
                   /workspace/.venv/bin/python /usr/local/bin/python3 /usr/bin/python3 2>/dev/null); do
    [ -x "$cand" ] || continue
    if "$cand" -c "import sglang" 2>/dev/null; then
        PY="$cand"
        break
    fi
done
PIP="$PY -m pip install -q --break-system-packages --retries 10 --timeout 120"
log "питон для сервера и обёртки: $PY · torch $("$PY" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, "доступна" if torch.cuda.is_available() else "НЕДОСТУПНА")' 2>/dev/null || echo 'нет')"

# ── 2. Weights, in the background: the single biggest item starts first. ──
#
# Install as little as possible. This image is a working SGLang environment with
# torch, transformers, numpy and huggingface_hub already pinned against each
# other; a blanket install (and --ignore-installed on top of it) pulled newer
# numpy and huggingface_hub over them, and eight packages then declared the
# result incompatible. So: ask Python what is actually missing and install only
# that. --ignore-installed is a fallback for a package apt owns (flask needs
# blinker, which pip may refuse to uninstall), never for the numeric stack.
missing() {
    local need=""
    for pair in $1; do
        pkg="${pair%%:*}"; mod="${pair##*:}"
        "$PY" -c "import $mod" 2>/dev/null || need="$need $pkg"
    done
    echo "$need"
}
install_deps() {
    local need
    need=$(missing "flask:flask requests:requests soundfile:soundfile numpy:numpy huggingface_hub:huggingface_hub")
    if [ -z "$need" ]; then log "зависимости обёртки уже в образе"; return 0; fi
    log "ставлю недостающее:$need"
    $PIP $need && return 0
    log "pip споткнулся о системном пакете — повторяю с --ignore-installed для:$need"
    $PIP --ignore-installed $need
}
retry install_deps || die "зависимости обёртки не встали"
# Several connections instead of one when pulling 8GB. Nice to have, never fatal.
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

# ── 3. The server itself. ──
#
# Measured on this image: it carries the dependencies but NOT the console
# script — the published recipe installs the repo inside the container. The
# source it was built against is at /sgl-workspace/sglang-omni; an editable
# --no-deps install of that is all it takes. A fresh clone of main is NOT a
# substitute: it wants sglang internals this image's sglang does not have, and
# the plugin then fails to load with a line that reads like a warning.
higgs_ok() { "$PY" -c "import sglang_omni.models.higgs_tts" 2>/dev/null; }
set_cmd() {
    if command -v sgl-omni >/dev/null 2>&1; then SGL_CMD="sgl-omni"
    elif "$PY" -c "import sglang_omni.cli" 2>/dev/null; then SGL_CMD="$PY -m sglang_omni.cli"
    else SGL_CMD=""; fi
}
SGL_CMD=""
set_cmd
if [ -n "$SGL_CMD" ] && higgs_ok; then
    log "sgl-omni уже в образе и плагин Higgs на месте"
else
    SRC=""
    for d in /sgl-workspace/sglang-omni /opt/sglang-omni /workspace/sglang-omni /root/sglang-omni; do
        [ -f "$d/pyproject.toml" ] && { SRC="$d"; break; }
    done
    if [ -z "$SRC" ]; then
        log "исходников sglang-omni в образе нет — клонирую"
        retry git clone --depth 1 https://github.com/sgl-project/sglang-omni /opt/sglang-omni \
            || die "не удалось склонировать sglang-omni"
        SRC=/opt/sglang-omni
    fi
    log "ставлю sglang-omni из $SRC (без зависимостей)"
    $PIP --no-deps -e "$SRC" || log "установка без зависимостей не прошла"
    set_cmd
    if [ -z "$SGL_CMD" ] || ! higgs_ok; then
        log "плагин не поднялся — ставлю с зависимостями"
        retry $PIP -e "$SRC" || die "не удалось поставить sglang-omni"
    fi
    set_cmd
    [ -n "$SGL_CMD" ] && higgs_ok || { log "плагин Higgs не импортируется:"; "$PY" -c "import sglang_omni.models.higgs_tts" 2>&1 | tail -5; die "плагин Higgs не импортируется"; }
fi

# ── 3b. Whatever the server still asks for, by name. ──
#
# This image is nearly complete and missing the odd package (msgpack, on the
# pod that taught this). The server names the module in its ModuleNotFoundError,
# so install that one module and ask again — never a resolving reinstall. A
# package that is present but blows up while initialising (an optional
# accelerator wanting a toolkit) is uninstalled instead; Higgs is not MoE.
pip_name() {
    case "$1" in
        zmq) echo pyzmq ;; PIL) echo pillow ;; cv2) echo opencv-python-headless ;;
        yaml) echo pyyaml ;; sklearn) echo scikit-learn ;; attr) echo attrs ;;
        google) echo protobuf ;; *) echo "$1" ;;
    esac
}
uninstall_name() {
    case "$1" in
        deep_ep) echo sgl-deep-ep ;; deep_gemm) echo sgl-deep-gemm ;; *) echo "$1" ;;
    esac
}
# The probe imports what the server will import — the model runner, not just
# the plugin — so a failure shows up here, in seconds, and not as a dead
# server after the weights are loaded.
PROBE="import sglang.srt.model_executor.model_runner, sglang_omni.models.higgs_tts, sglang_omni.cli"
probe_imports() { "$PY" -c "$PROBE" 2>&1; }
for _ in 1 2 3 4 5 6 7 8; do
    out=$(probe_imports) && break
    miss=$(printf '%s\n' "$out" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
    if [ -n "$miss" ]; then
        pkg=$(pip_name "${miss%%.*}")
        log "серверу не хватает модуля $miss — ставлю $pkg"
        $PIP "$pkg" || $PIP --ignore-installed "$pkg" || break
        continue
    fi
    broken=$(printf '%s\n' "$out" | sed -n 's#.*[/-]packages/\(deep_ep\|deep_gemm\)/.*#\1#p' | tail -1)
    case "$broken" in
        deep_ep|deep_gemm)
            log "пакет $broken не инициализируется на этом образе — снимаю $(uninstall_name "$broken")"
            "$PY" -m pip uninstall -y -q --break-system-packages "$(uninstall_name "$broken")" >/dev/null 2>&1 || break
            ;;
        *) break ;;
    esac
done
if ! "$PY" -c "$PROBE" 2>/dev/null; then
    log "сервер не импортируется:"; probe_imports | tail -12
    die "sglang / sglang_omni не импортируется"
fi
log "сервер: $SGL_CMD · sglang $("$PY" -c 'import sglang, importlib.metadata as m
try: print(m.version("sglang"))
except Exception: print("есть, без метаданных")' 2>/dev/null) · GPU $("$PY" -c 'import torch; print("видна" if torch.cuda.is_available() else "НЕ ВИДНА")' 2>/dev/null)"
mark pip_done

log "жду веса"
wait $MODEL_PID || die "веса не скачались"
MODEL_DIR=$(cat "$MODEL_PATH_FILE" 2>/dev/null)
[ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] || die "снимок весов без config.json: '$MODEL_DIR'"
mark weights_done

# ── 4. The model on the private port, the wrapper on the public one. ──
# Served from the downloaded snapshot: by name, SGLang would re-resolve "main"
# on Hugging Face and could pick up a newer commit than the one pinned.
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
        # It named what it was missing on the way out; install that and retry
        # once, rather than throw away a pod that has already paid for its
        # image and its weights.
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
