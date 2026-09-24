#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP entry point for a rented Pod running Higgs TTS 3 (bosonai/higgs-tts-3-4b).

The model is served by SGLang-Omni, which speaks its own OpenAI-shaped API on a
private port. This process sits in front of it on the pod's public port and
speaks *exactly* the contract pod_server.py speaks:

    GET  /health     {"model_loaded": bool, "loading": bool, "error": str, ...}
    GET  /boot.log   the boot log, so a pod that never came up can explain itself
    POST /generate   {"texts": [...], "ref_audio_b64": ..., "ref_text": ...,
                      "sampling": {...}, "head_sec": .., "tail_sec": .., "fade_sec": ..}
                  -> {"clips": [base64 FLAC], "sr": 24000, "gen_sec": .., ...}
    POST /shutdown   end the pod

That is the whole point of this file. Because the contract matches, the studio
drives a Higgs fleet with the same pod_manager, the same pod_client, the same
watchdogs, the same write-off clocks and the same ledger as a Qwen fleet — the
only difference between the two is which image the machine boots.

Audio shaping is imported from engine.py, so a clip from a Higgs pod gets the
identical edge treatment as one from a Qwen pod: a batch split between engines
must not have two kinds of seam.
"""

import base64
import io
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf
from flask import Flask, request, jsonify, Response

import engine                       # normalize_edges / encode_flac / describe
import pod_lifecycle

PORT = int(os.environ.get("POD_PORT", "8000"))
UPSTREAM = f"http://127.0.0.1:{os.environ.get('HIGGS_UPSTREAM_PORT', '8001')}"
TOKEN = os.environ.get("POD_TOKEN", "")
MODEL_ID = os.environ.get("HIGGS_MODEL_ID", "bosonai/higgs-tts-3-4b")
BOOT_LOG = os.environ.get("POD_BOOT_LOG", "/workspace/boot.log")
REFS_DIR = os.environ.get("HIGGS_REFS_DIR", "/workspace/refs")

# Same clocks, same names and same meanings as pod_server.py — the studio sets
# them per pod and must not have to remember which engine it is talking to.
IDLE_EXIT_SEC = float(os.environ.get("POD_IDLE_EXIT_SEC", "600"))
NO_WORK_EXIT_SEC = float(os.environ.get("POD_NO_WORK_EXIT_SEC", "1800"))
MAX_LIFE_SEC = float(os.environ.get("POD_MAX_LIFE_SEC", "14400"))
LOAD_TIMEOUT_SEC = float(os.environ.get("POD_LOAD_TIMEOUT_SEC", "1800"))
BUSY_MAX_SEC = float(os.environ.get("POD_BUSY_MAX_SEC", "1800"))

# SGLang-Omni batches continuously, so the throughput comes from keeping several
# requests in flight rather than from one big call. Eight is comfortable for a
# 4B model on one card and stays far from any queue limit.
CONCURRENCY = int(os.environ.get("HIGGS_CONCURRENCY", "8"))
UPSTREAM_TIMEOUT = (15, 600)
ATTEMPTS = 3

# Higgs' codec runs at 25 frames per second (40 ms per frame). The budget is the
# same idea as the Qwen engine's: a clone that loses its way stops early instead
# of babbling until the context limit.
#
# That limit is low and fixed: the Higgs stage is built with context_length 4096
# and refuses an override, and the reference prompt already occupies several
# hundred tokens of it. 3000 leaves room for the prompt while still allowing a
# paragraph twice as long as anything the studio's chunker produces.
FRAMES_PER_SEC = 25
CHARS_PER_SECOND_FLOOR = 7.0
# SGLang's Higgs stage clamps max_new_tokens to 2048 frames (~82 s of speech)
# and, when a clip hits it, answers 200 with the audio cut mid-sentence. So the
# budget never asks for more, and inputs are kept to what 2048 frames carry
# with margin: ~900 characters is ~60 s at narration pace. The studio's default
# chunk is 400.
MAX_NEW_TOKENS_CAP = 2048
MAX_INPUT_CHARS = 900

app = Flask(__name__)
_started = time.time()
_model_ready_at = 0.0
_last_seen = time.time()
_last_job = time.time()
_load_error = ""
_loading = True
# Requests in flight, each with the moment it started working. The watchdog
# measures the OLDEST one: with two jobs queued per pod the count never touches
# zero for the whole batch, and a clock that only reset at zero declared a
# thirty-minute chapter "one wedged generation" and deleted the machine.
_inflight: dict = {}
_req_seq = 0
_lock = threading.Lock()
_ref_lock = threading.Lock()
_ref_cache: dict = {}


def _touch():
    global _last_seen
    with _lock:
        _last_seen = time.time()


def _touch_job():
    global _last_job
    with _lock:
        _last_job = time.time()


def _authorised() -> bool:
    if not TOKEN:
        return True
    return request.headers.get("X-Studio-Token", "") == TOKEN


def _gpu_name() -> str:
    """Read the card's name without importing CUDA into this process: the model
    needs every megabyte of the card, and a context here would take some."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


# ─────────────────────── Upstream readiness ───────────────────────
def _upstream_up() -> bool:
    """Is SGLang serving?

    It binds its port only after the model is in, so while it loads there is no
    HTTP at all — a refused connection here means "still coming", not "broken".
    /health answers 200 with running=true once the pipeline is up; /v1/models is
    only a config echo, kept as a fallback for a future version that changes the
    health body.
    """
    try:
        r = requests.get(f"{UPSTREAM}/health", timeout=8)
        if r.status_code == 200:
            try:
                return bool((r.json() or {}).get("running", True))
            except Exception:
                return True
        return False
    except Exception:
        try:
            return requests.get(f"{UPSTREAM}/v1/models", timeout=8).status_code == 200
        except Exception:
            return False


def _wait_for_upstream():
    global _loading, _model_ready_at, _load_error
    t0 = time.time()
    while time.time() - t0 < LOAD_TIMEOUT_SEC:
        if _upstream_up():
            _model_ready_at = time.time()
            _loading = False
            print(f"[higgs] SGLang готов за {_model_ready_at - t0:.0f}s", flush=True)
            return
        time.sleep(5)
    _load_error = f"SGLang-Omni не поднялся за {LOAD_TIMEOUT_SEC:.0f} с"
    _loading = False
    print(f"[higgs] {_load_error}", flush=True)


# ─────────────────────── Reference handling ───────────────────────
def _ref_path(ref_id: str, ref_audio_b64: str) -> str:
    """Write the reference to a real file once and reuse it.

    SGLang takes the reference as a path it can open, and a path is also the
    form that needs no guessing about what the server accepts inline. Keyed by
    the audio's own bytes, so a voice rebuilt under the same id lands in a new
    file instead of being served from the old one.
    """
    import hashlib
    raw = base64.b64decode(ref_audio_b64)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    with _ref_lock:
        hit = _ref_cache.get(digest)
        if hit:
            return hit
        os.makedirs(REFS_DIR, exist_ok=True)
        path = os.path.join(REFS_DIR, f"{digest}.wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(raw)
        _ref_cache[digest] = path
        return path


def _token_budget(texts: list) -> int:
    longest = max((len(t) for t in texts), default=0)
    seconds = longest / CHARS_PER_SECOND_FLOOR
    return int(min(MAX_NEW_TOKENS_CAP, max(256, seconds * FRAMES_PER_SEC * 1.6 + 128)))


def _decode_audio(resp) -> tuple:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "json" in ct:
        d = resp.json()
        b64 = (((d.get("audio") or {}).get("data")) or d.get("data")
               or (d.get("audio") if isinstance(d.get("audio"), str) else None))
        if not b64:
            raise RuntimeError(f"ответ без аудио: {str(d)[:200]}")
        raw = base64.b64decode(b64.split(",", 1)[-1])
    else:
        raw = resp.content
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, int(sr)


def _speak(text: str, ref_path: str, ref_text: str, sampling: dict, budget: int,
           language: str = "") -> tuple:
    body = {
        "input": text,
        "references": [{"audio_path": ref_path, "text": ref_text}],
        "max_new_tokens": budget,
        "response_format": "wav",
    }
    # The language tag is part of the request schema and costs nothing to send;
    # the studio already knows which language the voice speaks.
    if language and language != "Auto":
        body["language"] = language
    for k in ("temperature", "top_p", "top_k", "repetition_penalty", "seed"):
        if sampling.get(k) is not None:
            body[k] = sampling[k]
    last = ""
    for attempt in range(ATTEMPTS):
        try:
            r = requests.post(f"{UPSTREAM}/v1/audio/speech", json=body, timeout=UPSTREAM_TIMEOUT)
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
            # The server reports a clip that ran into the length ceiling only
            # in a header. Silence here would write a truncated paragraph.
            if (r.headers.get("X-Finish-Reason") or "").lower() == "length":
                raise RuntimeError(f"клип упёрся в предел {MAX_NEW_TOKENS_CAP} кадров и обрезан — "
                                   f"уменьшите «Макс. чанк» (фрагмент {len(text)} знаков)")
            return _decode_audio(r)
        except Exception as e:
            last = str(e)
            if attempt < ATTEMPTS - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(last)


def generate(payload: dict) -> dict:
    """Same shape in and out as engine.generate, so pod_client cannot tell the
    two engines apart except by what it hears."""
    texts = payload.get("texts") or []
    if not texts:
        return {"error": "texts пуст"}
    long_ones = [i + 1 for i, t in enumerate(texts) if len(t) > MAX_INPUT_CHARS]
    if long_ones:
        return {"error": f"фрагменты {long_ones[:5]} длиннее {MAX_INPUT_CHARS} знаков — "
                         f"сервер их не примет; уменьшите «Макс. чанк»"}
    ref_text = (payload.get("ref_text") or "").strip()
    if not ref_text:
        return {"error": "ref_text обязателен: без него клон заметно хуже"}
    try:
        ref_path = _ref_path(payload.get("ref_id", ""), payload.get("ref_audio_b64", ""))
    except Exception as e:
        return {"error": f"подготовка голоса: {engine.describe(e)}"}

    sampling = payload.get("sampling") or {}
    language = payload.get("language") or ""
    budget = _token_budget(texts)
    head = float(payload.get("head_sec", engine.HEAD_SILENCE_SEC))
    tail = float(payload.get("tail_sec", engine.TAIL_SILENCE_SEC))
    fade = float(payload.get("fade_sec", engine.EDGE_FADE_SEC))

    t0 = time.time()
    out: dict = {}
    err: list = []

    def _one(i: int):
        try:
            out[i] = _speak(texts[i], ref_path, ref_text, sampling, budget, language)
        except Exception as e:
            err.append(f"фрагмент {i + 1}: {engine.describe(e)}")

    with ThreadPoolExecutor(max_workers=max(1, min(CONCURRENCY, len(texts)))) as pool:
        list(pool.map(_one, range(len(texts))))
    if err:
        return {"error": "; ".join(err[:3])}
    gen_sec = time.time() - t0

    clips, audio_sec, sr = [], 0.0, 24000
    for i in range(len(texts)):
        wav, sr = out[i]
        wav = engine.normalize_edges(wav.reshape(-1), sr, head, tail, fade)
        audio_sec += len(wav) / sr
        clips.append(engine.encode_flac(wav, sr))

    return {
        "sr": sr, "clips": clips,
        "gen_sec": round(gen_sec, 2), "audio_sec": round(audio_sec, 2),
        "realtime_x": round(audio_sec / max(gen_sec, 0.01), 2),
        "engine": "higgs", "model": MODEL_ID, "gpu": _gpu_name(),
        "concurrency": CONCURRENCY,
    }


# ─────────────────────── Watchdog ───────────────────────
def _watchdog():
    """Three ways to be worth nothing, three clocks — and the hard ceilings
    apply even while a request is in flight, because a generation that never
    returns used to keep the pod alive forever."""
    while True:
        time.sleep(15)
        with _lock:
            idle = time.time() - _last_seen
            no_work = time.time() - _last_job
            busy = len(_inflight)
            busy_for = (time.time() - min(_inflight.values())) if _inflight else 0.0
        alive = time.time() - _started
        if alive > MAX_LIFE_SEC:
            pod_lifecycle.self_terminate(f"предельное время жизни {alive:.0f}s")
        if busy and busy_for > BUSY_MAX_SEC:
            pod_lifecycle.self_terminate(f"одна генерация держит GPU {busy_for:.0f}s — завис")
        if busy:
            continue
        if _loading:
            if alive > LOAD_TIMEOUT_SEC:
                pod_lifecycle.self_terminate(f"модель не поднялась за {alive:.0f}s")
            continue
        if idle > IDLE_EXIT_SEC:
            pod_lifecycle.self_terminate(f"{idle:.0f}s без запросов")
        if no_work > NO_WORK_EXIT_SEC:
            pod_lifecycle.self_terminate(f"{no_work:.0f}s без единой задачи")


# ─────────────────────── Routes ───────────────────────
@app.get("/health")
def health():
    _touch()
    with _lock:
        idle = time.time() - _last_seen
    return jsonify({
        "boot": pod_lifecycle.boot_marks(),
        "serving_since": _model_ready_at,
        "ok": True,
        "model_loaded": (not _loading) and not _load_error,
        "loading": _loading,
        "error": _load_error,
        "engine": "higgs",
        "model": MODEL_ID,
        "gpu": _gpu_name(),
        "uptime_sec": round(time.time() - _started),
        "idle_sec": round(idle),
    })


@app.get("/boot.log")
def boot_log():
    try:
        with open(BOOT_LOG, encoding="utf-8", errors="replace") as f:
            return Response(f.read(), mimetype="text/plain; charset=utf-8")
    except Exception as e:
        return Response(f"нет лога: {e}", mimetype="text/plain; charset=utf-8")


@app.post("/generate")
def api_generate():
    if not _authorised():
        return jsonify({"error": "bad token"}), 401
    _touch()
    _touch_job()
    global _req_seq
    with _lock:
        _req_seq += 1
        rid = _req_seq
        _inflight[rid] = time.time()
    try:
        out = generate(request.get_json(silent=True) or {})
    finally:
        with _lock:
            _inflight.pop(rid, None)
    _touch()
    if out.get("error"):
        return jsonify(out), 500
    return jsonify(out)


@app.post("/shutdown")
def shutdown():
    if not _authorised():
        return jsonify({"error": "bad token"}), 401
    threading.Timer(0.5, lambda: pod_lifecycle.self_terminate("команда студии")).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_watchdog, daemon=True).start()
    threading.Thread(target=_wait_for_upstream, daemon=True).start()
    # Both clocks start when this process starts serving; the upstream model may
    # still be loading, and /health says so.
    _last_job = _last_seen = time.time()
    print(f"[higgs] обёртка слушает :{PORT}, апстрим {UPSTREAM}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
