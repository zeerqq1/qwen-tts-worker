#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless worker for Qwen3-TTS voice cloning.

Design goals, in order:

  1. Nothing persists. No network volume, no S3, no bucket to clean up. The
     model lives inside the image, the voice reference travels in the request,
     the audio travels back in the response, and the container's filesystem
     dies with the job. There is no state to accumulate and nothing to bill for
     between runs.

  2. Warm workers stay warm. The clone prompt (the expensive part of ICL
     cloning — encoding the reference audio) is cached in memory keyed by the
     reference's hash, so a burst of jobs hitting the same worker pays for it
     once.

  3. Linux means torch.compile actually works, and inductor's kernel fusion is
     free speed on every decoding step. CUDA graphs are deliberately NOT used:
     Qwen3-TTS carries hidden_states between steps, which a graph would
     overwrite. If compilation misbehaves anyway the worker drops back to eager
     mid-job rather than failing the request.

Request:
    {"input": {
        "texts":        ["абзац", "..."],     # required
        "language":     "Russian",
        "ref_audio_b64": "<base64 wav>",      # required on first use of a voice
        "ref_text":      "точная расшифровка референса",
        "ref_id":        "voice_ru",          # cache key for the prompt
        "sampling":     {"temperature": 0.8, "top_p": 1.0,
                         "top_k": 50, "repetition_penalty": 1.05},
        "head_sec": 0.12, "tail_sec": 0.15, "fade_sec": 0.008
    }}

Response:
    {"sr": 24000,
     "clips": ["<base64 flac>", ...],     # same order as texts
     "gen_sec": 12.3, "audio_sec": 45.6, "compiled": true}
"""

import base64
import hashlib
import io
import os
import time
import threading
import traceback

import numpy as np
import soundfile as sf
import torch
import runpod

MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "/models/qwen3-tts-1.7b-base")
# Off by default. torch.compile cost two build cycles here — CUDA graphs are
# outright incompatible with how this model carries state, and plain inductor
# compilation is worth far less than a pipeline that reliably works. Flip the
# endpoint's QWEN_COMPILE env var to "1" to measure it; that is a config change
# on RunPod, not an image rebuild, so it costs a redeploy rather than 15 minutes.
USE_COMPILE = os.environ.get("QWEN_COMPILE", "0") == "1"

CODEC_HZ = 12
CHARS_PER_SECOND_FLOOR = 7.0
PROMPT_TOKENS_ESTIMATE = 450
KV_BYTES_PER_TOKEN = 28 * 8 * 128 * 2 * 2      # 28 layers, 8 kv heads, 128 dim, K+V, bf16
WORKING_SET_FACTOR = 1.25
VRAM_HEADROOM_GB = 1.0

DEFAULT_SAMPLING = {
    "temperature": 0.8, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05,
}

_model = None
_model_lock = threading.Lock()
_compiled = False
_eager_forward = None          # kept so a bad compile can be undone at runtime
_prompt_cache: dict[str, object] = {}
_prompt_lock = threading.Lock()


# ─────────────────────── Model ───────────────────────
def _disable_compile():
    """Undo compilation for the rest of this worker's life.

    A compile failure must never cost a job: better to finish the work a little
    slower in eager mode than to return an error the caller has to retry.
    """
    global _compiled
    if not _compiled or _eager_forward is None or _model is None:
        return False
    try:
        _model.model.talker.forward = _eager_forward
        _compiled = False
        torch.cuda.empty_cache()
        print("[worker] compile disabled, falling back to eager", flush=True)
        return True
    except Exception as e:
        print(f"[worker] could not disable compile: {e}", flush=True)
        return False


def _load_model():
    global _model, _compiled
    with _model_lock:
        if _model is not None:
            return _model
        from qwen_tts import Qwen3TTSModel

        t0 = time.time()
        kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16)
        model = None
        for impl in ("flash_attention_2", "sdpa"):
            try:
                model = Qwen3TTSModel.from_pretrained(MODEL_DIR, attn_implementation=impl, **kwargs)
                print(f"[worker] attention: {impl}", flush=True)
                break
            except Exception as e:
                print(f"[worker] {impl} unavailable ({type(e).__name__})", flush=True)
        if model is None:
            model = Qwen3TTSModel.from_pretrained(MODEL_DIR, **kwargs)
        print(f"[worker] model loaded in {time.time() - t0:.1f}s", flush=True)

        if USE_COMPILE:
            # The talker is a standard HF module with its own generate(); its
            # per-step forward is the hot path worth compiling.
            #
            # NOT mode="reduce-overhead": that turns on CUDA graphs, which
            # require the model to stop referencing tensors from an earlier
            # run. Qwen3-TTS carries hidden_states forward between decoding
            # steps (past_hidden=hidden_states[:, -1:, :]), so the graph
            # overwrites a tensor the model still reads and generation dies
            # with "accessing tensor output of CUDAGraphs that has been
            # overwritten". Plain inductor compilation keeps the kernel fusion
            # without that constraint.
            try:
                talker = model.model.talker
                global _eager_forward
                _eager_forward = talker.forward
                talker.forward = torch.compile(
                    talker.forward, fullgraph=False, dynamic=True)
                _compiled = True
                print("[worker] torch.compile enabled on talker.forward", flush=True)
            except Exception as e:
                print(f"[worker] compile unavailable ({e}) — running eager", flush=True)

        _model = model
        return _model


# ─────────────────────── Voice prompt (cached per worker) ───────────────────────
def _get_prompt(model, ref_id: str, ref_audio_b64: str, ref_text: str):
    if not ref_text or not ref_text.strip():
        raise ValueError("ref_text обязателен: без него ICL-клонирование заметно хуже")

    digest = hashlib.sha256(
        (ref_id or "").encode() + (ref_text or "").encode()
        + hashlib.sha256(base64.b64decode(ref_audio_b64)).digest()
    ).hexdigest()

    with _prompt_lock:
        cached = _prompt_cache.get(digest)
    if cached is not None:
        return cached

    raw = base64.b64decode(ref_audio_b64)
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    prompt = model.create_voice_clone_prompt(
        ref_audio=(wav, sr), ref_text=ref_text.strip(), x_vector_only_mode=False)

    with _prompt_lock:
        # One voice per worker is the normal case; the cap only stops a long
        # lived worker from growing without bound.
        if len(_prompt_cache) > 8:
            _prompt_cache.clear()
        _prompt_cache[digest] = prompt
    return prompt


# ─────────────────────── Audio shaping (mirrors the local studio) ───────────────────────
def _normalize_edges(wav, sr, head_sec, tail_sec, fade_sec):
    if wav.size == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav
    thresh = max(0.004, peak * 0.02)
    loud = np.flatnonzero(np.abs(wav) > thresh)
    if loud.size == 0:
        return wav
    speech = wav[int(loud[0]):int(loud[-1]) + 1]
    out = np.concatenate([
        np.zeros(max(0, int(sr * head_sec)), dtype="float32"),
        speech,
        np.zeros(max(0, int(sr * tail_sec)), dtype="float32"),
    ]).astype("float32")
    n = int(sr * fade_sec)
    if n > 1 and out.size > 2 * n:
        ramp = np.linspace(0.0, 1.0, n, dtype="float32")
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def _encode_flac(wav, sr) -> str:
    """FLAC keeps the transfer lossless at roughly half the bytes of raw PCM —
    the point of sending WAV in the first place is sample-exact clips, so a
    lossy transport would undo it."""
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(wav), sr, format="FLAC", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────────── Batch sizing ───────────────────────
def _token_budget(texts):
    longest = max((len(t) for t in texts), default=0)
    seconds = longest / CHARS_PER_SECOND_FLOOR
    return int(min(8192, max(256, seconds * CODEC_HZ * 1.6 + 128)))


def _safe_batch(texts, requested):
    try:
        free = torch.cuda.mem_get_info(0)[0]
        cached = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        free_gb = (free + max(0, cached)) / 1e9
    except Exception:
        return requested
    per_seq_gb = (_token_budget(texts) + PROMPT_TOKENS_ESTIMATE) \
        * KV_BYTES_PER_TOKEN * WORKING_SET_FACTOR / 1e9
    budget = max(0.0, free_gb - VRAM_HEADROOM_GB)
    return max(1, min(requested, int(budget / per_seq_gb) if per_seq_gb > 0 else requested))


# ─────────────────────── Handler ───────────────────────
def _describe(e: BaseException) -> str:
    """A message that is actually usable from the client side.

    str(e) is empty for a surprising number of torch/dynamo exceptions, which
    turns a failed job into 'генерация: ' and leaves nothing to act on. The
    type name and the last frames of the traceback always say something.
    """
    text = (str(e) or "").strip()
    tb = traceback.format_exc()
    print(tb, flush=True)                      # full trace lands in RunPod logs
    tail = " | ".join(l.strip() for l in tb.strip().splitlines()[-4:])
    return f"{type(e).__name__}: {text or '(без сообщения)'} | {tail}"[:900]


def handler(job):
    inp = job.get("input") or {}
    texts = inp.get("texts") or []
    if not texts:
        return {"error": "texts пуст"}

    try:
        model = _load_model()
        prompt = _get_prompt(model, inp.get("ref_id", ""),
                             inp.get("ref_audio_b64", ""), inp.get("ref_text", ""))
    except Exception as e:
        return {"error": f"подготовка голоса: {_describe(e)}"}

    language = inp.get("language") or "Auto"
    sampling = {**DEFAULT_SAMPLING, **(inp.get("sampling") or {})}
    head = float(inp.get("head_sec", 0.12))
    tail = float(inp.get("tail_sec", 0.15))
    fade = float(inp.get("fade_sec", 0.008))

    kw = dict(sampling)
    kw["max_new_tokens"] = _token_budget(texts)

    wavs, sr = [], 24000
    t0 = time.time()
    i, step = 0, _safe_batch(texts, len(texts))
    while i < len(texts):
        part = texts[i:i + step]
        try:
            w, sr = model.generate_voice_clone(
                text=part, language=[language] * len(part),
                voice_clone_prompt=prompt, **kw)
            wavs.extend(w)
            i += step
        except Exception as e:
            # str(e) is empty for a lot of torch/dynamo exceptions, so the
            # traceback and the type name have to be part of what we match on.
            trace = traceback.format_exc()
            msg = f"{e} {type(e).__name__} {trace}".lower()

            if any(k in msg for k in ("cudagraph", "torch_dynamo", "dynamo",
                                      "recompile", "inductor", "triton")):
                if _disable_compile():
                    continue          # same slice again, now in eager mode
                return {"error": f"генерация: {_describe(e)}"}

            if "out of memory" not in msg or step == 1:
                return {"error": f"генерация: {_describe(e)}"}

            step = max(1, step // 2)
            torch.cuda.empty_cache()
            print(f"[worker] OOM, batch -> {step}", flush=True)
    gen_sec = time.time() - t0

    clips, audio_sec = [], 0.0
    for w in wavs:
        w = np.asarray(w, dtype="float32").reshape(-1)
        w = _normalize_edges(w, sr, head, tail, fade)
        audio_sec += len(w) / sr
        clips.append(_encode_flac(w, sr))

    return {
        "sr": sr,
        "clips": clips,
        "gen_sec": round(gen_sec, 2),
        "audio_sec": round(audio_sec, 2),
        "realtime_x": round(audio_sec / max(gen_sec, 0.01), 2),
        "compiled": _compiled,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


runpod.serverless.start({"handler": handler})
