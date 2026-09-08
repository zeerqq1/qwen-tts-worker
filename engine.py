#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation core shared by both worker shapes.

handler.py serves RunPod's Serverless queue; pod_server.py serves HTTP on a
rented Pod. They differ only in how a request arrives — the model, the ICL
cloning, the batching and the audio shaping must stay identical, or the two
modes would quietly produce different audio for the same text.
"""

import base64
import hashlib
import io
import os
import threading
import time
import traceback

import numpy as np
import soundfile as sf
import torch

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", "")   # empty -> download by id
# Off by default: CUDA graphs are incompatible with how this model carries
# hidden state between decoding steps, and plain inductor compilation is worth
# far less than a worker that simply never fails.
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

HEAD_SILENCE_SEC = 0.12
TAIL_SILENCE_SEC = 0.15
EDGE_FADE_SEC = 0.008

_model = None
_model_lock = threading.Lock()
_compiled = False
_eager_forward = None
_prompt_cache: dict[str, object] = {}
_prompt_lock = threading.Lock()


def describe(e: BaseException) -> str:
    """A message that is actually usable from the client side.

    str(e) is empty for a surprising number of torch/dynamo exceptions, which
    turns a failed request into a blank error and leaves nothing to act on.
    """
    text = (str(e) or "").strip()
    tb = traceback.format_exc()
    print(tb, flush=True)
    tail = " | ".join(l.strip() for l in tb.strip().splitlines()[-4:])
    return f"{type(e).__name__}: {text or '(без сообщения)'} | {tail}"[:900]


# ─────────────────────── Model ───────────────────────
def _disable_compile():
    """Undo compilation for the rest of this worker's life, mid-job if needed."""
    global _compiled
    if not _compiled or _eager_forward is None or _model is None:
        return False
    try:
        _model.model.talker.forward = _eager_forward
        _compiled = False
        torch.cuda.empty_cache()
        print("[engine] compile disabled, falling back to eager", flush=True)
        return True
    except Exception as e:
        print(f"[engine] could not disable compile: {e}", flush=True)
        return False


def load_model():
    global _model, _compiled, _eager_forward
    with _model_lock:
        if _model is not None:
            return _model
        from qwen_tts import Qwen3TTSModel

        src = MODEL_DIR or MODEL_ID
        t0 = time.time()
        kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16)
        model = None
        for impl in ("flash_attention_2", "sdpa"):
            try:
                model = Qwen3TTSModel.from_pretrained(src, attn_implementation=impl, **kwargs)
                print(f"[engine] attention: {impl}", flush=True)
                break
            except Exception as e:
                print(f"[engine] {impl} unavailable ({type(e).__name__})", flush=True)
        if model is None:
            model = Qwen3TTSModel.from_pretrained(src, **kwargs)
        print(f"[engine] model loaded in {time.time() - t0:.1f}s", flush=True)

        if USE_COMPILE:
            try:
                talker = model.model.talker
                _eager_forward = talker.forward
                talker.forward = torch.compile(talker.forward, fullgraph=False, dynamic=True)
                _compiled = True
                print("[engine] torch.compile enabled", flush=True)
            except Exception as e:
                print(f"[engine] compile unavailable ({e})", flush=True)

        _model = model
        return _model


def is_compiled() -> bool:
    return _compiled


def gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        return ""


# ─────────────────────── Voice prompt (cached per worker) ───────────────────────
def get_prompt(model, ref_id: str, ref_audio_b64: str, ref_text: str):
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

    wav, sr = sf.read(io.BytesIO(base64.b64decode(ref_audio_b64)),
                      dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    prompt = model.create_voice_clone_prompt(
        ref_audio=(wav, sr), ref_text=ref_text.strip(), x_vector_only_mode=False)

    with _prompt_lock:
        # A pod serves every language in a batch, so the cache has to hold more
        # than one voice; the cap only stops unbounded growth.
        if len(_prompt_cache) > 16:
            _prompt_cache.clear()
        _prompt_cache[digest] = prompt
    return prompt


# ─────────────────────── Audio shaping ───────────────────────
def normalize_edges(wav, sr, head_sec=HEAD_SILENCE_SEC, tail_sec=TAIL_SILENCE_SEC,
                    fade_sec=EDGE_FADE_SEC):
    """Give every clip identical air at both ends, then fade the edges so a cut
    never lands mid-waveform (which clicks on every splice in a montage)."""
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


def encode_flac(wav, sr) -> str:
    """Lossless and about half the bytes of raw PCM — the point of sending WAV
    at all is sample-exact clips, so a lossy transport would undo it."""
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(wav), sr, format="FLAC", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────────── Batch sizing ───────────────────────
def token_budget(texts):
    longest = max((len(t) for t in texts), default=0)
    return int(min(8192, max(256, (longest / CHARS_PER_SECOND_FLOOR) * CODEC_HZ * 1.6 + 128)))


def safe_batch(texts, requested):
    try:
        free = torch.cuda.mem_get_info(0)[0]
        cached = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        free_gb = (free + max(0, cached)) / 1e9
    except Exception:
        return requested
    per_seq_gb = (token_budget(texts) + PROMPT_TOKENS_ESTIMATE) \
        * KV_BYTES_PER_TOKEN * WORKING_SET_FACTOR / 1e9
    budget = max(0.0, free_gb - VRAM_HEADROOM_GB)
    return max(1, min(requested, int(budget / per_seq_gb) if per_seq_gb > 0 else requested))


# ─────────────────────── The one entry point both shapes call ───────────────────────
def generate(payload: dict) -> dict:
    """Turn a request dict into base64 FLAC clips. Never raises: errors come
    back in the returned dict so the caller always gets a usable message."""
    texts = payload.get("texts") or []
    if not texts:
        return {"error": "texts пуст"}

    try:
        model = load_model()
        prompt = get_prompt(model, payload.get("ref_id", ""),
                            payload.get("ref_audio_b64", ""), payload.get("ref_text", ""))
    except Exception as e:
        return {"error": f"подготовка голоса: {describe(e)}"}

    language = payload.get("language") or "Auto"
    kw = {**DEFAULT_SAMPLING, **(payload.get("sampling") or {})}
    kw["max_new_tokens"] = token_budget(texts)
    head = float(payload.get("head_sec", HEAD_SILENCE_SEC))
    tail = float(payload.get("tail_sec", TAIL_SILENCE_SEC))
    fade = float(payload.get("fade_sec", EDGE_FADE_SEC))

    wavs, sr = [], 24000
    t0 = time.time()
    i, step = 0, safe_batch(texts, len(texts))
    while i < len(texts):
        part = texts[i:i + step]
        try:
            w, sr = model.generate_voice_clone(
                text=part, language=[language] * len(part),
                voice_clone_prompt=prompt, **kw)
            wavs.extend(w)
            i += step
        except Exception as e:
            msg = f"{e} {type(e).__name__} {traceback.format_exc()}".lower()
            if any(k in msg for k in ("cudagraph", "torch_dynamo", "dynamo",
                                      "recompile", "inductor", "triton")):
                if _disable_compile():
                    continue
                return {"error": f"генерация: {describe(e)}"}
            if "out of memory" not in msg or step == 1:
                return {"error": f"генерация: {describe(e)}"}
            step = max(1, step // 2)
            torch.cuda.empty_cache()
            print(f"[engine] OOM, batch -> {step}", flush=True)
    gen_sec = time.time() - t0

    clips, audio_sec = [], 0.0
    for w in wavs:
        w = normalize_edges(np.asarray(w, dtype="float32").reshape(-1), sr, head, tail, fade)
        audio_sec += len(w) / sr
        clips.append(encode_flac(w, sr))

    return {
        "sr": sr, "clips": clips,
        "gen_sec": round(gen_sec, 2), "audio_sec": round(audio_sec, 2),
        "realtime_x": round(audio_sec / max(gen_sec, 0.01), 2),
        "compiled": _compiled, "gpu": gpu_name(),
    }
