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
    "temperature": 0.7, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05,
}

# Edge shaping. These must stay identical to qwen_engine.py: a batch that is
# split between the cloud and this machine has to come back sounding the same.
HEAD_SILENCE_SEC = 0.06
TAIL_SILENCE_SEC = 0.08
EDGE_FADE_SEC = 0.012
DECAY_KEEP_SEC = 0.20
ROOM_TONE_MAX_DBFS = -38.0
ROOM_TONE_MAX_PEAK_DBFS = -34.0
ROOM_TONE_BELOW_SPEECH_DB = 44.0
ROOM_TONE_MAX_GAIN_DB = 18.0
ROOM_RAMP_SEC = 0.0

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
def _quietest_stretch(speech, sr):
    """The quietest ~100ms inside a clip — an inter-word gap, i.e. its
    background — or None when nothing in the clip is quiet enough to copy.
    Both the mean and the peak must pass: a stretch that averages -40 dBFS
    while peaking at -24 is a consonant with silence around it."""
    win = max(1, int(sr * 0.025))
    n = speech.size // win
    if n < 4:
        return None
    frames = speech[:n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    csum = np.concatenate([[0.0], np.cumsum(rms)])

    # Longest usable gap wins: a longer seed tiles with a slower period and so
    # is less likely to read as a texture. A dense three-second sentence has no
    # 100ms gap but usually has a 25ms one, and taking that beats giving up and
    # padding with a hole.
    for need in (4, 3, 2, 1):
        if need > n - 2:
            continue
        runs = (csum[need:] - csum[:-need]) / need
        if runs.size == 0:
            continue
        k = int(np.argmin(runs))
        mean_rms = float(runs[k])
        if mean_rms <= 0:
            continue
        seg = speech[k * win:(k + need) * win].copy()
        peak = float(np.max(np.abs(seg)))
        if peak <= 0:
            continue
        if 20.0 * np.log10(mean_rms) > ROOM_TONE_MAX_DBFS:
            continue
        if 20.0 * np.log10(peak) > ROOM_TONE_MAX_PEAK_DBFS:
            continue
        return seg
    return None


def _tile_tone(seg, n):
    """Fill n samples with the background segment, mirroring every other copy
    so the loop seam is continuous instead of a step."""
    if seg is None or seg.size == 0 or n <= 0:
        return None
    reps = -(-n // seg.size)
    parts = [seg if i % 2 == 0 else seg[::-1] for i in range(reps)]
    return np.concatenate(parts)[:n].astype("float32").copy()


def _speech_level(speech, sr):
    """The clip's speaking level: 75th percentile of 10ms window RMS."""
    w = max(1, int(sr * 0.010))
    n = speech.size // w
    if n < 2:
        return float(np.sqrt((speech ** 2).mean()))
    frames = np.sqrt((speech[:n * w].reshape(n, w) ** 2).mean(axis=1))
    return float(np.percentile(frames, 75))


def _match_room_level(seg, seg_rms, speech, sr):
    """Scale the background so the pause sits the same distance under the
    speech as in the source recording these voices were cloned from."""
    if seg_rms <= 0:
        return seg
    target = _speech_level(speech, sr) * (10.0 ** (-ROOM_TONE_BELOW_SPEECH_DB / 20.0))
    if target <= 0:
        return seg
    gain = min(10.0 ** (ROOM_TONE_MAX_GAIN_DB / 20.0), target / seg_rms)
    return (seg * np.float32(gain)).astype("float32")


def _shape_pad(pad, outer_ramp, inner_ramp, at_head):
    """Ramp a background pad up from silence at the file edge and back down to
    silence where it meets the speech, so neither seam is a step."""
    if pad is None or pad.size == 0:
        return pad
    o = min(outer_ramp, pad.size // 2)
    i = min(inner_ramp, pad.size - o)
    if at_head:
        if o > 1:
            pad[:o] *= np.linspace(0.0, 1.0, o, dtype="float32")
        if i > 1:
            pad[-i:] *= np.linspace(1.0, 0.0, i, dtype="float32")
    else:
        if i > 1:
            pad[:i] *= np.linspace(0.0, 1.0, i, dtype="float32")
        if o > 1:
            pad[-o:] *= np.linspace(1.0, 0.0, o, dtype="float32")
    return pad


def normalize_edges(wav, sr, head_sec=HEAD_SILENCE_SEC, tail_sec=TAIL_SILENCE_SEC,
                    fade_sec=EDGE_FADE_SEC):
    """Give every clip identical air at both ends and make both seams inaudible.

    Mirror of qwen_engine.normalize_edges — see the long note there. In short:
    the boundary walks out through the natural decay so a final consonant is
    not amputated, the fade lands on the edge of the SPEECH rather than on the
    padding it used to multiply for nothing, and the pad carries the clip's own
    background instead of digital zero, so a splice is a pause rather than a
    noise gate slamming shut.
    """
    if wav.size == 0:
        return wav
    wav = np.asarray(wav, dtype="float32")
    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav

    loud = np.flatnonzero(np.abs(wav) > max(0.004, peak * 0.02))
    if loud.size == 0:
        return wav
    start, end = int(loud[0]), int(loud[-1]) + 1

    decay_thr = max(0.0008, peak * 0.004)
    keep = max(0, int(sr * DECAY_KEEP_SEC))
    lo = max(0, start - keep)
    pre = np.flatnonzero(np.abs(wav[lo:start]) > decay_thr)
    if pre.size:
        start = lo + int(pre[0])
    hi = min(wav.size, end + keep)
    post = np.flatnonzero(np.abs(wav[end:hi]) > decay_thr)
    if post.size:
        end = end + int(post[-1]) + 1

    speech = wav[start:end].copy()

    fade_n = int(sr * fade_sec)
    if fade_n > 1 and speech.size > 2 * fade_n:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype="float32")
        speech[:fade_n] *= ramp
        speech[-fade_n:] *= ramp[::-1]

    head_n = max(0, int(sr * head_sec))
    tail_n = max(0, int(sr * tail_sec))

    seg = _quietest_stretch(speech, sr)
    if seg is not None:
        seg = _match_room_level(seg, float(np.sqrt((seg ** 2).mean())), speech, sr)
        outer = int(sr * ROOM_RAMP_SEC)
        head = _shape_pad(_tile_tone(seg, head_n), outer, fade_n, True)
        tail = _shape_pad(_tile_tone(seg, tail_n), outer, fade_n, False)
    else:
        head = tail = None
    if head is None:
        head = np.zeros(head_n, dtype="float32")
    if tail is None:
        tail = np.zeros(tail_n, dtype="float32")

    return np.concatenate([head, speech, tail]).astype("float32")


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
    # Reported back so a throughput drop can be diagnosed from the client side
    # instead of guessed at: if the batch silently shrinks between jobs, the
    # model re-reads its weights more times per second of audio and everything
    # slows down, which looks identical to the GPU being slower.
    first_step = step
    steps_used = []
    try:
        _free0 = torch.cuda.mem_get_info(0)[0] / 1e9
    except Exception:
        _free0 = -1.0
    while i < len(texts):
        part = texts[i:i + step]
        try:
            w, sr = model.generate_voice_clone(
                text=part, language=[language] * len(part),
                voice_clone_prompt=prompt, **kw)
            wavs.extend(w)
            steps_used.append(len(part))
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
        "batch_first": first_step,
        "batch_sizes": steps_used,
        "free_gb_before": round(_free0, 1),
        "free_gb_after": round(torch.cuda.mem_get_info(0)[0] / 1e9, 1)
                         if torch.cuda.is_available() else -1,
    }
