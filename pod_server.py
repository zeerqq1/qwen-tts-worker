#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP entry point for a rented Pod.

A Pod is billed for uptime rather than per job, which is roughly 2.3x cheaper
per GPU-hour than Serverless — worth it when work arrives in predictable
batches. The studio drives several of these in parallel and destroys them when
the batch is done.

Same engine.py as the Serverless handler, so the audio is identical.

Safety: this process exits on its own if the studio stops talking to it
(IDLE_EXIT_SEC) or if it has simply been alive too long (MAX_LIFE_SEC). A pod
whose process exits stops billing, so an abandoned pod cannot quietly run up a
bill even if the studio crashed, lost its network, or was killed outright.
"""

import os
import threading
import time

from flask import Flask, request, jsonify

import engine

PORT = int(os.environ.get("POD_PORT", "8000"))
TOKEN = os.environ.get("POD_TOKEN", "")
# The studio pings every few seconds while a batch runs; ten minutes of silence
# means nobody is coming back for this pod.
IDLE_EXIT_SEC = float(os.environ.get("POD_IDLE_EXIT_SEC", "600"))
# A hard ceiling regardless of traffic, so a wedged client cannot keep a pod
# alive forever.
MAX_LIFE_SEC = float(os.environ.get("POD_MAX_LIFE_SEC", "14400"))   # 4 hours

app = Flask(__name__)
_started = time.time()
_last_seen = time.time()
_lock = threading.Lock()
# The model is not thread-safe: two generations running at once corrupt each
# other's tensors ("size of tensor a (16) must match ..."). Flask serves
# requests concurrently and the studio can hand a pod a second job while the
# first is still running, so generation has to be serialised here. Queuing is
# the right behaviour anyway — the GPU can only do one batch at a time, and a
# queued request costs nothing next to a failed one.
_gpu_lock = threading.Lock()


def _touch():
    global _last_seen
    with _lock:
        _last_seen = time.time()


def _authorised() -> bool:
    if not TOKEN:
        return True
    return request.headers.get("X-Studio-Token", "") == TOKEN


def _watchdog():
    """Self-destruct so an abandoned pod cannot keep billing."""
    while True:
        time.sleep(15)
        with _lock:
            idle = time.time() - _last_seen
        alive = time.time() - _started
        if idle > IDLE_EXIT_SEC:
            print(f"[pod] {idle:.0f}s без запросов — выхожу", flush=True)
            os._exit(0)
        if alive > MAX_LIFE_SEC:
            print(f"[pod] предельное время жизни {alive:.0f}s — выхожу", flush=True)
            os._exit(0)


@app.get("/health")
def health():
    _touch()
    with _lock:
        idle = time.time() - _last_seen
    return jsonify({
        "ok": True,
        "model_loaded": engine._model is not None,
        "gpu": engine.gpu_name(),
        "compiled": engine.is_compiled(),
        "uptime_sec": round(time.time() - _started),
        "idle_sec": round(idle),
    })


@app.post("/generate")
def generate():
    if not _authorised():
        return jsonify({"error": "bad token"}), 401
    _touch()
    with _gpu_lock:
        _touch()  # waiting for the lock still counts as being talked to
        out = engine.generate(request.get_json(silent=True) or {})
    _touch()      # generation can take minutes; do not let the watchdog fire
    if out.get("error"):
        return jsonify(out), 500
    return jsonify(out)


@app.post("/shutdown")
def shutdown():
    """Let the studio end billing the moment the batch is finished, rather than
    waiting for the idle timer."""
    if not _authorised():
        return jsonify({"error": "bad token"}), 401
    print("[pod] выключение по команде студии", flush=True)
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_watchdog, daemon=True).start()

    # Load before serving: the studio waits on /health reporting model_loaded,
    # and a pod that accepts work before it is ready would just stall a batch.
    print("[pod] загружаю модель...", flush=True)
    try:
        engine.load_model()
        print(f"[pod] готов на {engine.gpu_name()}", flush=True)
    except Exception as e:
        print(f"[pod] модель не загрузилась: {engine.describe(e)}", flush=True)

    app.run(host="0.0.0.0", port=PORT, threaded=True)
