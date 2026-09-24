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

from flask import Flask, request, jsonify, Response

import engine
import pod_lifecycle

PORT = int(os.environ.get("POD_PORT", "8000"))
TOKEN = os.environ.get("POD_TOKEN", "")
# The studio pings every few seconds while a batch runs; ten minutes of silence
# means nobody is coming back for this pod.
IDLE_EXIT_SEC = float(os.environ.get("POD_IDLE_EXIT_SEC", "600"))
# Being talked to is not the same as being used. The studio polls /health every
# few seconds, and that poll used to reset the idle timer — so a pod whose model
# never loaded could not self-destruct while the studio was watching it die.
# This is the ceiling on contact without work.
NO_WORK_EXIT_SEC = float(os.environ.get("POD_NO_WORK_EXIT_SEC", "1800"))
# A hard ceiling regardless of traffic, so a wedged client cannot keep a pod
# alive forever.
MAX_LIFE_SEC = float(os.environ.get("POD_MAX_LIFE_SEC", "14400"))   # 4 hours
# Loading is not idling. The watchdog starts before the model does, and nothing
# can talk to this pod until the model is in — so the idle timer had to be told
# to wait, or a slow load would kill the pod it was loading.
LOAD_TIMEOUT_SEC = float(os.environ.get("POD_LOAD_TIMEOUT_SEC", "1800"))
# What pod_boot.sh printed on the way here. The exec at the end of that script
# keeps this process's stdout pointed at the same tee, so the file goes on
# growing with everything printed below.
BOOT_LOG = os.environ.get("POD_BOOT_LOG", "/workspace/boot.log")
# One /generate may hold the GPU this long before the pod gives up on itself.
# The studio stops waiting for an answer after 900 s; a request still running
# twice that long is wedged, and «busy» must not exempt it from every clock.
BUSY_MAX_SEC = float(os.environ.get("POD_BUSY_MAX_SEC", "1800"))


# Ending this pod's billing lives in pod_lifecycle, shared with higgs_server:
# one copy, because "the process exited" and "the machine stopped costing
# money" are not the same thing and the difference is worth a weekend's rent.
_self_terminate = pod_lifecycle.self_terminate

app = Flask(__name__)
_started = time.time()
_model_ready_at = 0.0
_last_seen = time.time()
_last_job = time.time()
# Why this pod will never serve, if that is where it ended up. A failed load
# used to go to stdout and nowhere else: the studio saw a pod that answered
# /health but never reported model_loaded, could not tell that apart from a
# machine that had not booted at all, and said so in the log — wrongly.
_load_error = ""
_loading = True
# Generations in flight. A batch of 5400 characters can hold the GPU for
# minutes, and /generate only touches the clock on the way in and on the way
# out — so a long job looked exactly like an abandoned pod and could be killed
# halfway through its own work.
# Requests in flight, each with the moment it started working. The watchdog
# measures the OLDEST one: with two jobs queued per pod the count never touches
# zero for the whole batch, and a clock that only reset at zero declared a
# thirty-minute chapter "one wedged generation" and deleted the machine.
_inflight: dict = {}
_req_seq = 0
_lock = threading.Lock()
# The model is not thread-safe: two generations running at once corrupt each
# other's tensors ("size of tensor a (16) must match ..."). Flask serves
# requests concurrently and the studio can hand a pod a second job while the
# first is still running, so generation has to be serialised here. Queuing is
# the right behaviour anyway — the GPU can only do one batch at a time, and a
# queued request costs nothing next to a failed one.
_gpu_lock = threading.Lock()


def _touch():
    """Someone is still out there. Says nothing about whether they need us."""
    global _last_seen
    with _lock:
        _last_seen = time.time()


def _touch_job():
    """Actual work arrived — the only thing that proves this pod is earning."""
    global _last_job
    with _lock:
        _last_job = time.time()


def _authorised() -> bool:
    if not TOKEN:
        return True
    return request.headers.get("X-Studio-Token", "") == TOKEN


def _watchdog():
    """Self-destruct so an abandoned pod cannot keep billing.

    Three different ways to be worth nothing, and they need three clocks:
    nobody talking to us at all, somebody talking but never sending work (a pod
    whose model failed to load, kept awake by the studio's own health polls),
    and simply having been alive too long.
    """
    while True:
        time.sleep(15)
        with _lock:
            idle = time.time() - _last_seen
            no_work = time.time() - _last_job
            busy = len(_inflight)
            busy_for = (time.time() - min(_inflight.values())) if _inflight else 0.0
        alive = time.time() - _started
        # The hard ceilings apply whatever the pod is doing: a generate() that
        # never returns used to keep the pod alive forever because «busy»
        # skipped every clock below.
        if alive > MAX_LIFE_SEC:
            _self_terminate(f"предельное время жизни {alive:.0f}s")
        if busy and busy_for > BUSY_MAX_SEC:
            _self_terminate(f"одна генерация держит GPU {busy_for:.0f}s — завис")
        if busy:
            continue          # working is the opposite of abandoned
        if _loading:
            # The port is not even open yet, so silence here means nothing.
            if alive > LOAD_TIMEOUT_SEC:
                _self_terminate(f"модель не поднялась за {alive:.0f}s")
            continue
        if idle > IDLE_EXIT_SEC:
            _self_terminate(f"{idle:.0f}s без запросов")
        if no_work > NO_WORK_EXIT_SEC:
            _self_terminate(f"{no_work:.0f}s без единой задачи")


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as _m
        return _m.version(name)
    except Exception:
        return ""


_boot_marks = pod_lifecycle.boot_marks


@app.get("/health")
def health():
    _touch()
    with _lock:
        idle = time.time() - _last_seen
    return jsonify({
        "boot": _boot_marks(),
        "serving_since": _model_ready_at,
        "ok": True,
        "model_loaded": engine._model is not None,
        # Still coming, or never coming. Without these two the studio had to
        # guess from silence, and guessed wrong.
        "loading": _loading,
        "error": _load_error,
        "gpu": engine.gpu_name(),
        # Which torch actually ended up installed. On a base image whose torch
        # is older than the TTS stack wants, pip may replace it during boot —
        # a CPU build would be a 30x slowdown with no error, and even a CUDA
        # build costs minutes of download. This makes either visible.
        "torch": engine.torch.__version__,
        "cuda": engine.torch.version.cuda,
        # The inference code the pod actually runs. The studio pins the same
        # version; a mismatch here is a pod that would sound different.
        "qwen_tts": _pkg_version("qwen-tts"),
        "transformers": _pkg_version("transformers"),
        "model_revision": os.environ.get("QWEN_MODEL_REVISION", ""),
        "compiled": engine.is_compiled(),
        "uptime_sec": round(time.time() - _started),
        "idle_sec": round(idle),
    })


@app.get("/boot.log")
def boot_log():
    """What this pod printed on its way up — including why it failed.

    pod_boot.sh served this file itself, but it hands the port over before the
    model loads and its server never comes back. So the one pod whose log is
    actually worth reading — the one that got all the way here and then broke —
    was the one that answered 404. Serving it here closes that hole and keeps
    the studio's progress counter (a HEAD on this path) working to the end.
    """
    try:
        with open(BOOT_LOG, "rb") as f:
            return Response(f.read(), mimetype="text/plain; charset=utf-8")
    except OSError as e:
        return Response(f"(нет {BOOT_LOG}: {e})", status=404,
                        mimetype="text/plain; charset=utf-8")


@app.post("/generate")
def generate():
    if not _authorised():
        return jsonify({"error": "bad token"}), 401
    _touch()
    _touch_job()
    global _req_seq
    with _lock:
        _req_seq += 1
        rid = _req_seq
        # Queued behind another job: counted as busy, but its own clock only
        # starts once it holds the GPU — waiting is not being wedged.
        _inflight[rid] = time.time() + 10 ** 9
    try:
        with _gpu_lock:
            _touch()  # waiting for the lock still counts as being talked to
            with _lock:
                _inflight[rid] = time.time()
            out = engine.generate(request.get_json(silent=True) or {})
    finally:
        with _lock:
            _inflight.pop(rid, None)
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
    # The studio deletes the pod through its own API call right after this;
    # deleting ourselves as well is harmless (404 on a pod already gone) and
    # covers the case where the studio's call never arrives.
    threading.Timer(0.5, lambda: _self_terminate("команда студии")).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=_watchdog, daemon=True).start()

    # Load before serving: the studio waits on /health reporting model_loaded,
    # and a pod that accepts work before it is ready would just stall a batch.
    print("[pod] загружаю модель...", flush=True)
    try:
        engine.load_model()
        _model_ready_at = time.time()
        print(f"[pod] готов на {engine.gpu_name()}", flush=True)
    except Exception as e:
        # Serve anyway, but say what happened: /health carries the reason home,
        # so the studio can destroy this pod in seconds with the cause in its
        # log instead of waiting six minutes and inventing one.
        _load_error = engine.describe(e)
        print(f"[pod] модель не загрузилась: {_load_error}", flush=True)

    _loading = False
    # Both clocks start when serving starts. Leaving _last_seen at import time
    # meant a load slower than POD_IDLE_EXIT_SEC produced a pod that killed
    # itself in the first second of being useful — having paid for the whole
    # boot to get there.
    _last_job = _last_seen = time.time()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
