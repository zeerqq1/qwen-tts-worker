#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless entry point.

All the actual work lives in engine.py, which pod_server.py also uses — the two
worker shapes must produce identical audio, and the only way to guarantee that
is to run identical code.

Nothing persists here. The model is in the image, the voice reference arrives in
the request, the audio leaves in the response, and the container's filesystem
dies with the job. There is no volume, no bucket, and nothing to clean up.

Request:
    {"input": {
        "texts":         ["абзац", "..."],     # required
        "language":      "Russian",
        "ref_audio_b64": "<base64 wav>",
        "ref_text":      "точная расшифровка референса",
        "ref_id":        "voice_ru",
        "sampling":      {"temperature": 0.8, ...},
        "tail_sec":      0.15
    }}

Response:
    {"sr": 24000, "clips": ["<base64 flac>", ...],
     "gen_sec": 12.3, "audio_sec": 45.6, "realtime_x": 3.7, "gpu": "..."}
"""

import runpod

import engine


def handler(job):
    return engine.generate(job.get("input") or {})


runpod.serverless.start({"handler": handler})
