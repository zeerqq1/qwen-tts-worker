#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The one thing every pod in this fleet must get right: ending its own billing.

Both servers — pod_server.py (Qwen) and higgs_server.py (Higgs on SGLang-Omni)
— import this rather than carrying a copy each. The lesson that produced it is
expensive enough to state once: *exiting the process does not stop the bill*.
RunPod restarts a pod's container when its command exits, so a pod abandoned by
the studio looped through boot forever and billed until the next studio launch
swept it. The machine has to delete itself through the API.

RunPod injects RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY into every pod, so
the pod can call the same REST endpoint the studio uses.
"""

import json
import os
import time
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"


def self_terminate(reason: str, exit_code: int = 0):
    """Delete this pod through the RunPod API, then exit. Never returns."""
    print(f"[pod] {reason} — снимаю под", flush=True)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    key = os.environ.get("RUNPOD_API_KEY", "")
    if pod_id and key:
        _end_pod(pod_id, key)
    else:
        print("[pod] RUNPOD_POD_ID/RUNPOD_API_KEY нет в окружении — только выхожу", flush=True)
    time.sleep(2)
    os._exit(exit_code)


def _end_pod(pod_id: str, key: str) -> bool:
    """Every way this pod can end itself, in order, until one is accepted.

    The key RunPod injects is scoped to the pod, and the REST API answers it
    with 403 (measured). The GraphQL mutations are what a pod-scoped key is
    for. REST is still tried first: a full key, if one were ever given,
    works there too.
    """
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    attempts = (
        ("DELETE", f"{REST}/pods/{pod_id}", None),
        ("GQL", GRAPHQL, {"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id}),
        ("POST", f"{REST}/pods/{pod_id}/stop", None),
        ("GQL", GRAPHQL, {"query": 'mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod_id}),
    )
    for method, url, body in attempts:
        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, method="POST" if method == "GQL" else method,
                                         headers=headers, data=data)
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")[:200]
                if method == "GQL" and '"errors"' in text:
                    raise RuntimeError(text)
                print(f"[pod] {method} {url.rsplit('/', 1)[-1]}: {r.status} {text}", flush=True)
            return True
        except Exception as e:
            print(f"[pod] {method} {url.rsplit('/', 1)[-1]} не удался: {e}", flush=True)
    return False


def boot_marks(path: str = "/workspace/boot_times.txt") -> dict:
    """Timing marks the boot script left behind, as epoch seconds.

    The studio subtracts them from the moment it rented the pod, which is the
    only way to see the image pull — that happens before any of our code runs
    and is otherwise invisible.
    """
    out = {}
    try:
        for line in open(path, encoding="utf-8"):
            name, _, ts = line.strip().partition(" ")
            if name and ts:
                out[name] = float(ts)
    except Exception:
        pass
    return out
