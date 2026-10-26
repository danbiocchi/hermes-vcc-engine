"""Wait until the local model server is free, so a benchmark never competes with a person.

The benchmarks share one llama.cpp router (one slot) with the people using Hermes on the same
machine. Before each request, ``wait_until_idle`` blocks until all of these hold:

* the benchmark's model is the one already loaded (the benchmark never loads or swaps a model:
  two models on 8 GB cards run out of memory, and an image job may have unloaded it on purpose);
* no Hermes WebUI run is active on a local model (``/health``; runs on remote providers don't
  use the GPU and are ignored);
* nobody else's request has reached the router for ``VCC_IDLE_QUIET_S`` seconds (counted from
  the router's journal, minus this process's own requests);
* the slot is not processing.

The benchmark's model is ``VCC_BENCH_MODEL``, else Hermes's configured default (``model.default``).
Everything is read-only: it polls two HTTP endpoints and ``journalctl --user``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROUTER = os.environ.get("VCC_ROUTER_URL", "http://127.0.0.1:8080")
WEBUI_HEALTH = os.environ.get("VCC_WEBUI_HEALTH", "http://127.0.0.1:8787/health")
QUIET_S = float(os.environ.get("VCC_IDLE_QUIET_S", "120"))
POLL_S = 15.0
UNIT = os.environ.get("VCC_ROUTER_UNIT", "llama-server")
_PROXY_LINE = "proxying request to model"
_LOCAL_PROVIDERS = ("custom", "local", "llama")  # Hermes provider names that point at this router


def _get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def bench_model() -> str:
    if os.environ.get("VCC_BENCH_MODEL"):
        return os.environ["VCC_BENCH_MODEL"]
    try:
        from hermes_cli.config import load_config_readonly  # Hermes is importable in every benchmark

        model = (load_config_readonly() or {}).get("model") or {}
        if isinstance(model, dict) and model.get("default"):
            return str(model["default"])
    except Exception:
        pass
    raise SystemExit("set VCC_BENCH_MODEL: no model.default in ~/.hermes/config.yaml")


def loaded_models() -> list[str]:
    data = _get(ROUTER + "/models").get("data", [])
    return [m["id"] for m in data if (m.get("status") or {}).get("value") == "loaded"]


class Gate:
    """Counts the router's proxied requests that aren't ours, and blocks until the server is free."""

    def __init__(self, model: str | None = None):
        self.model = model or bench_model()
        self.ours = 0  # requests this process sent through the router since the cursor
        self.cursor = self._journal()[1]
        self.last_other = time.monotonic()  # assume someone was just active
        self.waited_s = 0.0

    def _journal(self) -> tuple[int, str]:
        cmd = ["journalctl", "--user", "-u", UNIT, "--no-pager", "-o", "cat", "--show-cursor"]
        cmd += [f"--after-cursor={self.cursor}"] if getattr(self, "cursor", "") else ["-n", "0"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.splitlines()
        cursor = next((line[len("-- cursor: "):] for line in reversed(out) if line.startswith("-- cursor: ")),
                      getattr(self, "cursor", ""))
        return sum(_PROXY_LINE in line for line in out), cursor

    def _others_active(self) -> bool:
        seen, self.cursor = self._journal()
        others = seen - self.ours
        self.ours = max(0, -others)  # our requests not logged yet stay counted
        if others > 0:
            self.last_other = time.monotonic()
        return time.monotonic() - self.last_other < QUIET_S

    def _webui_busy(self) -> bool:
        try:
            runs = _get(WEBUI_HEALTH).get("runs") or []
        except Exception:
            return False  # WebUI down: nobody is using it
        return any(str(r.get("provider") or "").lower().startswith(_LOCAL_PROVIDERS)
                   or self.model in str(r.get("model") or "") for r in runs)

    def _slot_busy(self) -> bool:
        self.ours += 1  # the router logs this poll like any other request
        q = urllib.parse.urlencode({"model": self.model})
        return any(s.get("is_processing") for s in _get(f"{ROUTER}/slots?{q}"))

    def reason(self) -> str:
        """Why the server isn't free right now, or "" when it is."""
        if self.model not in loaded_models():
            return f"waiting for {self.model} to be loaded (never loaded by the benchmark)"
        if self._webui_busy():
            return "a WebUI run is using the local model"
        if self._others_active():
            return f"someone used the router in the last {QUIET_S:.0f}s"
        if self._slot_busy():
            return "the slot is processing"
        return ""

    def wait(self) -> None:
        """Block until the server is free; call right before each request, then count it with sent()."""
        start, said = time.monotonic(), ""
        while True:
            why = self.reason()
            if not why:
                break
            if why != said:
                print(f"[idle] paused: {why}", file=sys.stderr, flush=True)
                said = why
            time.sleep(POLL_S)
        if said:
            print(f"[idle] resumed after {time.monotonic() - start:.0f}s", file=sys.stderr, flush=True)
        self.waited_s += time.monotonic() - start

    def sent(self) -> None:
        self.ours += 1


if __name__ == "__main__":  # print the current state once
    g = Gate()
    print(f"model {g.model}; loaded {loaded_models()}; " + (g.reason() or "free"))
