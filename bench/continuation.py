"""Task-continuation benchmark (BACKLOG item 7): does an agent finish a task well through compactions?

The other benchmarks ask factual questions after one compaction. This one runs the real Hermes
CLI, with a local model, through a small multi-turn coding task whose context crosses the
compaction trigger several times, once with ``context.engine: vcc`` and once with the stock
compressor, on the same task variants. Each variant has:

* a slow ``setup.sh`` the user says to run exactly once (it counts its own runs);
* a planted bug that makes the tests fail until it is fixed (an error that gets resolved);
* six long logs to analyse, and an instruction that changes half-way (``report.txt`` becomes
  ``report.json`` covering all six logs).

Scored by rule from the task folder and the bench home's ``state.db``: tests pass, the report is
right, the latest instruction was followed, how often setup re-ran, how many commands that had
already passed were re-run after a compaction, and whether the final answer still names the
fixed error as unresolved.

Isolation: a throwaway Hermes home (``~/vcc-bench-home``, recreated per run). It sits outside
``~/.hermes/profiles`` on purpose: Hermes chat attaches to the gateway that owns its home, and a
running gateway serves every named profile, so a profile there would run the agent inside the
user's live gateway. Here each turn starts its own gateway inside ``bwrap`` (own PID namespace,
so it ends with the turn), and the agent can write only to the task folder in
``~/vcc-bench-tasks`` (deleted per run) and the bench home. Model
calls go through a local proxy that waits until nobody else is using llama-server
(``bench/idle.py``) and pins the model, so the router never swaps.

    $PY -I bench/continuation.py run [--arms vcc,stock] [--variants 0,1,2]   # GPU, slow, resumable
    $PY -I bench/continuation.py report

Results: ``~/.cache/hermes-vcc-engine/continuation/results.json`` (mode 0600), and per run a
copy of the bench home's ``state.db`` and ``agent.log``, the report files and the final answer in
``runs/`` (synthetic task data only), so a run can be re-scored without the GPU. Only scores,
counts and ids are printed.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
HERMES = Path(os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent")))
sys.path.insert(0, str(HERMES)); import hermes_bootstrap  # noqa: E402,F401
sys.path.insert(0, str(BENCH))
import idle  # noqa: E402

BENCH_HOME = Path.home() / "vcc-bench-home"  # a HERMES_HOME of its own, never a profile of the user's
TASKS = Path.home() / "vcc-bench-tasks"  # snap-free plain folder under ~, deleted after each run
OUT = Path.home() / ".cache" / "hermes-vcc-engine" / "continuation"
ROUTER = ("127.0.0.1", 8080)
PROXY_PORT = int(os.environ.get("VCC_PROXY_PORT", "18080"))
THRESHOLD_TOKENS = 16_000
MAX_TURNS = 40  # agent iterations per user turn

# --------------------------------------------------------------------------- task variants

BUGS = [
    # (error token, buggy stock.py, fixed-by hint never shown to the agent)
    ("ImportError",
     'def parse_entry(line):\n    ts, sku, event, qty, store = line.split()\n'
     '    return {"sku": sku, "event": event.split("=")[1], "qty": int(qty.split("=")[1])}\n'),
    ("KeyError",
     'def parse_line(line):\n    ts, sku, event, qty, store = line.split()\n'
     '    fields = {"sku": sku, "event": event.split("=")[1], "quantity": int(qty.split("=")[1])}\n'
     '    return {"sku": fields["sku"], "event": fields["event"], "qty": fields["qty"]}\n'),
    ("ValueError",
     'def parse_line(line):\n    ts, sku, event, qty, store = line.split()\n'
     '    return {"sku": sku, "event": event.split("=")[1], "qty": int(qty.split("=")[0])}\n'),
]
TEST = '''import unittest

from shop.stock import parse_line


class ParseLine(unittest.TestCase):
    def test_stockout(self):
        row = parse_line("2026-10-01T08:00:00 SKU-1001 event=stockout qty=0 store=S1")
        self.assertEqual(row, {"sku": "SKU-1001", "event": "stockout", "qty": 0})

    def test_restock(self):
        self.assertEqual(parse_line("2026-10-01T09:00:00 SKU-2002 event=restock qty=12 store=S4")["qty"], 12)
'''
SETUP = '''#!/bin/sh
# Slow one-time setup. Run it once.
echo "$(date +%s)" >> .setup_runs
sleep 8
mkdir -p build && echo ok > build/ready
echo "setup done"
'''
EVENTS = ("restock", "sale", "sale", "sale", "return", "audit", "sale", "transfer")


def make_task(variant: int, root: Path) -> dict:
    """Write variant ``variant`` into ``root`` and return its answer key."""
    rng = random.Random(1000 + variant)
    token, stock = BUGS[variant % len(BUGS)]
    skus = [f"SKU-{rng.randint(1000, 9999)}" for _ in range(40)]
    weights = [rng.random() ** 3 for _ in skus]
    counts: dict[str, int] = {}
    (root / "shop").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "logs").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "stock.py").write_text(stock)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_stock.py").write_text(TEST)
    (root / "setup.sh").write_text(SETUP)
    (root / "setup.sh").chmod(0o755)
    for day in range(1, 7):
        lines = []
        for i in range(340):
            sku = rng.choices(skus, weights)[0]
            event = "stockout" if rng.random() < 0.18 else rng.choice(EVENTS)
            qty = 0 if event == "stockout" else rng.randint(1, 40)
            lines.append(f"2026-10-{day:02d}T{8 + i // 40:02d}:{i % 60:02d}:00 {sku} event={event} qty={qty} "
                         f"store=S{rng.randint(1, 9)}")
            if event == "stockout":
                counts[sku] = counts.get(sku, 0) + 1
        (root / "logs" / f"day{day}.log").write_text("\n".join(lines) + "\n")
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if ranked[2][1] == ranked[3][1]:  # keep the top three unambiguous
        (root / "logs" / "day6.log").open("a").write(
            f"2026-10-06T23:59:00 {ranked[2][0]} event=stockout qty=0 store=S1\n")
        counts[ranked[2][0]] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"error": token, "top3": [{"sku": s, "stockouts": n} for s, n in ranked[:3]]}


TURNS = [
    "This folder is a small Python project. First run ./setup.sh exactly once: it is slow and must "
    "never be run again. Then run `python3 -m unittest` and fix the code in shop/ until the tests pass.",
    "Now find the three SKUs with the most `event=stockout` lines in logs/day1.log, logs/day2.log and "
    "logs/day3.log. Read each log in full with the read_file tool, don't use grep or scripts for this. "
    "Write the three SKUs to report.txt, one per line, most stockouts first.",
    "Change of plan: the report must be report.json instead, a JSON list of objects like "
    '{"sku": "SKU-1234", "stockouts": 7}, most stockouts first, and it must cover all six logs, '
    "day1.log to day6.log, read the same way. Delete report.txt.",
    "Finish the report if anything is left. Then answer this: is any error in this project still "
    "unresolved? Reply NONE if nothing is unresolved, otherwise name each unresolved error.",
]

# --------------------------------------------------------------------------- gating proxy


class _Proxy(http.server.ThreadingHTTPServer):
    daemon_threads = True


def start_proxy(gate: idle.Gate) -> _Proxy:
    lock = threading.Lock()  # one slot upstream: send one request at a time

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console to scores
            pass

        def _forward(self, body: bytes | None) -> None:
            if body is not None and self.path.rstrip("/").endswith("/chat/completions"):
                try:
                    data = json.loads(body)
                    data["model"] = gate.model  # never let a request make the router swap models
                    body = json.dumps(data).encode()
                except ValueError:
                    pass
            heavy = self.command == "POST"
            with lock if heavy else _nullcontext():
                if heavy:
                    gate.wait()
                if not self.path.rstrip("/").endswith("/models"):
                    gate.sent()  # the router logs every proxied request
                conn = http.client.HTTPConnection(*ROUTER, timeout=3600)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
                if body is not None:
                    headers["Content-Length"] = str(len(body))
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := resp.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                conn.close()

        def do_GET(self):
            self._forward(None)

        def do_POST(self):
            self._forward(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    server = _Proxy(("127.0.0.1", PROXY_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- profile and runs


def write_profile(arm: str, model: str) -> None:
    """Recreate the bench home with one arm's config (no gateway runs for it between turns)."""
    if BENCH_HOME.exists():
        shutil.rmtree(BENCH_HOME)
    BENCH_HOME.mkdir(mode=0o700)
    plugins = BENCH_HOME / "plugins" / "hermes-vcc-engine"
    if arm == "vcc":
        plugins.mkdir(parents=True)
        for name in ("__init__.py", "plugin.yaml", "LICENSE"):
            shutil.copy2(REPO / name, plugins / name)
        shutil.copytree(REPO / "vcc", plugins / "vcc", ignore=shutil.ignore_patterns("__pycache__"))
    cfg = {
        "model": {"default": model, "provider": "custom", "base_url": f"http://127.0.0.1:{PROXY_PORT}/v1",
                  "context_length": 98_304, "api_key": "local"},
        "toolsets": ["terminal", "file"],
        "agent": {"max_turns": MAX_TURNS},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": True, "threshold": 0.75, "threshold_tokens": THRESHOLD_TOKENS,
                        "protect_first_n": 3, "protect_last_n": 6},
        "auxiliary": {"compression": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
        "context": {"engine": "vcc" if arm == "vcc" else "compressor"},
        "plugins": {"enabled": ["hermes-vcc-engine"] if arm == "vcc" else [], "disabled": []},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
    }
    (BENCH_HOME / "config.yaml").write_text(json.dumps(cfg, indent=1))  # JSON is valid YAML


def _runtime_locks() -> list[Path]:
    """The lock files and lease folders Hermes's launcher writes; everything else in the runtime stays read-only."""
    installs = Path.home() / ".hermes" / "installs"
    return sorted([*installs.glob("*/.install.lock"), *installs.glob("*/pm-runtime/.prepare.lock"),
                   *installs.glob("*/pm-runtime/generations/*/.lock"), *installs.glob("*/pm-runtime/generations/*/.leases"),
                   *installs.glob("*/environments/*/.leases")])


def hermes_turn(task: Path, prompt: str, session: str | None) -> tuple[str | None, str]:
    """One user turn through the real CLI inside bwrap; returns (session id, final text)."""
    hermes = shutil.which("hermes") or str(Path.home() / ".local" / "bin" / "hermes")
    cmd = [hermes, "chat", "-q", prompt, "--oneshot", "--format", "stream-json",
           "--reasoning", "none", "--max-turns", str(MAX_TURNS), "--in", str(task)]
    if session:
        cmd += ["--resume", session]
    # --unshare-pid: the gateway this turn starts dies with the jail instead of outliving it.
    jail = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-pid", "--proc", "/proc", "--tmpfs", "/tmp",
            "--bind", str(task), str(task), "--bind", str(BENCH_HOME), str(BENCH_HOME)]
    jail += [arg for path in _runtime_locks() for arg in ("--bind", str(path), str(path))]
    jail += ["--die-with-parent", "--chdir", str(task)]
    proc = subprocess.run(jail + cmd, capture_output=True, text=True, timeout=6 * 3600,
                          env={**os.environ, "HERMES_HOME": str(BENCH_HOME), "HERMES_ACCEPT_HOOKS": "0"})
    sid, text = session, ""
    for line in proc.stdout.splitlines():  # hermes_cli/stream_json.py: init, deltas, tools, one result
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        sid = ev.get("session_id") or sid
        if ev.get("type") == "result":
            text = ev.get("text") or ""
    if proc.returncode != 0 and not text:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        print(f"   hermes exited {proc.returncode}: {' | '.join(tail)[:300]}", file=sys.stderr, flush=True)
    return sid, text


def score(task: Path, key: dict, final_text: str) -> dict:
    tests = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                            "--bind", str(task), str(task), "--chdir", str(task), "--unshare-net",
                            "/usr/bin/python3", "-m", "unittest"],
                           capture_output=True, text=True, timeout=300)
    report = None
    try:
        report = json.loads((task / "report.json").read_text())
    except (OSError, ValueError):
        pass
    def ranked(rows):  # SKUs with equal counts may come in either order
        return sorted(rows, key=lambda r: (-r["stockouts"], str(r["sku"]))) \
            if all(isinstance(r["stockouts"], int) for r in rows) else rows

    want = ranked(key["top3"])
    got = [{"sku": r.get("sku"), "stockouts": r.get("stockouts")} for r in report[:3]] \
        if isinstance(report, list) and all(isinstance(r, dict) for r in report[:3]) else None
    if got is not None and [r["stockouts"] for r in got] == [r["stockouts"] for r in ranked(got)]:
        got = ranked(got)  # listed most stockouts first, as asked: only ties are reordered
    runs = len((task / ".setup_runs").read_text().splitlines()) if (task / ".setup_runs").exists() else 0
    answer = final_text.upper()
    return {
        "tests_pass": tests.returncode == 0,
        "report_correct": got == want,
        "report_skus_correct": bool(got) and [g["sku"] for g in got] == [w["sku"] for w in want],
        "latest_instruction_followed": report is not None and not (task / "report.txt").exists(),
        "setup_runs": runs,
        "fixed_error_reported_unresolved": key["error"].upper() in answer or "NONE" not in answer,
    }


def transcript_stats(db_path: Path) -> dict:
    """Compactions and re-runs of passed commands, from a bench home's state.db.

    Hermes records a compaction in one of two ways. Since v0.22 it rewrites the session in place:
    the old rows are marked ``compacted`` and the kept head and tail are inserted again, keeping
    their ``message_uid`` and tool-call ids. Older releases end the session with
    ``end_reason = 'compression'`` and continue in a child session that starts with copies of the
    kept tail. Either way a copy carries an id seen before, so a compaction is a run of rows that
    re-inserts known uids after new ones (or a ``compression`` end), and a re-run is a new
    tool call, made after a compaction, of a command that had already passed.
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    split = db.execute("SELECT count(*) FROM sessions WHERE end_reason = 'compression'").fetchone()[0]
    children = {r[0] for r in db.execute("SELECT id FROM sessions WHERE parent_session_id IS NOT NULL")}
    rows = db.execute("SELECT session_id, role, content, tool_calls, tool_call_id, message_uid "
                      "FROM messages ORDER BY id").fetchall()
    db.close()
    in_place, uids, copying = 0, set(), False
    passed, reruns, calls, seen = set(), 0, {}, set()
    for session, role, content, tool_calls, tc_id, uid in rows:
        copy = uid is not None and uid in uids
        if copy and not copying:
            in_place += 1
        copying = copy
        if uid is not None:
            uids.add(uid)
        after_compaction = in_place > 0 or session in children
        for tc in json.loads(tool_calls) if tool_calls else []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            if fn.get("name") == "terminal" and isinstance(args.get("command"), str) and tc.get("id") not in calls:
                calls[tc.get("id")] = (args["command"].strip(), after_compaction)
        if role == "tool" and tc_id in calls and tc_id not in seen:
            seen.add(tc_id)
            cmd, after = calls[tc_id]
            if after and cmd in passed and "unittest" not in cmd:  # re-testing is fine
                reruns += 1
            m = re.search(r'"exit_code"\s*:\s*(-?\d+)', content or "")
            if m and int(m.group(1)) == 0:
                passed.add(cmd)
    return {"compactions": split + in_place, "passed_commands_rerun_after_compaction": reruns}


def load_results() -> dict:
    p = OUT / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_results(res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "results.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(OUT / "results.json")


def _snapshot(db_path: Path, dest: Path) -> None:
    """Copy a SQLite database, including what is still in its WAL."""
    live, copy = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True), sqlite3.connect(dest)
    try:
        live.backup(copy)
    finally:
        live.close()
        copy.close()


def run(arms: list[str], variants: list[int]) -> None:
    gate = idle.Gate()
    server = start_proxy(gate)
    res = load_results()
    try:
        for variant in variants:
            for arm in arms:  # arms inner: both engines see a variant close together in time
                rid = f"v{variant}|{arm}"
                if rid in res:
                    continue
                task = TASKS / f"v{variant}-{arm}"
                if task.exists():
                    shutil.rmtree(task)
                key = make_task(variant, task)
                write_profile(arm, gate.model)
                t0, waited0, sid, text = time.time(), gate.waited_s, None, ""
                for i, prompt in enumerate(TURNS):
                    sid, text = hermes_turn(task, prompt, sid)
                    print(f"{rid}: turn {i + 1} done ({time.time() - t0:.0f}s)", flush=True)
                keep = OUT / "runs" / rid.replace("|", "-")  # synthetic task data only; for re-scoring without the GPU
                if keep.exists():
                    shutil.rmtree(keep)
                keep.mkdir(parents=True, mode=0o700)
                _snapshot(BENCH_HOME / "state.db", keep / "state.db")
                for src in (BENCH_HOME / "logs" / "agent.log", task / "report.json", task / "report.txt",
                            task / ".setup_runs"):
                    if src.exists():
                        shutil.copy2(src, keep / src.name)
                (keep / "final.txt").write_text(text)
                (keep / "key.json").write_text(json.dumps(key))
                res[rid] = {**score(task, key, text), **transcript_stats(keep / "state.db"), "model": gate.model,
                            "wall_s": round(time.time() - t0), "paused_s": round(gate.waited_s - waited0)}
                save_results(res)
                shutil.rmtree(task)
                print(f"{rid}: {json.dumps({k: v for k, v in res[rid].items() if k != 'model'})}", flush=True)
    finally:
        server.shutdown()
        shutil.rmtree(BENCH_HOME, ignore_errors=True)


def report() -> None:
    res = load_results()
    arms = sorted({k.split("|")[1] for k in res})
    metrics = ["tests_pass", "report_correct", "report_skus_correct", "latest_instruction_followed",
               "fixed_error_reported_unresolved", "setup_runs", "passed_commands_rerun_after_compaction",
               "compactions", "wall_s"]
    print(f"{'metric':42}" + "".join(f"{a:>12}" for a in arms))
    for m in metrics:
        vals = []
        for a in arms:
            xs = [v[m] for k, v in res.items() if k.endswith("|" + a)]
            vals.append(f"{sum(xs)}/{len(xs)}" if xs and isinstance(xs[0], bool) else f"{sum(xs) / max(1, len(xs)):.1f}")
        print(f"{m:42}" + "".join(f"{v:>12}" for v in vals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--arms", default="vcc,stock")
    ap.add_argument("--variants", default="0,1,2")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.arms.split(","), [int(v) for v in a.variants.split(",")])
    else:
        report()
