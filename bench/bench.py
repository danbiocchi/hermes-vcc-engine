"""Benchmark and byte-identity harness for hermes-vcc-engine.

Runs every compaction path over the sessions in a Hermes ``state.db`` (opened read-only)
and records two things:

* timings (median of ``--repeat`` runs) for the projector, the anchor index, the engine's
  ``_generate_summary`` and a full ``compress()``;
* a SHA-256 of every output, so an optimization can be proven to change nothing.

Run it with the interpreter Hermes itself uses, so the numbers match production:

    ~/.hermes/tools/python-*/bin/python3 -I bench/bench.py --write-golden   # before a change
    ~/.hermes/tools/python-*/bin/python3 -I bench/bench.py --check-golden   # after it

``--repo`` points at another checkout of this plugin (e.g. a worktree of an older tag) to
time two versions side by side.

``--write-golden`` freezes the transcripts it read into a private corpus file (mode 0600,
under ``~/.cache``, never in the repo) and ``--check-golden`` replays exactly that corpus,
so a session that keeps growing in ``state.db`` can't masquerade as a regression.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

DEFAULT_GOLDEN = Path.home() / ".cache" / "hermes-vcc-engine" / "golden.json"
DEFAULT_CORPUS = Path.home() / ".cache" / "hermes-vcc-engine" / "corpus.json"
PROJECTOR_BUDGETS = (6000, 23592)
CONTEXT_LENGTH = 98_304


def _setup(repo: Path, hermes_dir: Path) -> None:
    sys.path.insert(0, str(hermes_dir))
    import hermes_bootstrap  # noqa: F401  (activates Hermes's dependency tree)

    sys.path.insert(0, str(repo))


def _load_sessions(db_path: Path, min_messages: int) -> list[tuple[str, list[dict]]]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    ids = [r[0] for r in db.execute(
        "SELECT session_id FROM messages GROUP BY session_id HAVING count(*) >= ? ORDER BY session_id",
        (min_messages,))]
    sessions = []
    for sid in ids:
        msgs = []
        for r in db.execute("SELECT id, role, content, tool_calls, tool_call_id, tool_name FROM messages "
                            "WHERE session_id = ? ORDER BY id", (sid,)):
            m = {"role": r["role"], "content": r["content"], "_row_id": r["id"]}
            if r["tool_calls"]:
                try:
                    m["tool_calls"] = json.loads(r["tool_calls"])
                except ValueError:
                    pass
            if r["tool_call_id"]:
                m["tool_call_id"] = r["tool_call_id"]
            if r["tool_name"]:
                m["tool_name"] = r["tool_name"]
            msgs.append(m)
        sessions.append((sid, msgs))
    db.close()
    return sessions


_FUZZ = (
    "ab", " ", "\n", "\t", "`", "##", "<|im_start|>", "[INST]", "</s>", "<think>x</think>", "\x1b[31m", "\x1b]0;t\x07",
    "\x00", "\x07", "\u200b", "\u202e", "\u2028", "\u00a0", "\u3000", "\u2066", "é", "中文", "🙂",
    "/home/u/proj/src/app.py", "docs/README.md", "cfg.yaml", "x.json", "fix/login-bug", "feat/new-ui", "ent/thing",
    "ValueError: bad", "Exception", "Traceback (most recent call last)", "ENOSPC", "error:", "FAILED",
    "deadbeefcafe0123", "0123456789abcdef0123456789abcdef01234567", "a1b2c3d4e", "#1234", "#12", "@someone",
    "https://example.com/a/very/long/path?q=1", "exit code 2", "Permission denied", ".", ",", ";", ":",
)


def _synthetic_sessions() -> list[tuple[str, list[dict]]]:
    """Seeded transcripts that stress every code path, so byte-identity is proven beyond one user's data."""
    import random

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import transcripts as t

    out = [("synthetic-fix", t.fix_the_build())]
    for seed, rounds, chars in ((1, 30, 400), (2, 120, 2500), (3, 400, 1200)):
        out.append((f"synthetic-long-{seed}", t.long_session(rounds, output_chars=chars, seed=seed)))
    for seed in range(4):
        rng = random.Random(100 + seed)

        def blob(n):
            return "".join(rng.choice(_FUZZ) for _ in range(n))

        msgs = []
        for i in range(rng.randint(40, 160)):
            kind = rng.random()
            if kind < 0.2:
                msgs.append(t.user(blob(rng.randint(1, 80))))
            elif kind < 0.6:
                cid = f"f{seed}-{i}"
                name = rng.choice(["terminal", "read_file", "patch", "web_extract", "execute_code", "mystery"])
                args = {"command": blob(8)} if name == "terminal" else {"path": blob(3), "x": blob(4)}
                msgs.append(t.assistant(blob(rng.randint(0, 30)), [t.call(cid, name, **args)]))
                if rng.random() < 0.5:
                    payload = {"output": blob(rng.randint(0, 400)), "exit_code": rng.choice([0, 0, 1, 2]),
                               "error": rng.choice([None, None, blob(5)])}
                else:
                    payload = blob(rng.randint(0, 3000))
                msgs.append(t.result(cid, payload, name))
            else:
                msgs.append(t.assistant(blob(rng.randint(1, 60))))
        out.append((f"synthetic-fuzz-{seed}", msgs))
    return out


def _sha(value) -> str:
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8", "surrogatepass")).hexdigest()


def _median_ms(fn, repeat: int) -> tuple[float, object]:
    result = fn()  # warm-up; also the value we hash
    gc.collect()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), result


def _engine(engine_mod, sid: str):
    # No LLM fallback: a projection error must fail the benchmark, not quietly call a model.
    eng = engine_mod.VccEngine(config={"context": {"vcc": {"fallback_to_llm": False}}},
                               config_context_length=CONTEXT_LENGTH)
    eng.update_model(model="bench/local", context_length=CONTEXT_LENGTH)
    eng._session_id = sid
    return eng


def _no_fallback(eng) -> None:
    if eng._vcc_fallbacks:
        raise AssertionError("the projection failed; the engine had to fall back")


def run(args) -> int:
    _setup(Path(args.repo).resolve(), Path(args.hermes_dir).expanduser())
    import agent.context_compressor as cc
    import engine as engine_mod
    import projector

    anchors = None
    try:
        import anchors as anchors  # noqa: PLC0414 (present from v0.2)
    except ImportError:
        pass

    corpus = Path(args.corpus).expanduser()
    if args.check_golden:
        sessions = [(sid, msgs) for sid, msgs in json.loads(corpus.read_text())]
    else:
        sessions = _load_sessions(Path(args.state_db).expanduser(), args.min_messages)
        if args.write_golden:
            corpus.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(corpus, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sessions, fh, ensure_ascii=False)
    if not args.no_synthetic:
        sessions += _synthetic_sessions()
    hashes: dict[str, str] = {}
    rows = []
    for sid, msgs in sessions:
        size = sum(len(m.get("content") or "") for m in msgs)
        row = {"session": sid[:16], "messages": len(msgs), "chars": size}

        t, outs = _median_ms(lambda: [projector.project(msgs, budget_chars=b, session_id=sid, focus="error")
                                      for b in PROJECTOR_BUDGETS], args.repeat)
        row["projector_ms"] = t
        for b, out in zip(PROJECTOR_BUDGETS, outs):
            hashes[f"{sid}|projector|{b}"] = _sha(out)

        t, _ = _median_ms(lambda: cc._build_anchor_index(msgs), args.repeat)
        row["anchor_hermes_ms"] = t
        if anchors is not None:
            t, fast = _median_ms(lambda: anchors.build(msgs), args.repeat)
            row["anchor_fast_ms"] = t
            if fast != cc._build_anchor_index(msgs):
                print(f"!! fast anchor index differs from Hermes on {sid}", file=sys.stderr)
                return 2

        def one_step():
            eng = _engine(engine_mod, sid)
            out = eng._generate_summary(copy.copy(msgs))
            _no_fallback(eng)
            return out

        t, out = _median_ms(one_step, args.repeat)
        row["summary_ms"] = t
        hashes[f"{sid}|engine|1"] = _sha(out)

        half = len(msgs) // 2

        def two_step():
            eng = _engine(engine_mod, sid)
            first = eng._generate_summary(msgs[:half])
            second = eng._generate_summary(msgs[half:])
            _no_fallback(eng)
            return first, second

        t, (first, second) = _median_ms(two_step, args.repeat)
        row["summary_2step_ms"] = t
        hashes[f"{sid}|engine|2a"] = _sha(first)
        hashes[f"{sid}|engine|2b"] = _sha(second)

        def full_compress():
            eng = _engine(engine_mod, sid)
            out = eng.compress([{"role": "system", "content": "You are Hermes."}, *copy.deepcopy(msgs)], force=True)
            _no_fallback(eng)
            return out

        t, out = _median_ms(full_compress, args.repeat)
        row["compress_ms"] = t
        hashes[f"{sid}|compress"] = _sha(out)
        rows.append(row)

    _report(rows)
    return _golden(args, hashes)


def _report(rows: list[dict]) -> None:
    cols = ["projector_ms", "anchor_hermes_ms", "anchor_fast_ms", "summary_ms", "summary_2step_ms", "compress_ms"]
    cols = [c for c in cols if any(c in r for r in rows)]
    print(f"{'session':16} {'msgs':>5} {'chars':>10} " + " ".join(f"{c[:-3]:>15}" for c in cols))
    for r in sorted(rows, key=lambda r: r["chars"]):
        print(f"{r['session']:16} {r['messages']:5d} {r['chars']:10,} "
              + " ".join(f"{r[c]:15.2f}" if c in r else f"{'-':>15}" for c in cols))
    totals = {c: sum(r.get(c, 0.0) for r in rows) for c in cols}
    print(f"{'TOTAL':16} {sum(r['messages'] for r in rows):5d} {sum(r['chars'] for r in rows):10,} "
          + " ".join(f"{totals[c]:15.2f}" for c in cols))


def _golden(args, hashes: dict[str, str]) -> int:
    path = Path(args.golden).expanduser()
    if args.write_golden:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hashes, indent=1, sort_keys=True))
        print(f"wrote {len(hashes)} output hashes to {path}")
        return 0
    if args.check_golden:
        golden = json.loads(path.read_text())
        missing = sorted(set(golden) - set(hashes))
        changed = sorted(k for k in golden if k in hashes and golden[k] != hashes[k])
        if missing or changed:
            print(f"BYTE-IDENTITY FAILED: {len(changed)} changed, {len(missing)} missing")
            for key in (changed + missing)[:20]:
                print("  ", key)
            return 1
        print(f"byte-identical: {len(golden)}/{len(golden)} outputs match the golden hashes")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-db", default="~/.hermes/state.db")
    ap.add_argument("--hermes-dir", default="~/.hermes/hermes-agent")
    ap.add_argument("--repo", default=str(here), help="plugin checkout to benchmark")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--min-messages", type=int, default=12)
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="frozen inputs written/read with the golden")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the seeded stress transcripts")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write-golden", action="store_true")
    mode.add_argument("--check-golden", action="store_true")
    args = ap.parse_args()
    for name in ("state_db", "hermes_dir", "repo", "golden", "corpus"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    os.chdir(os.path.expanduser("~"))  # keep Hermes's cwd-sensitive helpers away from the repo
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
