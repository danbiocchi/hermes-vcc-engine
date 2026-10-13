"""Deterministic proofs that a vcc compaction invents nothing and loses nothing by reference.

1. Faithfulness: every line of a session log cites a message ``[#N]``. The line is checked
   against that message's raw stored data: the role and tool must match, an exit status must
   equal the recorded exit code, and every excerpt fragment must occur verbatim in the
   message (after the same documented sanitising: control characters, chat markers,
   whitespace). Nothing in a projection can come from anywhere else.
2. Reference integrity: every ``[#N]`` in the real summaries stored in ``state.db`` is opened
   with Hermes's own ``session_search`` tool (on a snapshot copy of the database) and must
   come back as the original message.

Runs on the frozen bench corpus plus every vcc summary in the database. Prints aggregate
numbers only; no transcript text leaves this machine.

    ~/.hermes/tools/python-*/bin/python3 -I bench/proofs.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

LINE_USER = re.compile(r"^\[#(\d+)\] (user|assistant): (.*)$")
LINE_CALL = re.compile(r"^\[#(\d+)\] → (\S+?)(?:: `(.*)`)?$")
LINE_RESULT = re.compile(r"^\[#(\d+)\] ← (\S+) (exit -?\d+|ok|ERROR exit -?\d+|ERROR)(?:: (.*))?$")
REF = re.compile(r"\[#(\d+)\]")
ALNUM_RUN = re.compile(r"[A-Za-z0-9]{2,}")


def _setup(repo: Path, hermes_dir: Path) -> None:
    sys.path.insert(0, str(hermes_dir))
    import hermes_bootstrap  # noqa: F401

    sys.path.insert(0, str(repo))


def _values(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _values(v)] + [str(k) for k in obj]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _values(v)]
    return [] if obj is None else [str(obj)]


class Checker:
    def __init__(self, projector):
        self.p = projector
        self.lines = 0
        self.verified = 0
        self.failures: list[str] = []

    def norm(self, text: str) -> str:
        return self.p.clean(text or "", 10**9)

    def _leaves_and_compositions(self, obj) -> list[str]:
        """Atomic string leaves of ``obj`` plus the projector's own ``_stringify`` of it and
        of each value — the exact composed forms (``", ".join`` of a list, ``key=value`` of a
        dict) that the projector can place in an excerpt."""
        out = _values(obj) + [self.p._stringify(obj)]
        if isinstance(obj, dict):
            out += [self.p._stringify(v) for v in obj.values()]
        return out

    def all_text(self, message: dict) -> str:
        """Everything the stored message contains, normalized the way clean() normalizes an
        excerpt. A fragment that occurs here came from this message and nowhere else."""
        parts = [self.p.content_text(message.get("content"))]
        raw = message.get("content")
        if isinstance(raw, str) and raw.lstrip().startswith("{"):
            try:
                parts += self._leaves_and_compositions(json.loads(raw))
            except ValueError:
                pass
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parts.append(fn.get("name") or "")
            try:
                parts += self._leaves_and_compositions(json.loads(fn.get("arguments") or "{}"))
            except (ValueError, TypeError):
                pass
        return self.norm(" ".join(p for p in parts if p))

    def fragments_in(self, excerpt: str, source: str) -> bool:
        """Faithfulness test: every alphanumeric run (>= 2 chars) in the excerpt occurs verbatim
        in the cited message's own text. The projector only ever copies contiguous spans of a
        message and joins them with punctuation it inserts (``:``, ``-``, spaces, repetition),
        so a real excerpt's every word is in the message; a fabricated word would not be. Robust
        to the projector's own composition (list ``, ``-joins, ``detail + body`` duplication,
        ``…`` elision) without reconstructing it."""
        return all(run in source for run in ALNUM_RUN.findall(excerpt))

    def check_log(self, summary: str, messages_by_id: dict[int, dict], label: str) -> None:
        log = self.p.split_sections(summary)[1].get(self.p.LOG_HEADING, "")
        for line in log.splitlines():
            if not line.startswith("[#"):
                continue
            self.lines += 1
            kind, ok = self._check_line(line, messages_by_id)
            if ok:
                self.verified += 1
            elif len(self.failures) < 40:
                ref = (REF.match(line) or [None, "?"])[1] if REF.match(line) else "?"
                self.failures.append(f"{label} [#{ref}] {kind}")  # identifier only, never content

    def _check_line(self, line: str, by_id: dict[int, dict]) -> tuple[str, bool]:
        """Return (line-kind, verified). Verified means role/tool/exit-code match the cited
        message AND every excerpt fragment occurs in that message's own stored text."""
        m = LINE_USER.match(line)
        if m:
            msg = by_id.get(int(m.group(1)))
            if not msg or msg.get("role") != m.group(2):
                return m.group(2), False
            text = self.p.content_text(msg.get("content"))
            if m.group(2) == "assistant":
                text = self.p.strip_think(text)
            return m.group(2), self.fragments_in(m.group(3), self.norm(text))
        m = LINE_CALL.match(line)
        if m:
            msg = by_id.get(int(m.group(1)))
            if not msg or msg.get("role") != "assistant":
                return "call", False
            if not any((tc.get("function") or {}).get("name") == m.group(2) for tc in msg.get("tool_calls") or []):
                return "call", False
            return "call", m.group(3) is None or self.fragments_in(m.group(3), self.all_text(msg))
        m = LINE_RESULT.match(line)
        if m:
            msg = by_id.get(int(m.group(1)))
            if not msg or msg.get("role") != "tool":
                return "result", False
            raw = self.p.content_text(msg.get("content"))
            try:
                data = json.loads(raw) if raw.lstrip().startswith("{") else None
            except ValueError:
                data = None
            code = re.search(r"-?\d+", m.group(3))
            if code and isinstance(data, dict) and isinstance(data.get("exit_code"), int):
                if int(code.group(0)) != data["exit_code"]:
                    return "result", False
            return "result", m.group(4) is None or self.fragments_in(m.group(4), self.all_text(msg))
        return "unparsed", False


def _retention(projector, corpus, budget: int) -> dict:
    """What share of a region's hard facts survive directly into the summary.

    Factory's compression evals found the artifact trail (which files were touched) the
    weakest dimension of every compressor they measured, so it is reported separately here.
    A fact counts as retained only if it appears verbatim in the summary text.
    """
    kinds = ("files_modified", "commands_run", "error_tokens", "user_requests")
    found = {k: 0 for k in kinds}
    total = {k: 0 for k in kinds}
    error_token = re.compile(r"\b([A-Z][A-Za-z]*Error|[A-Z][A-Za-z]*Exception|E\d{3,4}|ENOSPC|EACCES)\b")
    for sid, msgs in corpus:
        summary = projector.project(msgs, budget_chars=budget, session_id=sid)
        facts = {k: set() for k in kinds}
        for m in msgs:
            if m.get("role") == "user":
                text = projector.content_text(m.get("content")).strip()
                words = [w for w in re.findall(r"[A-Za-z0-9]{5,}", text)][:4]
                if words:
                    facts["user_requests"].add(tuple(words))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    continue
                if not isinstance(args, dict):
                    continue
                if fn.get("name") in ("patch", "write_file", "edit_file") and isinstance(args.get("path"), str):
                    facts["files_modified"].add(args["path"])
                if fn.get("name") == "terminal" and isinstance(args.get("command"), str):
                    facts["commands_run"].add(args["command"])
            if m.get("role") == "tool":
                # Only errors from calls that actually FAILED: those are what the engine
                # promises to surface (Blocked / Errors & Fixes). An error name quoted inside
                # a successful command's log is not a failure of this run.
                body = projector.content_text(m.get("content"))
                failed = False
                if body.lstrip().startswith("{"):
                    try:
                        data = json.loads(body)
                        code = data.get("exit_code")
                        failed = (isinstance(code, int) and code != 0) or bool(data.get("error")) \
                            or data.get("success") is False
                    except ValueError:
                        failed = False
                if failed:
                    for tok in error_token.findall(body):
                        facts["error_tokens"].add(tok)
        for kind, items in facts.items():
            for item in items:
                total[kind] += 1
                if kind == "user_requests":
                    hit = sum(1 for w in item if w in summary) >= max(1, len(item) // 2)
                elif kind == "commands_run":
                    hit = item[:24] in summary  # commands may be elided mid-string
                else:
                    hit = item in summary
                found[kind] += int(hit)
    return {k: {"facts": total[k], "retained_in_summary": found[k],
                "rate": round(found[k] / total[k], 3) if total[k] else None} for k in kinds}


def _snapshot(db_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / "state-snapshot.db"
    if snap.exists():
        snap.unlink()
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dst = sqlite3.connect(snap)
    src.backup(dst)
    src.close()
    dst.close()
    os.chmod(snap, 0o600)
    return snap


def run(args) -> int:
    _setup(Path(args.repo), Path(args.hermes_dir))
    import projector

    checker = Checker(projector)

    # 1a. Faithfulness on the frozen corpus (fresh projections of every session).
    corpus = json.loads(Path(args.corpus).read_text())
    for sid, msgs in corpus:
        by_id = {m["_row_id"]: m for m in msgs if isinstance(m.get("_row_id"), int)}
        for budget in (6000, 23592):
            summary = projector.project(msgs, budget_chars=budget, session_id=sid)
            checker.check_log(summary, by_id, f"corpus:{sid[:8]}")
    corpus_lines, corpus_ok = checker.lines, checker.verified

    # 1b + 2. Real summaries the engine wrote into state.db.
    snap = _snapshot(Path(args.state_db), Path(args.workdir))
    db = sqlite3.connect(snap)
    db.row_factory = sqlite3.Row
    summaries = db.execute("SELECT id, session_id, content FROM messages "
                           "WHERE content LIKE '%vcc projection: deterministic%' ORDER BY id").fetchall()

    from hermes_state import SessionDB
    from tools.session_search_tool import session_search

    sdb = SessionDB(snap, read_only=True)
    refs_total = refs_ok = 0
    for row in summaries:
        rows = db.execute("SELECT id, role, content, tool_calls, tool_call_id, tool_name FROM messages "
                          "WHERE session_id = ?", (row["session_id"],)).fetchall()
        by_id = {}
        for r in rows:
            m = {"role": r["role"], "content": r["content"], "_row_id": r["id"]}
            if r["tool_calls"]:
                try:
                    m["tool_calls"] = json.loads(r["tool_calls"])
                except ValueError:
                    pass
            by_id[r["id"]] = m
        for ref in sorted(set(int(x) for x in REF.findall(row["content"]))):
            refs_total += 1
            out = json.loads(session_search(db=sdb, session_id=row["session_id"], around_message_id=ref, window=0))
            found = [m for m in out.get("messages", []) if m.get("id") == ref]
            original = (by_id.get(ref) or {}).get("content") or ""
            if found:
                got = found[0].get("content") or ""
                if got == original or (found[0].get("content_truncated") and original.startswith(got.rstrip("…"))) \
                        or projector.clean(got, 10**9) == projector.clean(original, 10**9):
                    refs_ok += 1
    sdb.close()
    db.close()

    # Negative control: the checker must REJECT a planted fabrication, or a clean pass proves
    # nothing. For every session, splice an invented word into the first verifiable log line
    # and confirm the same check now fails it.
    planted = caught = 0
    for sid, msgs in corpus:
        by_id = {m["_row_id"]: m for m in msgs if isinstance(m.get("_row_id"), int)}
        log = projector.split_sections(projector.project(msgs, budget_chars=23592, session_id=sid))[1]
        for raw_line in log.get(projector.LOG_HEADING, "").splitlines():
            if not (LINE_USER.match(raw_line) or LINE_CALL.match(raw_line) or LINE_RESULT.match(raw_line)):
                continue
            tampered = raw_line + " ZXQVWFABRICATED"  # an invented word in no message
            probe = Checker(projector)
            probe.check_log(f"{projector.LOG_HEADING}\n{tampered}", by_id, "control")
            planted += 1
            caught += int(probe.verified == 0)
            break

    report = {
        "faithfulness_corpus": {"log_lines": corpus_lines, "verified": corpus_ok,
                                "note": "each line checked to quote its cited message's stored text"},
        "negative_control": {"sessions_with_planted_fabrication": planted, "rejected_by_checker": caught,
                             "note": "an invented word spliced into a log line is rejected by the same check"},
        "retention": _retention(projector, corpus, 23592),
        "reference_integrity": {"summaries_in_db": len(summaries), "references": refs_total,
                                "resolved_to_original_message": refs_ok,
                                "note": "every [#N] opened with Hermes's own session_search on a db snapshot"},
        "failures": checker.failures,
    }
    print(json.dumps(report, indent=1))
    Path(args.workdir, "proofs.json").write_text(json.dumps(report, indent=1))
    passed = (corpus_ok == corpus_lines and refs_ok == refs_total and planted > 0 and caught == planted)
    return 0 if passed else 1


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    cache = Path.home() / ".cache" / "hermes-vcc-engine"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-db", default="~/.hermes/state.db")
    ap.add_argument("--hermes-dir", default="~/.hermes/hermes-agent")
    ap.add_argument("--repo", default=str(here))
    ap.add_argument("--corpus", default=str(cache / "corpus.json"))
    ap.add_argument("--workdir", default=str(cache / "proofs"))
    args = ap.parse_args()
    for name in ("state_db", "hermes_dir", "repo", "corpus", "workdir"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    os.chdir(tempfile.gettempdir())
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
