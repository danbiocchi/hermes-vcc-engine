"""Builders for realistic Hermes transcripts (OpenAI chat format, as Hermes keeps them)."""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

_ids = itertools.count(1000)


def user(text: str, row: int | None = None) -> dict[str, Any]:
    return {"role": "user", "content": text, "_row_id": row or next(_ids)}


def assistant(text: str = "", calls: list[dict[str, Any]] | None = None, row: int | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": text, "_row_id": row or next(_ids)}
    if calls:
        msg["tool_calls"] = calls
    return msg


def call(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def result(call_id: str, payload: Any, name: str | None = None, row: int | None = None) -> dict[str, Any]:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    msg = {"role": "tool", "tool_call_id": call_id, "content": content, "_row_id": row or next(_ids)}
    if name:
        msg["tool_name"] = name
    return msg


def terminal_turn(call_id: str, command: str, output: str, exit_code: int = 0) -> list[dict[str, Any]]:
    return [
        assistant("", [call(call_id, "terminal", command=command)]),
        result(call_id, {"output": output, "exit_code": exit_code, "error": None}, "terminal"),
    ]


def fix_the_build() -> list[dict[str, Any]]:
    """User asks for a fix; test fails; file patched; test passes."""
    return [
        user("The build in ~/proj is red, please fix it and keep the public API unchanged."),
        assistant("<think>Start with the tests.</think>Running the suite first.",
                  [call("t1", "terminal", command="cd ~/proj && pytest -q")]),
        result("t1", {"output": "E   ImportError: cannot import name 'parse' from 'proj.core'\n1 failed",
                      "exit_code": 1, "error": None}, "terminal"),
        assistant("", [call("r1", "read_file", path="/home/u/proj/proj/core.py")]),
        result("r1", "[read_file] read /home/u/proj/proj/core.py from line 1 (3,660 chars)", "read_file"),
        assistant("", [call("p1", "patch", path="/home/u/proj/proj/core.py",
                            old_string="def _parse(", new_string="def parse(")]),
        result("p1", {"success": True, "diff": "-def _parse(\n+def parse(",
                      "files_modified": ["/home/u/proj/proj/core.py"]}, "patch"),
        *terminal_turn("t2", "cd ~/proj && pytest -q", "48 passed in 1.2s"),
        assistant("Fixed: `parse` was renamed to `_parse` in core.py. All 48 tests pass."),
    ]


def long_session(n_rounds: int, output_chars: int = 3000, seed: int = 7) -> list[dict[str, Any]]:
    """A tool-heavy session of roughly ``n_rounds * output_chars`` chars."""
    rng = random.Random(seed)
    msgs: list[dict[str, Any]] = [user("Audit the repo and clean up the logging module.")]
    for i in range(n_rounds):
        kind = rng.choice(["terminal", "read_file", "patch", "search_files"])
        cid = f"c{i}"
        if kind == "terminal":
            failed = rng.random() < 0.15
            msgs += terminal_turn(cid, f"python -m pytest tests/test_{i % 9}.py -q",
                                  ("Traceback (most recent call last):\n" if failed else "")
                                  + "x" * output_chars, 1 if failed else 0)
        elif kind == "read_file":
            msgs += [assistant("", [call(cid, "read_file", path=f"/repo/src/mod_{i % 13}.py")]),
                     result(cid, "line\n" * (output_chars // 5), "read_file")]
        elif kind == "patch":
            msgs += [assistant("Tightening the handler.", [call(cid, "patch", path=f"/repo/src/mod_{i % 13}.py",
                                                                old_string="a", new_string="b")]),
                     result(cid, {"success": True, "files_modified": [f"/repo/src/mod_{i % 13}.py"]}, "patch")]
        else:
            msgs += [assistant("", [call(cid, "search_files", pattern="logger", path="/repo/src")]),
                     result(cid, {"total_count": 3, "files": ["a.py", "b.py", "c.py"]}, "search_files")]
        if i % 10 == 9:
            msgs.append(user(f"Looks good so far, continue with batch {i // 10 + 1}."))
    return msgs
