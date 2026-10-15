"""Tests for bench/questions.py using made-up messages only."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from questions import grade, make_questions, paraphrase  # noqa: E402

_n = 0


def call(name, **args):
    global _n
    _n += 1
    cid = f"c{_n}"
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}
    return cid, msg


def result(cid, name, content):
    if not isinstance(content, str):
        content = json.dumps(content)
    return {"role": "tool", "tool_call_id": cid, "tool_name": name,
            "content": content}


def turns():
    out = [{"role": "user", "content": "Please refactor the zigzag parser "
            "and make the frobnicate step faster."}]
    c, m = call("read_file", path="/w/src/parser.py")
    out += [m, result(c, "read_file", "1|def parse(): pass")]
    c, m = call("read_file", path="/w/docs/notes.txt")
    out += [m, result(c, "read_file", "1|notes")]
    c, m = call("patch", path="/w/src/parser.py", old_string="a", new_string="b")
    out += [m, result(c, "patch", {"success": True})]
    c, m = call("write_file", path="/w/src/fast.py", content="x = 1")
    out += [m, result(c, "write_file", "[write_file] wrote /w/src/fast.py (1 line)")]
    c, m = call("terminal", command="cd /w && pytest -q tests")
    out += [m, result(c, "terminal", {"output": "E   KeyError: 'k'\n1 failed",
                                      "exit_code": 1, "error": None})]
    c, m = call("terminal", command="ls -la build")
    out += [m, result(c, "terminal", "[terminal] ran `ls -la build` -> exit 0, 3 lines output")]
    c, m = call("terminal", command="python -m build")
    out += [m, result(c, "terminal", {"output": "ok", "exit_code": 0, "error": None})]
    out.append({"role": "user", "content": "[System note: CONTEXT COMPACTION happened]"})
    return out


def by_family(qs):
    d = {}
    for q in qs:
        d.setdefault(q["family"], []).append(q)
    return d


def test_families_and_facts():
    qs = make_questions(turns())
    fam = by_family(qs)
    changed = {q["subject"] for q in fam["file_changed_yes"]}
    assert changed == {"parser.py", "fast.py"}
    assert [q["subject"] for q in fam["file_read_only_no"]] == ["notes.txt"]
    assert all(q["expect"] == "no" for q in fam["file_read_only_no"])
    outcomes = {q["subject"]: q["expect"] for q in fam["command_outcome"]}
    assert outcomes == {"pytest -q tests": "no", "ls -la build": "yes"}
    assert fam["last_state"][0]["subject"] == "python -m build"
    assert fam["last_state"][0]["expect"] == "yes"
    assert [q["subject"] for q in fam["error_seen"]] == ["KeyError"]
    ur = fam["user_request"]
    assert ur and all(q["expect"][0] in ("zigzag", "parser", "refactor",
                                         "frobnicate", "faster", "step")
                      for q in ur)
    for q in ur:
        for d in q["decoys"]:
            assert d not in json.dumps(turns()).lower()


def test_deterministic_and_capped():
    a = make_questions(turns(), seed=3)
    assert a == make_questions(turns(), seed=3)
    assert len(a) <= 14
    assert len({q["id"] for q in a}) == len(a)


def test_ambiguous_basename_and_mixed_outcomes_skipped():
    t = []
    c, m = call("write_file", path="/a/config.py", content="")
    t += [m, result(c, "write_file", "ok")]
    c, m = call("read_file", path="/b/config.py")
    t += [m, result(c, "read_file", "x")]
    for code in (1, 0):
        c, m = call("terminal", command="make test")
        t += [m, result(c, "terminal", {"output": "", "exit_code": code})]
    c, m = call("terminal", command="echo done")
    t += [m, result(c, "terminal", {"output": "done", "exit_code": 0})]
    qs = make_questions(t)
    assert not any(q.get("subject") == "config.py" for q in qs)
    mt = [q for q in qs if q["family"] == "command_outcome"]
    assert len(mt) == 1 and mt[0]["q"].startswith("The last time")
    assert mt[0]["expect"] == "yes"


def test_file_touched_by_shell_is_not_read_only():
    t = []
    c, m = call("read_file", path="/w/a.cfg")
    t += [m, result(c, "read_file", "x")]
    c, m = call("terminal", command="sed -i s/x/y/ /w/a.cfg")
    t += [m, result(c, "terminal", {"output": "", "exit_code": 0})]
    assert not any(q["family"] == "file_read_only_no" for q in make_questions(t))


def test_grade():
    yn = {"grade": "yesno", "expect": "no"}
    assert grade(yn, "<think>yes, maybe</think>No.")
    assert not grade(yn, "Yes")
    assert not grade(yn, "UNKNOWN")
    assert not grade(yn, "unknown, but probably no")
    ex = {"grade": "exact_any", "expect": ["zigzag"], "decoys": ["walnut", "quartz"]}
    assert grade(ex, "Zigzag")
    assert not grade(ex, "zigzagged")
    assert not grade(ex, "zigzag, walnut, quartz")
    assert not grade(ex, "UNKNOWN")
    num = {"grade": "number", "expect": 3}
    assert grade(num, "It was called 3 times.")
    assert grade(num, "three")
    assert not grade(num, "13")
    assert not grade(num, "UNKNOWN")


def test_paraphrase_shares_few_words():
    for q in make_questions(turns()):
        p = paraphrase(q)
        assert p["expect"] == q["expect"] and p["grade"] == q["grade"]
        assert p["id"] != q["id"]
        assert grade(p, q["expect"] if q["grade"] == "yesno"
                     else str(q["expect"][0] if q["grade"] == "exact_any"
                              else q["expect"]))
        a = set(q["q"].lower().split())
        b = set(p["q"].lower().split())
        assert len(a & b) / len(a) < 0.5, (q["family"], a & b)
