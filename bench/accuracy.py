"""End-to-end accuracy: can the model answer factual questions about compacted work?

For each region of real conversation we build three contexts and ask the same deterministic,
machine-gradable questions under each:

  FULL   the full transcript of the region (the upper bound; no compaction)
  VCC    the vcc projection of the region (what this engine produces)
  STOCK  an LLM summary of the region, generated with Hermes's own summary prompt (what the
         engine replaces)

Questions are generated from the transcript, not by a model, and graded by exact rule (a path,
an exit outcome, an error token, a yes/no with planted absent-fact controls). "No accuracy loss"
means VCC scores at least as high as FULL within noise, and at least as high as STOCK.

Needs the local model at --base-url. Writes bench/accuracy.json. Transcript text stays local;
only scores are printed.

    ~/.hermes/tools/python-*/bin/python3 -I bench/accuracy.py --model <id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ALNUM = re.compile(r"[A-Za-z0-9]{2,}")
ERROR_TOKEN = re.compile(r"\b([A-Z][A-Za-z]*Error|[A-Z][A-Za-z]*Exception|E\d{3,4}|ENOSPC|EACCES)\b")
STOP = set("the a an and or to of in on for with your you this that was were is are be it its from "
           "please can could would make made run ran use used file files code need want".split())


def _setup(repo: Path, hermes_dir: Path) -> None:
    sys.path.insert(0, str(hermes_dir))
    import hermes_bootstrap  # noqa: F401

    sys.path.insert(0, str(repo))


def _text(projector, content) -> str:
    return projector.content_text(content)


def _call(tc):
    fn = tc.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return fn.get("name"), (args if isinstance(args, dict) else {})


# --------------------------------------------------------------------------- region selection

def pick_region(msgs: list[dict], max_chars: int) -> list[dict]:
    """Longest contiguous message window under ``max_chars`` with the most tool activity."""
    def size(a, b):
        return sum(len(json.dumps(m)) for m in msgs[a:b])

    best, best_tools = (0, 0), -1
    a = 0
    for b in range(1, len(msgs) + 1):
        while size(a, b) > max_chars and a < b - 1:
            a += 1
        if size(a, b) <= max_chars:
            tools = sum(1 for m in msgs[a:b] if m.get("tool_calls") or m.get("role") == "tool")
            if tools > best_tools:
                best, best_tools = (a, b), tools
    return msgs[best[0]:best[1]]


# --------------------------------------------------------------------------- probe generation

def make_probes(projector, region: list[dict]) -> list[dict]:
    by_call = {}
    files, commands, errors = {}, [], {}
    first_user = ""
    for m in region:
        if m.get("role") == "user" and not first_user:
            t = projector.content_text(m.get("content")).strip()
            if t:
                first_user = t
        for tc in m.get("tool_calls") or []:
            name, args = _call(tc)
            by_call[tc.get("id")] = (name, args)
            if name in ("patch", "write_file", "edit_file") and isinstance(args.get("path"), str):
                files[os.path.basename(args["path"])] = args["path"]
            if name == "terminal" and isinstance(args.get("command"), str):
                commands.append([tc.get("id"), args["command"]])
        if m.get("role") == "tool":
            name, args = by_call.get(m.get("tool_call_id"), (None, {}))
            body = projector.content_text(m.get("content"))
            for tok in ERROR_TOKEN.findall(body):
                errors[tok] = True

    probes: list[dict] = []
    # File presence (+ two absent controls) and one open recall probe
    for base in list(files)[:3]:
        probes.append({"q": f"Was the file `{base}` created or modified during this work? Answer yes or no.",
                       "grade": "yesno", "expect": "yes"})
    for absent in ("nonexistent_zqxw_module.py", "legacy_vault_adapter.py")[:2 if files else 0]:
        probes.append({"q": f"Was the file `{absent}` created or modified during this work? Answer yes or no.",
                       "grade": "yesno", "expect": "no"})
    if files:
        probes.append({"q": "Name one file that was created or modified during this work. "
                            "Answer with just the file name.", "grade": "any_of", "expect": list(files)})
    # Command outcome
    for cid, cmd in commands[:5]:
        for m in region:
            if m.get("role") == "tool" and m.get("tool_call_id") == cid:
                try:
                    code = json.loads(projector.content_text(m.get("content"))).get("exit_code")
                except (ValueError, TypeError, AttributeError):
                    code = None
                if isinstance(code, int):
                    short = cmd if len(cmd) < 70 else cmd[:67] + "..."
                    probes.append({"q": f"Did the command `{short}` succeed or fail? Answer 'succeed' or 'fail'.",
                                   "grade": "succeed_fail", "expect": "succeed" if code == 0 else "fail"})
                break
    # Error presence (+ absent control)
    for tok in list(errors)[:3]:
        probes.append({"q": f"Did an error or exception named `{tok}` occur during this work? Answer yes or no.",
                       "grade": "yesno", "expect": "yes"})
    if errors:
        probes.append({"q": "Did an error named `ZmxqWidgetError` occur during this work? Answer yes or no.",
                       "grade": "yesno", "expect": "no"})
    # Original request (keyword overlap)
    kw = [w.lower() for w in ALNUM.findall(first_user) if w.lower() not in STOP and len(w) > 3]
    if len(kw) >= 3:
        probes.append({"q": "In one sentence, what did the user originally ask you to do?",
                       "grade": "keywords", "expect": kw[:8]})
    return probes


# --------------------------------------------------------------------------- grading

def _last_of(text: str, words: tuple[str, ...]) -> int:
    """Index of the last standalone occurrence of any word, or -1."""
    best = -1
    for w in words:
        for m in re.finditer(rf"\b{w}\b", text):
            best = max(best, m.start())
    return best


def grade(probe: dict, answer: str) -> bool:
    a = answer.strip().lower()
    if probe["grade"] == "yesno":
        # A reasoning model may deliberate before concluding; the verdict is the LAST yes/no.
        yes, no = _last_of(a, ("yes", "correct", "affirmative")), _last_of(a, ("no", "not", "never", "incorrect"))
        if yes == no == -1:
            return False
        return (yes > no) == (probe["expect"] == "yes")
    if probe["grade"] == "succeed_fail":
        pos, neg = _last_of(a, ("succeed", "success", "succeeded", "passed", "pass")), \
            _last_of(a, ("fail", "failed", "failure", "error", "nonzero"))
        if pos == neg == -1:
            return False
        return (pos > neg) == (probe["expect"] == "succeed")
    if probe["grade"] == "any_of":
        return any(os.path.basename(x).lower() in a for x in probe["expect"])
    if probe["grade"] == "keywords":
        hits = sum(1 for k in probe["expect"] if k in a)
        return hits >= max(2, len(probe["expect"]) // 3)
    return False


# --------------------------------------------------------------------------- model

def ask(base_url: str, model: str, context: str, question: str, max_tokens: int = 48) -> str:
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},  # factual lookup, not a reasoning task
        "messages": [
            {"role": "system", "content": "You answer ONLY from the provided context. If the context does not "
             "contain the answer, reply UNKNOWN. Be terse: a single word or short phrase. /no_think"},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{question} /no_think"},
        ],
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=600))
            msg = d["choices"][0]["message"]
            return (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2)
    return ""


LOOKUP = re.compile(r"LOOKUP\s*#?(\d+)")
RECALL_RULE = ("\n\nThe context is a compacted summary. Each `[#N]` marks a message you may read in "
               "full: to read one, reply with exactly `LOOKUP <N>` and nothing else, and you will be "
               "given that message. Do this when the summary alone does not answer the question.")


def _ask_with_recovery(args, context: str, probe: dict, rendered: dict[int, str]) -> bool:
    """The engine's documented promise: a detail not inlined is still recoverable via [#N].
    One lookup round — ask, and if the model requests a message, supply it and re-ask."""
    first = ask(args.base_url, args.model, context + RECALL_RULE, probe["q"])
    m = LOOKUP.search(first)
    if not m:
        return grade(probe, first)
    body = rendered.get(int(m.group(1)))
    if body is None:
        return grade(probe, first)
    followup = f"{context}\n\nMessage [#{m.group(1)}] in full:\n{body[:3000]}"
    return grade(probe, ask(args.base_url, args.model, followup, probe["q"]))


def render_full(projector, region: list[dict]) -> str:
    out = []
    for i, m in enumerate(region):
        role = m.get("role")
        text = projector.content_text(m.get("content"))
        if role == "user":
            out.append(f"USER: {text}")
        elif role == "assistant":
            if text:
                out.append(f"ASSISTANT: {text}")
            for tc in m.get("tool_calls") or []:
                name, args = _call(tc)
                out.append(f"TOOL CALL {name}: {json.dumps(args)[:600]}")
        elif role == "tool":
            out.append(f"TOOL RESULT: {text[:1500]}")
    return "\n".join(out)


# --------------------------------------------------------------------------- run

def run(args) -> int:
    _setup(Path(args.repo), Path(args.hermes_dir))
    import projector

    corpus = json.loads(Path(args.corpus).read_text())
    corpus.sort(key=lambda s: -sum(len(m.get("content") or "") for m in s[1]))
    conditions = ["FULL", "VCC", "VCC+RECALL"] + (["STOCK"] if not args.no_stock else [])
    totals = {c: [0, 0] for c in conditions}
    per_region = []

    stock_engine = None
    if "STOCK" in conditions:
        import engine
        stock_engine = engine.VccEngine(config={"context": {"vcc": {"fast_anchors": False}}},
                                        config_context_length=args.ctx)
        stock_engine.update_model(model=args.model, context_length=args.ctx, base_url=args.base_url,
                                  api_key=args.api_key, provider="custom")

    regions_done = 0
    for sid, msgs in corpus:
        if regions_done >= args.regions:
            break
        region = pick_region(msgs, args.max_region_chars)
        probes = make_probes(projector, region)
        if len(probes) < args.min_probes:
            continue
        regions_done += 1

        contexts = {"FULL": render_full(projector, region),
                    "VCC": projector.project(region, budget_chars=args.vcc_budget, session_id=sid)}
        contexts["VCC+RECALL"] = contexts["VCC"]
        if "STOCK" in conditions:
            contexts["STOCK"] = _stock_summary(stock_engine, region)

        rendered = {m["_row_id"]: render_full(projector, [m]) for m in region if isinstance(m.get("_row_id"), int)}
        scores = {c: [0, 0] for c in conditions}
        # Condition outer, probes inner: the context prefix stays in llama.cpp's prompt cache,
        # so each condition pays one prefill instead of one per question.
        for c in conditions:
            ctx = contexts[c]
            if ctx is None:
                continue
            for probe in probes:
                if c == "VCC+RECALL":
                    ok = _ask_with_recovery(args, ctx, probe, rendered)
                else:
                    ok = grade(probe, ask(args.base_url, args.model, ctx, probe["q"]))
                scores[c][0] += int(ok)
                scores[c][1] += 1
                totals[c][0] += int(ok)
                totals[c][1] += 1
        per_region.append({"session": sid[:12], "region_messages": len(region), "probes": len(probes),
                           "scores": {c: scores[c] for c in conditions}})
        print(f"[{regions_done}/{args.regions}] {sid[:12]} probes={len(probes)} "
              + " ".join(f"{c}={scores[c][0]}/{scores[c][1]}" for c in conditions), flush=True)

    report = {"model": args.model, "regions": regions_done, "conditions": conditions,
              "overall": {c: {"correct": totals[c][0], "total": totals[c][1],
                              "accuracy": round(totals[c][0] / totals[c][1], 3) if totals[c][1] else None}
                          for c in conditions},
              "per_region": per_region}
    print(json.dumps(report["overall"], indent=1))
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)
    return 0


def _stock_summary(engine_obj, region):
    try:
        engine_obj._vcc_passthrough = True  # force the base ContextCompressor LLM summary path
        out = engine_obj._generate_summary(list(region))
        engine_obj._vcc_passthrough = False
        return out
    except Exception as exc:
        print("  stock summary failed:", type(exc).__name__, exc, flush=True)
        engine_obj._vcc_passthrough = False
        return None


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    cache = Path.home() / ".cache" / "hermes-vcc-engine"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--ctx", type=int, default=98304)
    ap.add_argument("--hermes-dir", default="~/.hermes/hermes-agent")
    ap.add_argument("--repo", default=str(here))
    ap.add_argument("--corpus", default=str(cache / "corpus.json"))
    ap.add_argument("--out", default=str(here / "bench" / "accuracy.json"))
    ap.add_argument("--regions", type=int, default=5)
    ap.add_argument("--min-probes", type=int, default=6)
    ap.add_argument("--max-region-chars", type=int, default=40000)
    ap.add_argument("--vcc-budget", type=int, default=24000)
    ap.add_argument("--no-stock", action="store_true")
    args = ap.parse_args()
    for name in ("hermes_dir", "repo", "corpus", "out"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    os.chdir(os.path.expanduser("~"))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
