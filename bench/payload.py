"""Questions about what tool results said, for the payload-excerpt comparison (BACKLOG item 10).

``accuracy.py`` and ``questions.py`` ask what was done (files, commands, outcomes, errors, the
request), never what a result contained. This module covers that gap with facts that only a
tool result holds, each graded by exact rule:

* ``search_count``: the ``total_count`` a ``search_files`` call reported;
* ``file_lines``: the ``total_lines`` a ``read_file`` call reported;
* ``web_domain``: a site a ``web_search`` returned;
* ``clarify_answer``: what the user answered to a question the agent asked with ``clarify``.

It reads only the raw messages (Hermes's untrusted-content envelope removed by the same rule
Hermes uses to add it), shares no code with the projector, and is deterministic.
"""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

QUOTA = 2  # per family per window
_ENVELOPE = re.compile(r'\A<untrusted_tool_result source="[^"\n]*">\n[^\n]*\n\n(.*)\n</untrusted_tool_result>',
                       re.DOTALL)
_WORD = re.compile(r"[a-z0-9]{4,}")
_STOP = {"that", "this", "with", "have", "from", "your", "will", "just", "what", "when", "then", "them",
         "they", "into", "also", "only", "like", "dont", "please", "yeah", "sure", "http", "https"}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def _json(content) -> dict | None:
    text = _text(content)
    m = _ENVELOPE.match(text)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _args(tc) -> tuple[str, dict]:
    fn = tc.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return fn.get("name") or "", args if isinstance(args, dict) else {}


def _short(s: str, n: int = 70) -> str:
    s = " ".join(str(s).split()).replace("`", "'")
    return s if len(s) <= n else s[: n - 3] + "..."


def make_probes(turns: list[dict]) -> list[dict]:
    calls: dict[str, tuple[str, dict]] = {}
    out: dict[str, list[dict]] = {}

    def add(family: str, probe: dict) -> None:
        bucket = out.setdefault(family, [])
        if len(bucket) < QUOTA and all(p["q"] != probe["q"] for p in bucket):
            bucket.append({**probe, "family": family})

    for m in turns:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                calls[tc.get("id")] = _args(tc)
        if m.get("role") != "tool":
            continue
        name, args = calls.get(m.get("tool_call_id"), (m.get("tool_name") or "", {}))
        data = _json(m.get("content"))
        if not data or data.get("error") or data.get("success") is False:
            continue
        if name == "search_files" and isinstance(data.get("total_count"), int) and args.get("pattern"):
            add("search_count", {"q": f"How many results did the file search for `{_short(args['pattern'])}` report? "
                                      "Answer with just the number.", "grade": "number", "expect": data["total_count"]})
        elif name == "read_file" and isinstance(data.get("total_lines"), int) and data["total_lines"] > 0 \
                and isinstance(args.get("path"), str):
            add("file_lines", {"q": f"How many lines long is the file `{os.path.basename(args['path'])}`, as reported "
                                    "when it was read? Answer with just the number.",
                               "grade": "number", "expect": data["total_lines"]})
        elif name == "web_search" and isinstance(args.get("query"), str):
            hits = ((data.get("data") or {}).get("web") or []) if isinstance(data.get("data"), dict) else []
            domains = sorted({urlparse(h.get("url") or "").netloc.removeprefix("www.") for h in hits
                              if isinstance(h, dict)} - {""})
            if domains:
                add("web_domain", {"q": f"Name one website (domain) that the web search for `{_short(args['query'])}` "
                                        "returned. Just the domain.", "grade": "any_of", "expect": domains})
        elif name == "clarify" and isinstance(data.get("responses"), list):
            for r in data["responses"]:
                answer = r.get("user_response") if isinstance(r, dict) else None
                if not isinstance(answer, str) or not r.get("question"):
                    continue
                # Words that tell this answer apart from the other choices offered: naming any one
                # of them identifies it ("Telegram" for "Telegram: connect it to Hermes ...").
                others = {w for c in r.get("choices_offered") or [] if isinstance(c, str) and c.strip() != answer.strip()
                          for w in _WORD.findall(c.lower())}
                words = [w for w in dict.fromkeys(_WORD.findall(answer.lower())) if w not in _STOP and w not in others]
                if words:
                    add("clarify_answer", {"q": f"When you asked the user \"{_short(r['question'], 120)}\", what did "
                                                "they answer? Be brief.", "grade": "any_word", "expect": words[:6]})
    return [p for fam in sorted(out) for p in out[fam]]


def paraphrase(q: dict) -> dict:
    """Second wording, fixed before any results."""
    text = q["q"]
    if q["family"] == "search_count":
        return {**q, "q": text.replace("How many results did the file search for", "What result count came back from searching files for")}
    if q["family"] == "file_lines":
        return {**q, "q": text.replace("How many lines long is the file", "What line count was reported for the file")}
    if q["family"] == "web_domain":
        return {**q, "q": text.replace("Name one website (domain) that the web search for", "Give one site that came up in the web search for")}
    if q["family"] == "clarify_answer":
        return {**q, "q": text.replace("what did they answer? Be brief.", "what was their reply? Be brief.")}
    return q


def grade(q: dict, answer: str) -> bool:
    a = answer.lower()
    if q["grade"] == "number":
        return str(q["expect"]) in re.findall(r"\d+", a.replace(",", ""))
    if q["grade"] == "any_of":
        return any(d.lower() in a for d in q["expect"])
    if q["grade"] == "any_word":
        return any(re.search(rf"\b{re.escape(w)}", a) for w in q["expect"])
    return False
