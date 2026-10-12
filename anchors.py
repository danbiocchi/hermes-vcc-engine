"""Faster, output-identical replacement for Hermes's anchor index builder.

Hermes appends an "Anchor Index" to every lean compaction summary: identifiers (file paths,
commit hashes, error names, URLs ...) regex-harvested from the whole compacted region. On a
big region that is most of the CPU time of a compaction, because every pattern scans every
character of the region.

Every pattern Hermes uses is line-local (none can match across a newline) and needs a literal
that is cheap to look for: a file match contains an extension like ``.py``, a branch match
contains ``fix/`` and so on. So each pattern only has to look at the lines containing its
literal. Those lines are joined with the same newline separators, so the regex sees the
exact same lines in the same order and produces the exact same matches.

Ranking, caps, noise filtering, the budget and formatting mirror Hermes's
``agent.context_compressor._build_anchor_index`` (MIT License, Copyright (c) 2025 Nous
Research), using Hermes's own patterns and constants imported at runtime. A pattern this
module doesn't know is scanned in full, so an upstream change can cost speed, never output.
On top of that the engine checks this module against Hermes's builder before trusting it
(``verify``) and compares both on the first real compaction of every process.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# How to find the lines that can match each of Hermes's patterns (keyed by exact pattern
# source, so an upstream edit to a pattern just falls back to a full scan):
#   ("literals", (...), also)  every match contains one of these strings (str.find, C speed)
#   ("regex", compiled, also)  same idea, when one regex with a shared literal prefix is faster
#   ("hexrun", 9, None)        every match contains 9 consecutive ASCII hex digits
# ``also`` is a second literal every match must contain (files need a "/").
_PREFILTERS: dict[str, tuple[str, Any, Optional[str]]] = {
    r"#\d{3,6}\b": ("literals", ("#",), None),
    r"\b[0-9a-f]{9,40}\b": ("hexrun", 9, None),
    r"\b(?:fix|feat|docs|refactor|chore|salvage|ent)/[A-Za-z0-9._/-]{3,60}":
        ("literals", ("fix/", "feat/", "docs/", "refactor/", "chore/", "salvage/", "ent/"), None),
    r"\b[\w./-]+/[\w.-]+\.(?:py|ts|tsx|js|rs|md|yaml|yml|json|toml|sh)\b":
        ("regex", re.compile(r"\.(?:py|ts|js|rs|md|yaml|yml|json|toml|sh)"), "/"),
    r"\b(?:[A-Z][a-zA-Z]*Error|Exception|ENOSPC|EACCES|SIGKILL|Traceback)\b[^\n]{0,90}":
        ("literals", ("Error", "Exception", "ENOSPC", "EACCES", "SIGKILL", "Traceback"), None),
    r"@[A-Za-z0-9-]{3,30}\b": ("literals", ("@",), None),
    r"https?://[^\s)\"']{10,110}": ("literals", ("http",), None),
}

# bytes.translate table: ASCII hex digit -> b"h", every other byte -> b"."
_HEX_MASK = bytes(0x68 if chr(b) in "0123456789abcdef" else 0x2E for b in range(256))


def _hermes():
    import agent.context_compressor as cc

    return cc


def _literal_lines(text: str, literals: tuple[str, ...], cap: int) -> Optional[list[str]]:
    spans: dict[int, int] = {}
    find, rfind, size = text.find, text.rfind, len(text)
    for literal in literals:
        i = find(literal)
        while i != -1:
            cap -= 1
            if cap < 0:
                return None
            start = rfind("\n", 0, i) + 1
            end = find("\n", i)
            if end == -1:
                spans[start] = size
                break
            spans[start] = end
            i = find(literal, end + 1)
    return [text[s:spans[s]] for s in sorted(spans)]


def _regex_lines(text: str, finder: Any, cap: int) -> Optional[list[str]]:
    lines: list[str] = []
    find, rfind, search, size = text.find, text.rfind, finder.search, len(text)
    m = search(text)
    while m:
        cap -= 1
        if cap < 0:
            return None
        i = m.start()
        start = rfind("\n", 0, i) + 1
        end = find("\n", i)
        if end == -1:
            lines.append(text[start:])
            break
        lines.append(text[start:end])
        m = search(text, end + 1)
    return lines


def _hexrun_lines(text: str, run: int, cap: int) -> Optional[list[str]]:
    # Work on UTF-8 bytes: an ASCII hex digit is the same single byte there, every other
    # character is bytes >= 0x80 or a non-hex ASCII byte, and "\n" is one byte, so line
    # boundaries and hex runs are found exactly; lone surrogates round-trip via surrogatepass.
    data = text.encode("utf-8", "surrogatepass")
    mask = data.translate(_HEX_MASK)
    needle = b"h" * run
    lines: list[str] = []
    find, rfind, size = data.find, data.rfind, len(data)
    i = mask.find(needle)
    while i != -1:
        cap -= 1
        if cap < 0:
            return None
        start = rfind(b"\n", 0, i) + 1
        end = find(b"\n", i)
        if end == -1:
            lines.append(data[start:].decode("utf-8", "surrogatepass"))
            break
        lines.append(data[start:end].decode("utf-8", "surrogatepass"))
        i = mask.find(needle, end + 1)
    return lines


def _candidate_text(text: str, n_lines: int, pattern_src: str) -> str:
    """The lines of ``text`` that can contain a match, joined exactly as they are in ``text``.

    Lines without the required literal cannot match, so leaving them out changes nothing.
    When more than half the lines qualify, scanning the whole text is cheaper and, being the
    original input, trivially gives the same result.
    """
    spec = _PREFILTERS.get(pattern_src)
    if spec is None:
        return text
    kind, arg, also = spec
    cap = n_lines // 2
    if kind == "literals":
        lines = _literal_lines(text, arg, cap)
    elif kind == "regex":
        lines = _regex_lines(text, arg, cap)
    else:
        lines = _hexrun_lines(text, arg, cap)
    if lines is None:
        return text
    if also is not None:
        lines = [line for line in lines if also in line]
    return "\n".join(lines)


def build(turns: list[dict[str, Any]], cc: Any = None) -> str:
    """Same string as Hermes's ``_build_anchor_index(turns)``."""
    cc = cc or _hermes()
    text = "\n".join(c for c in (msg.get("content") for msg in turns) if isinstance(c, str) and c)
    if not text:
        return ""
    n_lines = text.count("\n") + 1

    sections: list[str] = []
    used = 0
    noise = cc._ANCHOR_NOISE
    budget = cc._LEAN_ANCHOR_BUDGET_CHARS
    for label, pattern, cap in cc._ANCHOR_PATTERNS:
        counts: dict[str, int] = {}
        last_seen: dict[str, int] = {}
        for n, m in enumerate(pattern.finditer(_candidate_text(text, n_lines, pattern.pattern))):
            val = m.group(0).strip().rstrip(".,;:")
            if val.lower() in noise:
                continue
            counts[val] = counts.get(val, 0) + 1
            last_seen[val] = n
        if not counts:
            continue
        ranked = sorted(counts, key=lambda v: (-counts[v], -last_seen[v]))[:cap]
        line = f"{label}: " + ", ".join(f"{v}(x{counts[v]})" if counts[v] > 1 else v for v in ranked)
        if used + len(line) > budget:
            break
        sections.append(line)
        used += len(line)
    if not sections:
        return ""
    return "\n\n" + cc._LEAN_ANCHOR_HEADING + "\n" + "\n".join(sections) + _footer(cc)


_FOOTER: Optional[str] = None


def _footer(cc: Any) -> str:
    """Hermes's closing note, taken from Hermes's own output so its wording can never drift."""
    global _FOOTER
    if _FOOTER is None:
        probe = "fix/anchor-probe"
        reference = cc._build_anchor_index([{"content": probe}])
        label = next(lbl for lbl, pat, _ in cc._ANCHOR_PATTERNS if pat.search(probe))
        head = "\n\n" + cc._LEAN_ANCHOR_HEADING + "\n" + f"{label}: {probe}"
        if not reference.startswith(head):
            raise RuntimeError("Hermes anchor index layout changed")
        _FOOTER = reference[len(head):]
    return _FOOTER


def _fixture() -> list[dict[str, Any]]:
    """Edge cases for every pattern: noise, ties, counts, caps, budget overflow, punctuation."""
    rows = [
        "PR #1234 and #99 and #123456 then #1234 again; also abc#4567",
        "commits deadbeef01 0123456789abcdef0123456789abcdef01234567 DEADBEEF01 xdeadbeef012 deadbeefcafe0123.",
        "branches fix/login-bug feat/new-ui, docs/readme. refactor/x chore/deps salvage/old-work ent/thing prefix/fix/x",
        "files ./src/app.py src/app.py, /home/u/proj/README.md tests/test_x.py: cfg/settings.yaml a/b.json x.py",
        "ValueError: bad input\nTraceback (most recent call last):\n  File x, line 1\nKeyError: 'k'\nENOSPC no space",
        "@someone @teknium @teknium1 @ab @a-b-c-d-e and mail me@example.com",
        "see https://example.com/a/very/long/path?q=1, http://short.io/abcdefghij) and 'https://quoted.example/x'",
        None, 42, "", "\n\n",
    ]
    turns: list[dict[str, Any]] = [{"content": r} for r in rows]
    turns += [{"content": f"src/module_{i}.py raised Error{i}Error: boom {i} #{1000 + i}"} for i in range(160)]
    turns += [{"content": "https://example.com/" + "p" * 60 + str(i)} for i in range(50)]
    return turns


def verify(cc: Any = None) -> bool:
    """True when ``build`` reproduces Hermes's builder on the built-in edge-case fixture."""
    cc = cc or _hermes()
    try:
        turns = _fixture()
        return build(turns, cc) == cc._build_anchor_index(turns) and build([], cc) == cc._build_anchor_index([])
    except Exception as exc:  # any drift in Hermes internals means "don't use the fast path"
        logger.debug("vcc: anchor self-check failed: %s", exc)
        return False
