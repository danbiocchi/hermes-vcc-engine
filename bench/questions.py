"""Independent question generator for the context-compaction benchmark.

This module deliberately shares no code with the compressor or the old probe
generator. It reads only the raw OpenAI-style messages that were compacted
(tool-call arguments, tool-result JSON / exit codes, user text) and turns
facts found there into questions with exact, rule-based answers.

Public API:
    make_questions(turns, seed=0) -> list[dict]
    paraphrase(question) -> dict
    grade(question, answer_text) -> bool

Stdlib only, deterministic for a given (turns, seed).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

MAX_QUESTIONS = 14

# Tool names whose calls change a file named by a "path"-like argument.
WRITE_TOOLS = {"write_file", "patch", "edit_file", "edit", "str_replace",
               "create_file", "apply_patch", "replace_in_file", "append_file"}
READ_TOOLS = {"read_file", "view_file", "open_file", "cat_file"}
TERMINAL_TOOLS = {"terminal", "shell", "bash", "run_command", "exec"}
PATH_KEYS = ("path", "file_path", "filepath", "file", "filename", "target_file")

# Fill order for the round-robin. Quotas keep any one family from dominating.
FAMILY_ORDER = ["file_changed_yes", "file_read_only_no", "command_outcome",
                "user_request", "error_seen", "last_state", "command_ran",
                "count"]
QUOTA = {"file_changed_yes": 2, "file_read_only_no": 2, "command_outcome": 3,
         "user_request": 2, "error_seen": 2, "last_state": 1,
         "command_ran": 1, "count": 1}

ERRNO_NAMES = {"ENOENT", "EACCES", "EPERM", "EEXIST", "EADDRINUSE",
               "ECONNREFUSED", "ECONNRESET", "EISDIR", "ENOTDIR", "ENOTEMPTY",
               "EPIPE", "EAGAIN", "ETIMEDOUT", "ENOSPC", "EMFILE", "EBUSY",
               "EINVAL", "EHOSTUNREACH", "ENETUNREACH"}
ERROR_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*?(?:Error|Exception))\b")
ERRNO_RE = re.compile(r"\b(E[A-Z]{3,12})\b")

STOPWORDS = set("""
about above after again against along already also although always among
another anything anyway around because become been before being below
between both cannot could couldn doesn doing during each either else
enough every everything first from further getting going gonna great
having here hello however instead into itself just know later least
let's maybe might more most much must myself need needs never next
nothing now often once only other others otherwise ought our ours
ourselves over please pretty probably quite rather really right same
seems should shouldn since some something sometimes still such sure
than thank thanks that that's their theirs them themselves then there
these they thing things think this those though through thus today
together too under until upon used using very want wants wanna was
wasn well were weren what whatever when where whether which while who
whole whom whose why will with within without won't would wouldn yeah
your yours yourself yourselves actually basically again going doing
make makes making take takes look looks looking like likes check okay
again first second third start stuff thing works working work worked
right left gonna isn't aren't didn't don't can't hasn't haven't
""".split())

# Plausible decoy vocabulary for user_request. A decoy is used only if it
# appears nowhere in the turns (case-insensitive substring check).
DECOYS = """
lantern harbor violet saddle quartz meadow pepper falcon walnut canvas
glacier tundra orchid ember marble cobalt thimble juniper basil copper
velvet anchor beacon crimson dolphin feather granite hazel ivory jasmine
kettle lemonade mango nectar olive pebble quiver raven sapphire timber
umbrella vortex willow yellow zephyr almond blossom candle dagger eagle
fossil garnet hammock iceberg jigsaw kayak lobster magnet noodle oyster
parrot quilt ribbon sparrow tulip unicorn vapor walrus barley cactus
desert engine fabric gravel honey island jacket koala ladder mirror
needle orbit pillow radish salmon tablet valley wagon zebra spreadsheet
invoice payroll mortgage calendar podcast newsletter warehouse shipment
voucher passport recipe garden laundry vacation airport stadium museum
""".split()


# ---------------------------------------------------------------------------
# Parsing helpers


def _loads(s):
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return None


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal parts
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict))
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _args(tc) -> dict:
    fn = tc.get("function") or {}
    a = _loads(fn.get("arguments"))
    return a if isinstance(a, dict) else {}


def _path_of(args: dict):
    for k in PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return os.path.normpath(v.strip())
    return None


def _exit_code(result_text: str):
    """Exit code from a terminal result, or None if not explicit."""
    j = _loads(result_text)
    if isinstance(j, dict) and isinstance(j.get("exit_code"), int) \
            and not isinstance(j.get("exit_code"), bool):
        return j["exit_code"]
    m = list(re.finditer(r'"exit_code"\s*:\s*(-?\d+)', result_text))
    if m:
        return int(m[-1].group(1))
    m = list(re.finditer(r"->\s*exit(?:\s+code)?\s+(-?\d+)\b", result_text))
    if m:
        return int(m[-1].group(1))
    return None


def _is_background(args: dict, result_text: str) -> bool:
    if args.get("background") in (True, "true", 1):
        return True
    j = _loads(result_text)
    return isinstance(j, dict) and ("pid" in j or "session_id" in j)


def _write_failed(result_text: str) -> bool:
    j = _loads(result_text)
    if isinstance(j, dict):
        if j.get("success") is False or j.get("error"):
            return True
        if j.get("success") is True:
            return False
    return bool(re.search(r"(?i)\b(error|failed|failure|could not|not found|"
                          r"no match)\b", result_text[:400]))


def _short_cmd(cmd: str):
    """The command as the agent wrote it, minus a leading `cd X &&`, one line,
    whitespace collapsed, truncated at a word boundary."""
    c = cmd.strip()
    c = re.sub(r"^cd\s+\S+\s*(&&|;)\s*", "", c)
    c = c.splitlines()[0].strip() if c else ""
    c = re.sub(r"\s+", " ", c)
    if not c or "`" in c:
        return None
    if len(c) > 70:
        cut = c[:70]
        sp = cut.rfind(" ")
        c = (cut[:sp] if sp > 30 else cut) + " ..."
    return c


def _machine_user(text: str) -> bool:
    t = text.lstrip()
    if not t or t.startswith("[") or t.startswith("<"):
        return True
    up = text.upper()
    return any(k in up for k in ("CONTEXT COMPACTION", "SUMMARY",
                                 "SYSTEM REMINDER", "TOOL OUTPUT"))


def _hid(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Fact extraction


class _Facts:
    def __init__(self, turns):
        self.calls = []          # (index, name, args, result_text or None)
        self.user_texts = []
        self.all_text_lower = []
        results = {}
        for m in turns:
            if m.get("role") == "tool" and m.get("tool_call_id"):
                results[m["tool_call_id"]] = _text(m.get("content"))
        for i, m in enumerate(turns):
            role = m.get("role")
            content = _text(m.get("content"))
            self.all_text_lower.append(content.lower())
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                self.all_text_lower.append(str(fn.get("arguments") or "").lower())
                self.calls.append((i, fn.get("name") or "", _args(tc),
                                   results.get(tc.get("id"))))
            if role == "user" and not _machine_user(content):
                self.user_texts.append(content)
        self.blob = "\n".join(self.all_text_lower)
        self.tool_results = [r for (_, _, _, r) in self.calls if r]


# ---------------------------------------------------------------------------
# Candidate builders (one list per family)


def _q(family, target, q, grade, expect, **extra):
    d = {"id": f"{family}-{_hid(family, target)}", "family": family, "q": q,
         "grade": grade, "expect": expect}
    d.update(extra)
    return d


def _file_candidates(f: _Facts):
    paths_by_base = defaultdict(set)
    written, write_ok, read = set(), set(), set()
    for _, name, args, res in f.calls:
        p = _path_of(args)
        if not p:
            continue
        if name in WRITE_TOOLS or name in READ_TOOLS or name == "search_files":
            paths_by_base[os.path.basename(p)].add(p)
        if name in WRITE_TOOLS:
            written.add(p)
            if res is None or not _write_failed(res):
                write_ok.add(p)
        elif name in READ_TOOLS:
            read.add(p)
    written_bases = {os.path.basename(p) for p in written}

    # Text where a file could be changed by other means (shell, code, outputs).
    other_text = []
    for _, name, args, res in f.calls:
        if name in READ_TOOLS or name == "search_files":
            continue
        other_text.append(json.dumps(args).lower())
        if res and name not in WRITE_TOOLS:
            other_text.append(res.lower())
    other_blob = "\n".join(other_text)

    yes, no = [], []
    for p in sorted(write_ok):
        b = os.path.basename(p)
        if not b or len(paths_by_base[b]) != 1 or "`" in b:
            continue
        yes.append(_q("file_changed_yes", p,
                      f"Was the file `{b}` modified or created in this part of "
                      f"the session? Answer yes or no.", "yesno", "yes",
                      subject=b))
    for p in sorted(read - written):
        b = os.path.basename(p)
        if (not b or "`" in b or len(paths_by_base[b]) != 1
                or b in written_bases or b.lower() in other_blob):
            continue
        no.append(_q("file_read_only_no", p,
                     f"Was the file `{b}` modified or created in this part of "
                     f"the session? Answer yes or no.", "yesno", "no",
                     subject=b))
    return yes, no


def _terminal_runs(f: _Facts):
    runs = []  # (display, exit_code or None, background)
    for _, name, args, res in f.calls:
        if name not in TERMINAL_TOOLS:
            continue
        cmd = args.get("command") or args.get("cmd")
        if not isinstance(cmd, str):
            continue
        disp = _short_cmd(cmd)
        if not disp:
            continue
        code = _exit_code(res) if res else None
        runs.append((disp, code, _is_background(args, res or "")))
    return runs


def _command_candidates(f: _Facts, rng):
    runs = _terminal_runs(f)
    last = None
    if runs and runs[-1][1] is not None and not runs[-1][2]:
        last = runs[-1]
    last_q = []
    if last:
        disp, code, _ = last
        last_q.append(_q("last_state", disp,
                         f"Did the last terminal command in this part of the "
                         f"session (`{disp}`) succeed (exit code 0)? Answer yes "
                         f"or no.", "yesno", "yes" if code == 0 else "no",
                         subject=disp))

    by_disp = defaultdict(list)
    for disp, code, bg in runs:
        by_disp[disp].append((code, bg))
    ok, bad = [], []
    for disp, rs in sorted(by_disp.items()):
        if last and disp == last[0]:
            continue
        known = [c for c, bg in rs if c is not None and not bg]
        if not known:
            continue
        outcomes = {c == 0 for c in known}
        final_known = rs[-1][0] is not None and not rs[-1][1]
        if len(outcomes) == 1 and len(known) == len(rs):
            succ = known[0] == 0
            q = (f"Did the command `{disp}` succeed (exit code 0)? "
                 f"Answer yes or no.")
        elif final_known:
            succ = rs[-1][0] == 0
            q = (f"The last time the command `{disp}` was run, did it succeed "
                 f"(exit code 0)? Answer yes or no.")
        else:
            continue  # ambiguous
        (ok if succ else bad).append(
            _q("command_outcome", disp, q, "yesno", "yes" if succ else "no",
               subject=disp))
    rng.shuffle(ok)
    rng.shuffle(bad)
    # Interleave failures first so the mix is as balanced as the data allows.
    outcome = []
    for i in range(max(len(ok), len(bad))):
        if i < len(bad):
            outcome.append(bad[i])
        if i < len(ok):
            outcome.append(ok[i])

    asked = {q["subject"] for q in outcome[:QUOTA["command_outcome"]]}
    if last:
        asked.add(last[0])
    ran = [_q("command_ran", d,
              f"Was the command `{d}` run? Answer yes or no.", "yesno", "yes",
              subject=d)
           for d in sorted(by_disp) if d not in asked]
    rng.shuffle(ran)
    return outcome, ran, last_q


def _error_candidates(f: _Facts, rng):
    names = Counter()
    for r in f.tool_results:
        found = set(ERROR_RE.findall(r))
        found |= {e for e in ERRNO_RE.findall(r) if e in ERRNO_NAMES}
        names.update(found)
    out = [_q("error_seen", n,
              f"Did the error `{n}` appear in any tool output in this part of "
              f"the session? Answer yes or no.", "yesno", "yes", subject=n)
           for n in sorted(names) if n not in ("Error", "Exception")]
    rng.shuffle(out)
    return out


def _count_candidates(f: _Facts, rng):
    c = Counter(name for _, name, _, _ in f.calls if name)
    out = [_q("count", t,
              f"How many times was the `{t}` tool called in this part of the "
              f"session? Answer with a number.", "number", n, subject=t)
           for t, n in sorted(c.items()) if 1 <= n <= 4]
    rng.shuffle(out)
    return out


_WORD_STRIP = ".,!?;:\"'()[]{}*"


def _user_candidates(f: _Facts, rng):
    word_freq = Counter(re.findall(r"[a-z]+", f.blob))
    decoys = [d for d in DECOYS if d not in f.blob]
    cands = {}
    for text in f.user_texts:
        for tok in text.split():
            w = tok.strip(_WORD_STRIP)
            if not re.fullmatch(r"[A-Za-z]{5,20}", w):
                continue
            lw = w.lower()
            if lw in STOPWORDS or lw in DECOYS:
                continue
            cands.setdefault(lw, word_freq[lw])
    # Distinctive = rare across the whole segment.
    words = sorted(cands, key=lambda w: (cands[w], w))[:12]
    rng.shuffle(words)
    out = []
    for w in words:
        if len(decoys) < 2:
            break
        ds = rng.sample(decoys, 2)
        for d in ds:
            decoys.remove(d)
        opts = [w] + ds
        rng.shuffle(opts)
        out.append(_q("user_request", w,
                      f"Which of these words appeared in the user's request: "
                      f"{', '.join(opts)}? Answer with the word.",
                      "exact_any", [w], decoys=ds, options=opts))
    return out


# ---------------------------------------------------------------------------
# Public API


def make_questions(turns, seed=0, max_questions=MAX_QUESTIONS):
    f = _Facts(turns or [])
    fp = hashlib.sha256(json.dumps(
        [(_text(m.get("content"))[:200], m.get("role")) for m in turns or []],
        sort_keys=True).encode()).hexdigest()
    rng = random.Random(f"{seed}:{fp}")

    yes, no = _file_candidates(f)
    rng.shuffle(yes)
    rng.shuffle(no)
    outcome, ran, last_q = _command_candidates(f, rng)
    pools = {
        "file_changed_yes": yes,
        "file_read_only_no": no,
        "command_outcome": outcome,
        "user_request": _user_candidates(f, rng),
        "error_seen": _error_candidates(f, rng),
        "last_state": last_q,
        "command_ran": ran,
        "count": _count_candidates(f, rng),
    }
    taken = Counter()
    out = []

    def take(fam, idx=0):
        out.append(pools[fam].pop(idx))
        taken[fam] += 1

    # Pass 1: round-robin within per-family quotas.
    progress = True
    while progress and len(out) < max_questions:
        progress = False
        for fam in FAMILY_ORDER:
            if len(out) < max_questions and pools[fam] \
                    and taken[fam] < QUOTA[fam]:
                take(fam)
                progress = True

    # Pass 2: top up, steering the yes/no balance toward even and allowing
    # at most double the quota per family.
    def balance():
        return sum(1 if q["expect"] == "yes" else -1
                   for q in out if q["grade"] == "yesno")

    while len(out) < max_questions:
        open_fams = [f for f in FAMILY_ORDER
                     if pools[f] and f not in ("last_state", "count")
                     and taken[f] < 2 * QUOTA[f]]
        if not open_fams:
            break
        pick = None
        if balance() > 0:
            for f in open_fams:
                for i, q in enumerate(pools[f]):
                    if q["grade"] != "yesno" or q["expect"] == "no":
                        pick = (f, i)
                        break
                if pick:
                    break
        if pick is None:
            if balance() > 0:
                break  # only "yes" left; don't deepen the imbalance
            f = min(open_fams, key=lambda f: (taken[f] / QUOTA[f],
                                              FAMILY_ORDER.index(f)))
            pick = (f, 0)
        take(*pick)
    return out


_PARA = {
    "file_changed_yes": "Over these turns, did anyone edit, overwrite or newly "
                        "write `{s}` to disk? Respond yes/no.",
    "file_read_only_no": "Over these turns, did anyone edit, overwrite or "
                         "newly write `{s}` to disk? Respond yes/no.",
    "command_outcome": "Running `{s}`: did it finish cleanly with a zero "
                       "status? Respond yes/no.",
    "command_ran": "Is there any record of the agent executing `{s}`? "
                   "Respond yes/no.",
    "error_seen": "Was `{s}` ever reported by a tool result here? "
                  "Respond yes/no.",
    "last_state": "Final shell invocation here, `{s}`: did it end cleanly "
                  "with status zero? Respond yes/no.",
    "count": "Give the total of `{s}` invocations as an integer.",
}


def paraphrase(question: dict) -> dict:
    """Second wording that shares as few words as possible with the first."""
    p = dict(question)
    p["id"] = question["id"] + "-p"
    fam = question["family"]
    if fam == "user_request":
        p["q"] = (f"The person typed one of these terms: "
                  f"{', '.join(question['options'])}. Reply using that term.")
    elif fam == "command_outcome" and question["q"].startswith("The last time"):
        p["q"] = (f"Most recent execution of `{question['subject']}`: did it "
                  f"finish cleanly with a zero status? Respond yes/no.")
    else:
        p["q"] = _PARA[fam].format(s=question["subject"])
    return p


# ---------------------------------------------------------------------------
# Grading

_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
              "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "once": 1, "twice": 2}


def _strip_thinking(text: str) -> str:
    text = text or ""
    text = re.sub(r"(?is)<(think|thinking|reasoning)>.*?</\1>", " ", text)
    # Unclosed block or a bare closing tag: keep what comes after the last close.
    m = list(re.finditer(r"(?i)</(think|thinking|reasoning)>", text))
    if m:
        text = text[m[-1].end():]
    text = re.sub(r"(?is)<(think|thinking|reasoning)>.*", " ", text)
    return text.strip()


def grade(question: dict, answer_text: str) -> bool:
    a = _strip_thinking(answer_text)
    low = a.lower()
    if low.startswith("unknown"):
        return False
    g = question["grade"]
    if g == "yesno":
        m = re.search(r"\b(yes|no|unknown)\b", low)
        return bool(m) and m.group(1) == str(question["expect"]).lower()
    if re.search(r"\bunknown\b", low):
        return False
    if g == "exact_any":
        def has(w):
            return re.search(r"(?<![a-z0-9])" + re.escape(w.lower())
                             + r"(?![a-z0-9])", low) is not None
        if any(has(d) for d in question.get("decoys", [])):
            return False  # listing every option is not an answer
        return any(has(e) for e in question["expect"])
    if g == "number":
        m = re.search(r"-?\d+|\b(" + "|".join(_NUM_WORDS) + r")\b", low)
        if not m:
            return False
        tok = m.group(0)
        val = _NUM_WORDS[tok] if tok in _NUM_WORDS else int(tok)
        return val == int(question["expect"])
    raise ValueError(f"unknown grade type {g!r}")
