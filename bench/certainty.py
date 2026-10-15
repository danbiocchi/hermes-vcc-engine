"""Round-2 head-to-head: stock Hermes vs vcc 0.1 / 0.2 / 0.3, as the model really sees it.

Differences from headtohead.py, each closing a gap a reviewer would point at:

- Real context. Questions are asked as a new user turn appended to the actual message list
  that Hermes's compress() returns (head, summary, anchor index, verbatim user messages, tail),
  repaired and sanitized by Hermes's own helpers, not to a bare summary string.
- Two question sets. bench/accuracy.py's generator, plus bench/questions.py, written without
  reading the projector, so the questions cannot have been chosen to match what vcc keeps.
- Held-out data. Windows are cut at several context sizes; windows from sessions that were not
  used while developing 0.3 are flagged held_out.
- End-to-end time. Besides the summary step, the first request after compaction is timed with
  the prompt cache off, since that re-read happens with either engine.

Run with Hermes's interpreter; every stage is resumable and writes only to
~/.cache/hermes-vcc-engine/certainty (mode 0600). Only ids, scores and timings are printed.

    $PY -I bench/certainty.py freeze <ctx> [<ctx> ...]       # CPU
    $PY -I bench/certainty.py compact STOCK                  # GPU, slow
    VCC_REPO=<checkout> $PY -I bench/certainty.py compact <name>
    $PY -I bench/certainty.py prefill <cond> [<cond> ...]    # GPU
    $PY -I bench/certainty.py ask <cond> [<cond> ...]        # GPU; FULL = uncompacted window
    $PY -I bench/certainty.py report
"""
import hashlib, json, math, os, random, re, sqlite3, sys, time, urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VCC_REPO", str(BENCH.parent))).resolve()
HERMES = Path(os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent")))
BASE_URL = os.environ.get("VCC_BASE_URL", "http://localhost:8080/v1")
HERE = Path.home() / ".cache" / "hermes-vcc-engine" / "certainty"
HERE.mkdir(mode=0o700, parents=True, exist_ok=True)
SYSTEM = {"role": "system", "content": "You are Hermes."}
DEV_SESSIONS_FILE = Path.home() / ".cache" / "hermes-vcc-engine" / "headtohead" / "events.json"
RULE = ("Answer from this conversation only. If it does not tell you, reply UNKNOWN. "
        "Be terse: one word or a short phrase.")

sys.path.insert(0, str(HERMES)); import hermes_bootstrap  # noqa: E402,F401
sys.path.insert(0, str(REPO)); sys.path.append(str(BENCH))
os.chdir(Path.home())
import engine, projector, accuracy, questions  # noqa: E402
from agent.model_metadata import estimate_messages_tokens_rough as est_msgs, estimate_tokens_rough as est  # noqa: E402
from agent.agent_runtime_helpers import repair_message_sequence, sanitize_api_messages  # noqa: E402


def load(name, default=None):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else default


def save(name, obj):
    p = HERE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj)); os.chmod(tmp, 0o600); tmp.replace(p)


def model_id():
    return json.load(urllib.request.urlopen(BASE_URL + "/models"))["data"][0]["id"]


def make_engine(sid, ctx, stock, model=""):
    eng = engine.VccEngine(config={"context": {"vcc": {"fallback_to_llm": False}}}, config_context_length=ctx)
    eng.update_model(model=model or "bench/local", context_length=ctx, base_url=BASE_URL, api_key="ollama",
                     provider="custom")
    eng._session_id = sid
    eng._vcc_passthrough = stock
    cap = {}
    inner = eng._generate_summary

    def wrapped(turns, *a, **k):
        cap["turns"] = list(turns)
        t = time.perf_counter()
        cap["summary"] = inner(turns, *a, **k)
        cap["summary_s"] = time.perf_counter() - t
        return cap["summary"]
    eng._generate_summary = wrapped
    return eng, cap


# ------------------------------------------------------------------ freeze

def load_sessions():
    db = sqlite3.connect(f"file:{Path.home()}/.hermes/state.db?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    ids = [r[0] for r in db.execute(
        "SELECT session_id FROM messages GROUP BY session_id HAVING count(*) >= 12 "
        "ORDER BY sum(length(coalesce(content,''))+length(coalesce(tool_calls,''))) DESC")]
    out = []
    for sid in ids:
        msgs = []
        for r in db.execute("SELECT id, role, content, tool_calls, tool_call_id, tool_name FROM messages "
                            "WHERE session_id = ? ORDER BY id", (sid,)):
            m = {"role": r["role"], "content": r["content"], "_row_id": r["id"]}
            if r["tool_calls"]:
                try: m["tool_calls"] = json.loads(r["tool_calls"])
                except ValueError: pass
            if r["tool_call_id"]: m["tool_call_id"] = r["tool_call_id"]
            if r["tool_name"]: m["tool_name"] = r["tool_name"]
            msgs.append(m)
        out.append((sid, msgs))
    db.close()
    return out


def windows(msgs, threshold):
    """Consecutive windows, each the size at which Hermes would compact."""
    start = 0
    while start < len(msgs):
        end = next((i for i in range(start + 1, len(msgs) + 1) if est_msgs([SYSTEM] + msgs[start:i]) >= threshold), None)
        if end is None:
            return
        while end < len(msgs) and msgs[end].get("role") == "tool":  # never cut a call from its results
            end += 1
        yield start, end
        start = end
        while start < len(msgs) and msgs[start].get("role") == "tool":
            start += 1


def freeze(ctxs):
    events = load("events.json", [])
    have = {e["id"] for e in events}
    dev = {e["sid"] for e in json.loads(DEV_SESSIONS_FILE.read_text())} if DEV_SESSIONS_FILE.exists() else set()
    sessions = load_sessions()
    for ctx in ctxs:
        for sid, msgs in sessions:
            eng0, _ = make_engine(sid, ctx, stock=False)
            for wi, (a, b) in enumerate(windows(msgs, eng0.threshold_tokens)):
                label = f"c{ctx // 1024}k-{hashlib.sha256(sid.encode()).hexdigest()[:6]}-w{wi}"
                if label in have:
                    continue
                convo = [SYSTEM, *msgs[a:b]]
                eng, cap = make_engine(sid, ctx, stock=False)
                eng.compress(json.loads(json.dumps(convo)), force=True)
                if eng._vcc_fallbacks or not cap.get("turns"):
                    continue
                turns = cap["turns"]
                # The generator's "keywords" question is dropped: its expected words come partly from
                # Hermes's own banner text, and it scored 0-1 of 26 in every context in round 1.
                qs = [{**p, "src": "gen", "family": p["grade"] + ("-no" if p.get("expect") == "no" else "")}
                      for p in accuracy.make_probes(projector, turns) if p["grade"] != "keywords"]
                qs += [{**q, "src": "ind"} for q in questions.make_questions(turns, seed=wi)]
                events.append({"id": label, "sid": sid, "ctx": ctx, "convo": convo, "n_turns": len(turns),
                               "questions": qs, "tokens_before": est_msgs(convo),
                               "threshold": eng0.threshold_tokens, "held_out": sid not in dev})
                have.add(label)
                print(f"{label}: msgs {b - a}, tokens {est_msgs(convo):,}, turns {len(turns)}, questions {len(qs)}"
                      f"{' held-out' if sid not in dev else ''}", flush=True)
                save("events.json", events)
    print(f"{len(events)} events, {sum(len(e['questions']) for e in events)} questions, "
          f"{sum(e['held_out'] for e in events)} held-out")


# ------------------------------------------------------------------ compaction

def compact(name):
    events, store = load("events.json"), load("compacted.json", {})
    stock = name == "STOCK"
    model = model_id() if stock else ""
    done = store.setdefault(name, {"repo": None if stock else str(REPO), "events": {}})
    for e in events:
        if e["id"] in done["events"]:
            continue
        runs, after, cap, err = [], None, {}, None
        for _ in range(1 if stock else 5):  # vcc is milliseconds: median of 5 fresh engines
            eng, cap = make_engine(e["sid"], e["ctx"], stock=stock, model=model)
            t = time.perf_counter()
            try:
                after = eng.compress(json.loads(json.dumps(e["convo"])), force=True)
            except Exception as exc:
                after, err = None, f"{type(exc).__name__}"
            runs.append(time.perf_counter() - t)
        s = cap.get("summary") or ""
        done["events"][e["id"]] = {
            "after": after, "summary_sha": hashlib.sha256(s.encode()).hexdigest()[:16],
            "compress_s": sorted(runs)[len(runs) // 2], "summary_s": cap.get("summary_s"),
            "tokens_after": est_msgs(after) if after else None, "summary_tokens": est(s), "error": err,
            "same_turns": len(cap.get("turns") or []) == e["n_turns"],
            "fallbacks": 0 if stock else eng._vcc_fallbacks}
        save("compacted.json", store)
        print(f"{e['id']}: {name} {done['events'][e['id']]['compress_s']:8.3f}s "
              f"tokens_after {done['events'][e['id']]['tokens_after']} same_turns "
              f"{done['events'][e['id']]['same_turns']} {err or ''}", flush=True)
    print(f"compact {name} done", flush=True)


# ------------------------------------------------------------------ requests

def api_messages(msgs):
    """What Hermes would send: Hermes's own repair and sanitize passes, private keys dropped."""
    msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in json.loads(json.dumps(msgs))]
    repair_message_sequence(None, msgs)
    return sanitize_api_messages(msgs)


def chat(model, msgs, max_tokens, cache=True):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": cache,
                       "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}).encode()
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            t = time.perf_counter()
            d = json.load(urllib.request.urlopen(req, timeout=3600))
            msg = d["choices"][0]["message"]
            return (msg.get("content") or "") + " " + (msg.get("reasoning_content") or ""), \
                time.perf_counter() - t, d.get("usage", {}), d.get("timings", {})
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)


def context_for(cond, e, comp):
    if cond == "FULL":
        return e["convo"]
    c = comp.get(cond, {}).get("events", {}).get(e["id"])
    return c["after"] if c and c.get("after") else None


def ask_messages(ctx_msgs, q):
    msgs = api_messages(ctx_msgs)
    if msgs and msgs[-1].get("role") == "user":  # keep alternation: fold the question into it
        msgs[-1] = {**msgs[-1], "content": f"{msgs[-1].get('content') or ''}\n\n{RULE}\n{q}"}
    else:
        msgs.append({"role": "user", "content": f"{RULE}\n{q}"})
    return msgs


def prefill(conds):
    events, comp, out, model = load("events.json"), load("compacted.json", {}), load("prefill.json", {}), model_id()
    for e in events:
        for cond in conds:
            key = f"{e['id']}|{cond}"
            ctx = context_for(cond, e, comp)
            if key in out or ctx is None:
                continue
            _, wall, usage, timings = chat(model, ask_messages(ctx, "Reply OK."), 1, cache=False)
            out[key] = {"wall_s": wall, "prompt_tokens": usage.get("prompt_tokens"),
                        "prompt_ms": timings.get("prompt_ms")}
            save("prefill.json", out)
            print(f"{key}: first request after compaction {wall:6.1f}s, prompt {usage.get('prompt_tokens')} tok",
                  flush=True)
    print("prefill done", flush=True)


def ask(conds):
    events, comp, answers, model = load("events.json"), load("compacted.json", {}), load("answers.json", {}), model_id()
    for e in events:
        for cond in conds:
            ctx = context_for(cond, e, comp)
            if ctx is None:
                print(f"{e['id']} {cond}: no context, skipped", flush=True)
                continue
            score = {}
            ctx_sha = hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()[:16]
            for wording in ("orig", "para"):  # condition outer, questions inner: prefix stays cached
                for i, q in enumerate(e["questions"]):
                    key = f"{e['id']}|{cond}|{wording}|{i}"
                    if key not in answers:
                        qq = q if wording == "orig" else paraphrase(q)
                        a, wall, usage, tm = chat(model, ask_messages(ctx, qq["q"]), 64)
                        # prompt_ms / cache_n of the first question in a condition is the cost of
                        # re-reading the compacted prompt, which the next real turn would pay too.
                        answers[key] = {"answer": a.strip()[:300], "ok": bool(grade(qq, a)), "s": round(wall, 2),
                                        "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                                        "prompt_ms": tm.get("prompt_ms"), "ctx_sha": ctx_sha}
                        save("answers.json", answers)
                    score[wording] = score.get(wording, 0) + answers[key]["ok"]
            n = len(e["questions"])
            print(f"{e['id']} {cond:8} orig {score.get('orig', 0):2}/{n}  para {score.get('para', 0):2}/{n}", flush=True)
    print("ask done", flush=True)


def paraphrase(q):
    """Second wording, fixed before any results (headtohead.py's for the generator's questions)."""
    if q["src"] == "ind":
        return questions.paraphrase(q)
    text, g = q["q"], q["grade"]
    m = re.search(r"`([^`]+)`", text)
    term = m.group(1) if m else ""
    if g == "yesno" and text.startswith("Was the file"):
        return {**q, "q": f"Did `{term}` get changed at any point during this work? Answer yes or no."}
    if g == "yesno" and "error" in text:
        return {**q, "q": f"Was `{term}` raised or reported at any point? Answer yes or no."}
    if g == "succeed_fail":
        return {**q, "q": f"Did `{term}` run without errors? Answer yes or no.", "grade": "yesno",
                "expect": "yes" if q["expect"] == "succeed" else "no"}
    if g == "any_of":
        return {**q, "q": "Give the name of one file the agent changed. Just the file name."}
    return q


def grade(q, a):
    return questions.grade(q, a) if q["src"] == "ind" else accuracy.grade(q, a)


# ------------------------------------------------------------------ report

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def report():
    events, answers, comp, pre = load("events.json"), load("answers.json", {}), load("compacted.json", {}), \
        load("prefill.json", {})
    conds = sorted({k.split("|")[1] for k in answers}, key=lambda c: (c != "FULL", c != "STOCK", c))
    by_id = {e["id"]: e for e in events}

    def rows(cond, keep=lambda e, q: True):
        out = []
        for k, v in answers.items():
            eid, c, w, i = k.split("|")
            if c == cond and eid in by_id and keep(by_id[eid], by_id[eid]["questions"][int(i)]):
                out.append((eid, w, int(i), v["ok"]))
        return out

    slices = {
        "all": lambda e, q: True,
        "no-planted-controls": lambda e, q: not q["family"].endswith("-no"),
        "held-out": lambda e, q: e["held_out"],
        "dev": lambda e, q: not e["held_out"],
        "independent-qs": lambda e, q: q["src"] == "ind",
        "generator-qs": lambda e, q: q["src"] == "gen",
    }
    res = {"accuracy": {}, "families": {}, "pairs": {}, "timing": {}, "identity": {}}
    for sname, keep in slices.items():
        for c in conds:
            r = rows(c, keep)
            res["accuracy"][f"{sname}|{c}"] = [sum(x[3] for x in r), len(r)]
    fams = sorted({(q["src"], q["family"]) for e in events for q in e["questions"]})
    for src, fam in fams:
        for c in conds:
            r = rows(c, lambda e, q, s=src, f=fam: q["src"] == s and q["family"] == f)
            res["families"][f"{src}:{fam}|{c}"] = [sum(x[3] for x in r), len(r)]

    rng = random.Random(0)
    for sname in ("all", "held-out", "independent-qs"):
        keep = slices[sname]
        for ai, a in enumerate(conds):
            for b in conds[ai + 1:]:
                ra = {(x[0], x[1], x[2]): x[3] for x in rows(a, keep)}
                rb = {(x[0], x[1], x[2]): x[3] for x in rows(b, keep)}
                common = sorted(set(ra) & set(rb))
                if not common:
                    continue
                a_only = sum(ra[k] and not rb[k] for k in common)
                b_only = sum(rb[k] and not ra[k] for k in common)
                per_ev = {}
                for k in common:
                    per_ev.setdefault(k[0], []).append(ra[k] - rb[k])
                diffs = [sum(v) for v in per_ev.values()]
                obs = sum(diffs)
                # paired sign-flip permutation over windows (questions in a window are not independent)
                perm = sum(abs(sum(d if rng.random() < .5 else -d for d in diffs)) >= abs(obs) for _ in range(20000))
                boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(common) for _ in range(5000))
                res["pairs"][f"{sname}|{a} vs {b}"] = {
                    "n": len(common), "windows": len(diffs), "a_only": a_only, "b_only": b_only,
                    "diff_pts": round(100 * obs / len(common), 1),
                    "ci95_pts": [round(100 * boots[124], 1), round(100 * boots[4874], 1)],
                    "p_mcnemar": round(mcnemar(a_only, b_only), 4), "p_window_perm": round((perm + 1) / 20001, 4)}

    for name, v in comp.items():
        ev = v["events"]
        res["timing"][name] = {
            "compress_s": sorted(x["compress_s"] for x in ev.values()),
            "tokens_after": [x["tokens_after"] for x in ev.values()],
            "errors": sum(bool(x["error"]) for x in ev.values()),
            "same_turns": sum(x["same_turns"] for x in ev.values()), "n": len(ev)}
    for name in comp:
        for other in comp:
            if name < other:
                common = set(comp[name]["events"]) & set(comp[other]["events"])
                res["identity"][f"{name} == {other}"] = [sum(
                    comp[name]["events"][k]["summary_sha"] == comp[other]["events"][k]["summary_sha"] for k in common),
                    len(common)]
    res["prefill"] = pre
    save("report.json", res)
    for k, (ok, n) in res["accuracy"].items():
        print(f"{k:28} {ok:5}/{n:<5} {100 * ok / n if n else 0:5.1f}%")
    for k, v in res["pairs"].items():
        print(f"{k:36} n={v['n']:5} w={v['windows']:3} {v['a_only']:4}/{v['b_only']:<4} diff {v['diff_pts']:+5.1f} pts "
              f"CI {v['ci95_pts']} p_mc={v['p_mcnemar']} p_win={v['p_window_perm']}")
    for k, v in res["identity"].items():
        print(f"identical summaries {k}: {v[0]}/{v[1]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "freeze": freeze([int(x) for x in sys.argv[2:]])
    elif cmd == "compact": compact(sys.argv[2])
    elif cmd == "prefill": prefill(sys.argv[2:])
    elif cmd == "ask": ask(sys.argv[2:])
    elif cmd == "report": report()
