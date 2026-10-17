"""Head-to-head: stock Hermes compaction vs vcc, on real compaction points from your state.db.

Each session is cut into the windows at which Hermes would actually compact it. Each window is
compacted once by stock Hermes (its own LLM summary, on your model) and by one or more vcc
checkouts, then the same machine-graded questions (bench/accuracy.py) are asked against each
result, in their generated wording and in a fixed paraphrase that shares no wording with the
summaries. Paired results are compared with McNemar's exact test.

Run with Hermes's interpreter. Every stage is resumable. Everything is written to
~/.cache/hermes-vcc-engine/headtohead at mode 0600; only ids, scores and timings are printed.

    PY=~/.hermes/tools/python-*/bin/python3
    $PY -I bench/headtohead.py freeze                       # CPU: windows + questions
    $PY -I bench/headtohead.py stock                        # GPU: stock summaries (slow)
    VCC_REPO=<checkout> $PY -I bench/headtohead.py variant <name> [earlier_share]
    $PY -I bench/headtohead.py ask FULL STOCK <name> ...    # GPU: answers
    $PY -I bench/headtohead.py report

FULL is the uncompacted window, STOCK the stock summary, anything else a variant name.
VCC_BASE_URL (default http://localhost:8080/v1) and VCC_CONTEXT (default 98304) set the model
endpoint and the context length the trigger is computed from.
"""
import dataclasses, hashlib, json, math, os, re, sqlite3, sys, time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = Path(os.environ.get("VCC_REPO", str(BENCH.parent))).resolve()
HERMES = Path(os.environ.get("HERMES_AGENT_DIR", str(Path.home() / ".hermes" / "hermes-agent")))
BASE_URL = os.environ.get("VCC_BASE_URL", "http://localhost:8080/v1")
CTX = int(os.environ.get("VCC_CONTEXT", "98304"))
HERE = Path.home() / ".cache" / "hermes-vcc-engine" / "headtohead"
HERE.mkdir(mode=0o700, parents=True, exist_ok=True)
SYSTEM = {"role": "system", "content": "You are Hermes."}
MIN_PROBES = 5

sys.path.insert(0, str(HERMES)); import hermes_bootstrap  # noqa: E402,F401
sys.path.insert(0, str(REPO / "vcc" if (REPO / "vcc" / "engine.py").is_file() else REPO)); sys.path.append(str(BENCH))
os.chdir(Path.home())
import engine, projector, accuracy  # noqa: E402
from agent.model_metadata import estimate_messages_tokens_rough as est_msgs, estimate_tokens_rough as est  # noqa: E402


def load(name, default=None):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else default


def save(name, obj):
    p = HERE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj)); os.chmod(tmp, 0o600); tmp.replace(p)


def model_id():
    import urllib.request
    return json.load(urllib.request.urlopen(BASE_URL + "/models"))["data"][0]["id"]


def make_engine(sid, stock, model=""):
    eng = engine.VccEngine(config={"context": {"vcc": {"fallback_to_llm": False}}}, config_context_length=CTX)
    eng.update_model(model=model or "bench/local", context_length=CTX, base_url=BASE_URL, api_key="ollama",
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


def strip_banner(s):
    return engine.ContextCompressor._strip_summary_prefix(s) if s else s


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


def freeze():
    if load("events.json"):
        sys.exit("events.json exists; delete it to refreeze")
    events = []
    for si, (sid, msgs) in enumerate(load_sessions()):
        eng0, _ = make_engine(sid, stock=False)
        for wi, (a, b) in enumerate(windows(msgs, eng0.threshold_tokens)):
            convo = [SYSTEM, *msgs[a:b]]
            eng, cap = make_engine(sid, stock=False)
            eng.compress(json.loads(json.dumps(convo)), force=True)
            if eng._vcc_fallbacks or not cap.get("summary"):
                continue
            probes = accuracy.make_probes(projector, cap["turns"])
            label = f"s{si}w{wi}"
            print(f"{label}: msgs {b - a}, tokens {est_msgs(convo):,}, turns {len(cap['turns'])}, probes {len(probes)}",
                  flush=True)
            if len(probes) >= MIN_PROBES:
                events.append({"id": label, "sid": sid, "convo": convo, "turns": cap["turns"], "probes": probes,
                               "tokens_before": est_msgs(convo), "threshold": eng0.threshold_tokens})
    save("events.json", events)
    print(f"{len(events)} events, {sum(len(e['probes']) for e in events)} probes")


# ------------------------------------------------------------------ summaries

def stock():
    events, done, model = load("events.json"), load("stock.json", {}), model_id()
    for e in events:
        if e["id"] in done:
            continue
        eng, cap = make_engine(e["sid"], stock=True, model=model)
        t = time.perf_counter()
        try:
            after, err = eng.compress(json.loads(json.dumps(e["convo"])), force=True), None
        except Exception as exc:
            after, err = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        wall = time.perf_counter() - t
        done[e["id"]] = {"summary": cap.get("summary"), "compress_s": wall, "summary_s": cap.get("summary_s"),
                         "tokens_after": est_msgs(after) if after else None,
                         "summary_tokens": est(cap.get("summary") or ""), "error": err,
                         "same_turns": len(cap.get("turns") or []) == len(e["turns"])}
        save("stock.json", done)
        print(f"{e['id']}: stock {wall:7.1f}s ok={bool(cap.get('summary'))} same_turns={done[e['id']]['same_turns']} "
              f"{err or ''}", flush=True)
    print("stock done", flush=True)


def variant(name, earlier_share=None):
    if earlier_share is not None:
        orig = projector.project
        projector.project = lambda *a, limits=None, **k: orig(
            *a, limits=dataclasses.replace(limits or projector.Limits(), earlier_share=earlier_share), **k)
    store = load("variants.json", {})
    store[name] = {"repo": str(REPO), "earlier_share": earlier_share, "events": {}}
    for e in load("events.json"):
        runs = []
        for _ in range(5):  # median of 5 fresh engines: these are milliseconds
            eng, cap = make_engine(e["sid"], stock=False)
            convo = json.loads(json.dumps(e["convo"]))
            t = time.perf_counter()
            after = eng.compress(convo, force=True)
            runs.append(time.perf_counter() - t)
        s = cap.get("summary") or ""
        store[name]["events"][e["id"]] = {"summary": s, "compress_s": sorted(runs)[2], "tokens_after": est_msgs(after),
                                          "summary_tokens": est(s), "fallbacks": eng._vcc_fallbacks,
                                          "sha": hashlib.sha256(s.encode()).hexdigest()[:16]}
    save("variants.json", store)
    v = store[name]["events"].values()
    print(f"{name}: {len(v)} events, fallbacks {sum(x['fallbacks'] for x in v)}, "
          f"median compress total {sum(x['compress_s'] for x in v):.3f}s, avg summary {sum(x['summary_tokens'] for x in v) / len(v):,.0f} tok")


# ------------------------------------------------------------------ asking

def paraphrase(p):
    """Fixed before any results: shares wording with neither the generated questions nor the summaries."""
    q, g = p["q"], p["grade"]
    m = re.search(r"`([^`]+)`", q)
    term = m.group(1) if m else ""
    if g == "yesno" and q.startswith("Was the file"):
        return {**p, "q": f"Did `{term}` get changed at any point during this work? Answer yes or no."}
    if g == "yesno" and "error" in q:
        return {**p, "q": f"Was `{term}` raised or reported at any point? Answer yes or no."}
    if g == "succeed_fail":
        return {**p, "q": f"Did `{term}` run without errors? Answer yes or no.", "grade": "yesno",
                "expect": "yes" if p["expect"] == "succeed" else "no"}
    if g == "any_of":
        return {**p, "q": "Give the name of one file the agent changed. Just the file name."}
    if g == "keywords":
        return {**p, "q": "Summarize the user's first request in one sentence."}
    return p


def context(cond, e, stock_d, variants):
    if cond == "FULL":
        return accuracy.render_full(projector, e["turns"])
    if cond == "STOCK":
        return strip_banner((stock_d.get(e["id"]) or {}).get("summary"))
    return strip_banner(variants[cond]["events"][e["id"]]["summary"])


def ask(conds):
    events, stock_d, variants = load("events.json"), load("stock.json", {}), load("variants.json", {})
    answers, model = load("answers.json", {}), model_id()
    for e in events:
        for cond in conds:
            ctx = context(cond, e, stock_d, variants)
            if not ctx:
                print(f"{e['id']} {cond}: no context yet, skipped", flush=True)
                continue
            score = {"orig": 0, "para": 0}
            for pset in ("orig", "para"):
                for i, p in enumerate(e["probes"]):
                    key = f"{e['id']}|{cond}|{pset}|{i}"
                    if key not in answers:
                        probe = p if pset == "orig" else paraphrase(p)
                        a = accuracy.ask(BASE_URL, model, ctx, probe["q"])
                        answers[key] = {"answer": a.strip()[:300], "ok": bool(accuracy.grade(probe, a)),
                                        "kind": p["grade"]}
                        save("answers.json", answers)
                    score[pset] += answers[key]["ok"]
            n = len(e["probes"])
            print(f"{e['id']} {cond:14} orig {score['orig']:2}/{n}  para {score['para']:2}/{n}", flush=True)
    print("ask done", flush=True)


# ------------------------------------------------------------------ report

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def report():
    events, answers = load("events.json"), load("answers.json", {})
    stock_d, variants = load("stock.json", {}), load("variants.json", {})
    conds = sorted({k.split("|")[1] for k in answers}, key=lambda c: (c != "FULL", c != "STOCK", c))
    out = {"conditions": {}, "pairs": {}, "kinds": {}, "timing": {}, "unknown": {}}
    for pset in ("orig", "para", "both"):
        for c in conds:
            ks = [k for k in answers if k.split("|")[1] == c and (pset == "both" or k.split("|")[2] == pset)]
            out["conditions"][f"{c}|{pset}"] = [sum(answers[k]["ok"] for k in ks), len(ks)]
            out["unknown"][f"{c}|{pset}"] = sum("UNKNOWN" in answers[k]["answer"].upper() for k in ks)
    probes = {e["id"]: e["probes"] for e in events}

    def kind(key):  # yes/no questions split by the expected answer: "no" is a different skill
        eid, _, _, i = key.split("|")
        p = probes[eid][int(i)]
        return f"yesno-{p['expect']}" if p["grade"] == "yesno" else p["grade"]
    for c in conds:
        for kd in ("yesno-yes", "yesno-no", "succeed_fail", "any_of", "keywords"):
            ks = [k for k in answers if k.split("|")[1] == c and kind(k) == kd]
            out["kinds"][f"{c}|{kd}"] = [sum(answers[k]["ok"] for k in ks), len(ks)]
    for a in conds:
        for b in conds:
            if a >= b:
                continue
            common = {k.split("|", 1)[0] + "|" + k.split("|", 2)[2] for k in answers if k.split("|")[1] == a} & \
                     {k.split("|", 1)[0] + "|" + k.split("|", 2)[2] for k in answers if k.split("|")[1] == b}
            ab = sum(answers[f"{x.split('|')[0]}|{a}|{x.split('|', 1)[1]}"]["ok"] and not
                     answers[f"{x.split('|')[0]}|{b}|{x.split('|', 1)[1]}"]["ok"] for x in common)
            ba = sum(answers[f"{x.split('|')[0]}|{b}|{x.split('|', 1)[1]}"]["ok"] and not
                     answers[f"{x.split('|')[0]}|{a}|{x.split('|', 1)[1]}"]["ok"] for x in common)
            out["pairs"][f"{a} vs {b}"] = {"n": len(common), "a_only": ab, "b_only": ba, "p": round(mcnemar(ab, ba), 4)}
    out["timing"]["STOCK"] = {k: v["compress_s"] for k, v in stock_d.items()}
    for name, v in variants.items():
        out["timing"][name] = {k: x["compress_s"] for k, x in v["events"].items()}
    out["events"] = [{"id": e["id"], "messages": len(e["convo"]) - 1, "tokens_before": e["tokens_before"],
                      "probes": len(e["probes"])} for e in events]
    out["tokens_after"] = {"STOCK": {k: v["tokens_after"] for k, v in stock_d.items()},
                           **{n: {k: x["tokens_after"] for k, x in v["events"].items()} for n, v in variants.items()}}
    save("report.json", out)
    for k, (ok, n) in out["conditions"].items():
        print(f"{k:24} {ok:4}/{n:<4} {100 * ok / n if n else 0:5.1f}%   UNKNOWN replies {out['unknown'][k]}")
    for k, (ok, n) in out["kinds"].items():
        print(f"{k:24} {ok:4}/{n:<4}")
    for k, v in out["pairs"].items():
        print(f"{k:34} n={v['n']:4} first-only {v['a_only']:3} second-only {v['b_only']:3} p={v['p']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "freeze": freeze()
    elif cmd == "stock": stock()
    elif cmd == "variant": variant(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif cmd == "ask": ask(sys.argv[2:])
    elif cmd == "report": report()
