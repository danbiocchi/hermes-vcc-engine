# hermes-vcc-engine

**Context compaction for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that takes milliseconds instead of minutes, and doesn't lose anything.**

Hermes compacts a long conversation by asking an LLM to summarize the old turns. This plugin replaces that LLM call with a deterministic projection of the transcript: what you asked, what ran, what failed, what got fixed, which files changed. Every line points back to the message it came from, so the agent can open the original whenever it needs a detail.

| | stock Hermes (LLM summary) | **vcc** |
|---|---|---|
| Time per compaction, local 27B | 4 to 15 minutes, chat frozen | **0.15 seconds** |
| Model answers questions about the compacted work | 8 / 26 | **18 / 26** |
| Matches the full, uncompacted transcript | | **31 / 45 vs 31 / 45** |
| Summary lines that quote their source exactly | not checkable | **722 / 722** |
| Pointers that open the original message | none | **322 / 322** |
| Extra model calls | 1 per compaction | **0** |

Measured on my own sessions. Every number here comes from a script in this repo that you can rerun against yours.

## Install

```bash
hermes plugins install danbiocchi/hermes-vcc-engine --enable
hermes config set context.engine vcc
```

Restart whatever runs Hermes (CLI, gateway, web UI) and look for `Using context engine: vcc` in the agent log. Needs Hermes 0.21.7 or newer, and nothing else.

To go back: `hermes config set context.engine compressor`.

## Why I built it

I run Hermes on a local Qwen 27B split across a GTX 1080 and a 1070. Every compaction froze the chat while the model wrote prose: 237 s, 365 s, 553 s, 781 s for my last four, plus one I killed at 15 minutes. Twice the "summary" came back longer than what it replaced.

The projection does the same job in 0.149 s on my largest session (1,486 messages), which is 1,600x to 5,200x faster.

## What a compaction looks like

```
## Goal
The build in ~/proj is red, please fix it and keep the public API unchanged.

## Errors & Fixes
- [#1001] terminal `cd ~/proj && pytest -q` — ERROR exit 1: E ImportError: cannot import name 'parse' from 'proj.core' 1 failed → fixed at [#1008]

## Relevant Files
- /home/u/proj/proj/core.py (read, patched)

## Commands Run
- `cd ~/proj && pytest -q`

## Detailed Session Log (oldest first)
[#1000] user: The build in ~/proj is red, please fix it and keep the public API unchanged.
[#1001] → terminal: `cd ~/proj && pytest -q`
[#1002] ← terminal ERROR exit 1: E ImportError: cannot import name 'parse' from 'proj.core' 1 failed
[#1003] → read_file: `/home/u/proj/proj/core.py`
[#1005] → patch: `/home/u/proj/proj/core.py`
[#1007] → terminal: `cd ~/proj && pytest -q`
[#1008] ← terminal exit 0: 48 passed in 1.2s
[#1009] assistant: Fixed: 'parse' was renamed to '_parse' in core.py. All 48 tests pass.
```

Each `[#N]` is a real message id. `session_search(session_id=..., around_message_id=N)`, which Hermes already has, returns the original in full. Old messages stop being re-sent on every request but stay in `state.db`.

Hermes then adds its own sections as usual: the anchor index, your messages verbatim, and the recovery note. On later compactions the previous projection gets merged instead of summarized again, so the original goal survives any number of passes.

## The evidence

### 1. It doesn't make anything up

```
faithfulness         722 / 722   ██████████████████████████████  100%
negative control      13 /  13   ██████████████████████████████  100%
reference integrity  322 / 322   ██████████████████████████████  100%
byte-identity        126 / 126   ██████████████████████████████  100%
```

- **Faithfulness:** for every log line, the role, tool and exit code match the message it cites, and every word and number in its excerpt appears verbatim in that message.
- **Negative control:** plant an invented word in a log line and the same check must catch it. It caught 13 of 13, so the clean pass above means something.
- **Reference integrity:** every `[#N]` in the real summaries in my `state.db`, opened through Hermes's own `session_search`, returns the right message.
- **Byte-identity:** v0.2 is a speed release, so all 126 outputs must hash the same as v0.1 on frozen inputs. They do.

CI runs the same proofs on generated transcripts, with no private data, on every push (1,132 of 1,132 lines).

### 2. The model can still answer questions

`bench/accuracy.py` generates factual questions from the transcript, each with one checkable answer: a file path, whether a command succeeded, an error name, or a yes/no that includes made-up facts as controls. It then asks the same questions with different contexts.

**Against the full transcript, 45 questions over 6 real sessions:**

| context given to the model | correct | accuracy |
|---|---:|---:|
| full transcript (best case) | 31 / 45 | 69% |
| **vcc summary** | **31 / 45** | **69%** |
| vcc summary + `[#N]` lookup | 30 / 45 | 67% |

The full transcript and the vcc summary disagree on 6 questions and split them 3 to 3 (McNemar exact test, p = 1.0). This shows no detectable loss on this set; it doesn't prove the two are identical. With 45 questions the margin is about ±13 points, and 69% is the ceiling of my local model on these questions, not a score for the engine.

**Against the summary it replaces, 26 questions:**

```
full transcript     ███████████████████████  20/26   77%
vcc + [#N] lookup   ██████████████████████   19/26   73%
vcc summary         █████████████████████    18/26   69%
stock LLM summary   █████████                 8/26   31%
```

To see why, I checked the 7 facts those questions asked about: the vcc summary contained 7 of 7, Hermes's anchor index 3 of 7, and the local model's prose had dropped the exact commands and paths. That's the stock compressor running on a local 27B, which is the setup this plugin is for. A frontier model writes a much better summary.

### 3. What survives inside the summary itself

Factory's [compression evals](https://factory.com/news/evaluating-compression) found that tracking which files got touched was the weakest area of every compressor they measured, so here it is per compression ratio rather than as one average:

| compression | files modified | commands run | user requests |
|---|---:|---:|---:|
| up to 2.4x (11 sessions) | 9 / 9 | 29 / 30 | 26 / 26 |
| 10x (1 session) | 2 / 2 | 32 / 42 | 6 / 6 |
| 83x (worst case) | 16 / 20 | 57 / 88 | 23 / 30 |

Errors from calls that really failed: 2 of 2. At the ratios Hermes actually compacts at, almost everything survives. The 83x row is an artificial stress test, a 1.96M-character history forced into one 24k-character summary. Anything past the caps is still in `state.db` and one `session_search` away.

### 4. Speed

Whole summary step, median of 7 runs:

```
1,486 messages    v0.1 ██████████████████████████████  325 ms
                  v0.2 ██████████████                  149 ms

254 messages      v0.1 ██████████████████████████████   71 ms
                  v0.2 ██████████████                   34 ms

whole corpus      v0.1 ██████████████████████████████  840 ms
3,717 messages    v0.2 ██████████████████▋             523 ms
```

For scale, the stock LLM call this replaces took 237,000 to 781,000 ms on the same machine.

**Test setup:** 13 real sessions (3,717 messages, 4.0M characters) plus 8 seeded stress transcripts, run on a Ryzen 5 2600X with the Python 3.14 that ships with Hermes and Qwen3.8-27B IQ3_S across a GTX 1080 and a 1070.

## Is it for you?

**Yes** if you run Hermes on a local or slow model, or your sessions are long and full of tool calls.

**Probably not** if you're on a frontier API where compaction already takes a couple of seconds and you like the model's narrative summary.

Trying it costs little. If a projection ever throws, that one pass falls back to the normal LLM summary. If a Hermes update changes something the engine depends on, it notices at startup and behaves exactly like the stock compressor. Rolling back is one command.

## Limits

- It quotes, it doesn't reason. There's no "Key Decisions" section, because that takes judgement.
- Recall depends on the agent actually calling `session_search` when it needs a detail. The compaction note tells it to, but that's an instruction, not a guarantee.
- Every compaction rewrites the prompt, so the prompt cache resets. The summary is kept small (6% of the window by default) to keep the re-read cheap.
- `plugins.isolation: host` is untested. In-process, the default, is what it's built and tested against.

## How it hooks in

The engine subclasses Hermes's own `ContextCompressor` and overrides one method, `_generate_summary`. Everything else is stock Hermes: when to compact, what to keep verbatim, tool call and result pairing, the session-DB archive, cancellation and secret redaction. It registers through `ctx.register_context_engine` with no monkeypatching, and `hermes plugins validate` passes with a clean security scan.

## Safety

No network, no file I/O, no subprocesses, no `eval`. The projector is pure stdlib Python.

Tool output is treated as untrusted data: ANSI escapes, control and bidi characters are stripped; chat-template markers (`<|im_start|>`, `[INST]`, `<<SYS>>`, `<think>`) are neutralized; and quoted text can't open a new `##` section. Secrets go through Hermes's own `_redact_compaction_text` before anything is stored. Output size stays bounded whatever the transcript contains, and that's fuzz-tested.

## Config

Optional, under `context.vcc` in `config.yaml`:

```yaml
context:
  engine: vcc
  vcc:
    budget_ratio: 0.06       # summary size as a share of the context window
    min_budget_tokens: 1500
    max_budget_tokens: 8000
    fallback_to_llm: true    # if a projection ever fails, use the stock summary for that pass
    fast_anchors: true       # build Hermes's anchor index the fast way; false uses Hermes's builder
```

Your usual `compression:` settings (threshold, threshold_tokens, protect_first_n, protect_last_n, tail_mode, model_thresholds) are honored exactly as the built-in compressor honors them.

<details>
<summary>Tip: let the trigger follow your context window</summary>

If you set `compression.threshold_tokens` back when your context was smaller, delete it. Without it the trigger follows `model.context_length` at 75% of the usable window, so it adjusts itself when you change context. Note that `model.context_length` only pins your default model; other models auto-detect, and on a llama.cpp router that can resolve to the GGUF's native window rather than your actual `n_ctx`. A per-route `custom_providers` entry with `context_length` fixes that.
</details>

## Reproduce the numbers

```bash
PY=~/.hermes/tools/python-*/bin/python3          # the interpreter Hermes itself uses

$PY -I bench/proofs.py   --synthetic               # proofs on generated data, no Hermes or GPU needed
$PY -I bench/proofs.py                             # faithfulness, negative control, retention, references on your state.db
$PY -I bench/bench.py    --write-golden            # freeze inputs, record output hashes
$PY -I bench/bench.py    --check-golden            # timings plus byte-identity against the golden
$PY -I bench/accuracy.py --model <your-model-id>   # model-level accuracy; needs the GPU and is slow
python   bench/analyze.py bench/accuracy-paired.json
```

None of the scripts print transcript text, only scores, counts and hashes. `--write-golden` keeps its frozen corpus in `~/.cache/hermes-vcc-engine/` at mode 0600, never in the repo.

<details>
<summary>How v0.2 got 2x faster without changing a byte of output</summary>

Profiling real compactions put the time somewhere I didn't expect. On the biggest session, half of it was Hermes's anchor index running regexes over every character of the compacted region. A quarter was the projector, mostly regex passes plus `isinstance` checks against `typing.Mapping`.

- Text cleanup runs each rewrite step only when the text could contain what that step rewrites: an ANSI sequence needs an ESC, a chat marker needs `<` or `[`. Whitespace collapses with `split()` and `join` instead of a regex. Tests check the character tables against the old regexes across all 1,114,112 code points.
- Error detection was a case-insensitive regex. Every alternative in it is a fixed phrase or starts at a fixed word, so the text is lowercased once and the regex runs only where one of those words occurs. Lowercasing is exact unless the text contains one of the four non-ASCII characters that fold onto ASCII letters (`İ ı ſ K`, found by exhaustive scan), in which case the original regex runs. That's about 13x faster on real tool output.
- Hermes's anchor patterns are all line-local and each needs a literal (`.py`, `fix/`, `Error`, `http`), so each pattern only scans lines that contain its literal. Commit hashes have no literal, so those lines are found with a `bytes.translate` hex mask. Dense text falls back to a full scan. The result is checked against Hermes's own builder at startup and on the first real compaction of every process, and any difference turns the fast path off.
- Then the small stuff: `dict` fast paths, slotted dataclasses, and not copying multi-megabyte tool results just to look at their first character.

Tried and not shipped: dropping the second redaction pass (Hermes's redactor isn't idempotent, so removing it would have meant less redaction), CPython 3.14's JIT (no measurable difference, since the hot paths are already C), and shrinking the summary format (only 3.4% of summary tokens are exact duplicates, and the tokenizer charges the same for `→` as for `->`).
</details>

<details>
<summary>Research notes</summary>

- [The Complexity Trap](https://arxiv.org/abs/2508.21433) (JetBrains and TUM): simply masking old tool output performs about as well as LLM summarization for SWE agents at roughly half the cost. Deterministic projection is in the same family as masking, with pointers back to the original.
- Factory's [compression evals](https://factory.com/news/evaluating-compression): tracking which files were touched was the weakest area for every method they tested, and they argue for measuring tokens per task rather than per request. That's why this keeps exact paths and commands.
- [LLMLingua-2](https://arxiv.org/abs/2403.12968) and similar prompt compressors drop tokens using a small model and lose a lot of accuracy outside their training domain. That's the wrong trade when exact identifiers are the point.
- Chroma's [context rot](https://research.trychroma.com/context-rot) and [Lost in the Middle](https://arxiv.org/abs/2307.03172) are why the summary stays small and the newest material stays at the end.
- Manus's [context engineering notes](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus): keep the prefix stable, and make compression restorable. The `[#N]` pointers are the restorable part.
</details>

## Development

```bash
git clone https://github.com/danbiocchi/hermes-vcc-engine && cd hermes-vcc-engine
uv venv --python 3.14 && uv pip install pytest
.venv/bin/python -m pytest -q          # 78 tests; CI runs 3.11 through 3.14
```

Projector tests run anywhere. Engine and anchor tests need a real Hermes checkout (`HERMES_AGENT_DIR`, default `~/.hermes/hermes-agent`) and skip without one. They mock the LLM so the test fails if it's ever called, run full `compress()` passes, compare the fast anchor index against Hermes's own, and load the plugin through Hermes's actual loader. `HERMES_HOME` is sandboxed, so tests never touch your real profile.

Before changing anything that affects output, run `bench/bench.py --write-golden` first and `--check-golden` afterwards.

## License

Apache-2.0. `anchors.py` mirrors the structure of Hermes's anchor index builder (MIT); see [NOTICE](NOTICE). The idea comes from [VCC](https://github.com/lllyasviel/VCC) (arXiv 2603.29678); this is an independent implementation.
