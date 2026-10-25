# Changelog

## 0.4.2

Fixes for the problems found in a public review of 0.4.1 and two found while checking them (`docs/BACKLOG.md`, items 1 to 5, 8 and 9). The summary text changes, so the benchmark golden was rewritten.

- **A failure fixed after a compaction leaves Blocked.** Before, a call that failed before one compaction and succeeded after it stayed in Blocked, while Commands Run said it had succeeded. Inherited Blocked entries now parse back to the same `tool:target` key, and a later success of that call moves the entry to Errors & Fixes as `… → fixed at [#N]`. A repeat failure replaces the inherited entry instead of adding a second one.
- **A failed file action is no longer listed as done.** File verbs used to be recorded when the call was made, so a `write_file` that hit a permission error was listed as `written (created or overwritten)`. The verb is now recorded when the result arrives. A failure reads `write failed (not written)`, `read failed`, `patch failed (not modified)` and so on, and a later success on the same file replaces it.
- **Successful MCP results keep their payload.** `{"result": …}` and `structuredContent` now reach the log line, so ticket ids and query results are no longer lost behind `ok`. Core tools that return only a one-line `message` (`memory`, for example) keep it too. A JSON error is quoted once instead of twice.
- **Memory provider context is kept.** The `memory_context` Hermes passes from a memory provider's `on_pre_compress` used to be dropped. It now gets a `## Memory Provider Context` section. The section is treated as data, capped like `## Earlier Summary`, and carried into later summaries until the provider sends new text.
- **`compression.proactive_prune_*` settings reach the engine**, with the same parsing and floors as Hermes. Before, they stayed at the defaults (off) with `context.engine: vcc`.
- **Web, browser and MCP results are classified by their content.** Hermes wraps the results of these tools in an `<untrusted_tool_result>` envelope with a fixed preamble. Because the text then no longer started with `{`, the JSON inside was never read: the outcome came from keyword matching, and every log line quoted Hermes's preamble instead of the result. On the bench corpus, 4 of 163 wrapped results were marked failed although they reported `"success": true`, and 8 failures were marked ok. The envelope is now removed before the result is classified, so all 163 match their own success flag, and the MCP fix above also works in real sessions.
- **The budget fit no longer overshoots by its own note.** Dropping log lines adds a "… N line(s) omitted" note, which wasn't counted. The summary could end up a few characters over budget, so every kept line got shortened, or the middle of the summary was cut off. On 22 real compaction windows, the cut-off happened on 4 and now on none, and kept lines are shortened on 2 instead of 6.
- Chat-template markers are now also neutralized in inherited and memory-provider text (`_sanitize_block`).
- `bench/proofs.py` on the bench corpus: 821 / 821 log lines verified (0.4.1: 783 / 783), 11 / 11 planted fabrications caught, 467 / 467 `[#N]` links in stored summaries opened the right message. Commands retained in the summary: 526 / 890 (0.4.1: 512); user requests: 166 / 216 (0.4.1: 161).
- Tested against Hermes v0.21.7, v0.22.2 and v0.22.4: all 101 tests pass on each.

## 0.4.1

A tidier repository. The summary output and the engine's behavior are unchanged.

- The engine's modules (`engine.py`, `projector.py`, `anchors.py`) and `NOTICE` moved into `vcc/`. The top level keeps only what Hermes needs to load the plugin (`__init__.py`, `plugin.yaml`), the README and the license.
- `CHANGELOG.md` moved to `docs/`, `pytest.ini` to `tests/` (run the suite with `pytest tests`), and the round-0 accuracy results to `bench/results/`.
- The benchmark scripts still accept older checkouts through `--repo` / `VCC_REPO`: they look for the modules in `vcc/` and fall back to the checkout root.
- Tested against Hermes v0.22.2: all engine tests pass, and a live `/compress` on a local model kept every command outcome.

## 0.4.0

Command outcomes are now right when Hermes has already shortened a result.

- Before the summary step, Hermes replaces old terminal output with a one-line stub (``[terminal] ran `cmd` -> exit 1, 40 lines output``). 0.1 to 0.3 didn't read the exit code from it, so a failed command was treated as a success, and 0.3 listed it as `succeeded`. On 69 real compaction windows, 0.3 showed 13 of 43 failing commands as failed and stated the wrong final outcome for 20 commands. 0.4 shows all 43 and gets every final outcome right.
- A run whose result has no exit code at all reads `ran (no exit code)` instead of `succeeded`.
- `bench/proofs.py`: a log line's status must match a stub's exit code, and reference integrity counts only the `[#N]` pointers vcc wrote, not ones quoted from pasted text.
- `bench/questions.py`: a question generator written without reading the projector, including real "no" questions (files that were only read, commands that failed).
- `bench/certainty.py`: a head-to-head that asks inside the real post-compaction message list, with an untruncated baseline, window-level statistics and the cost of re-reading the compacted prompt.
- Engine tests pass against the tagged Hermes 0.21.7 release.

## 0.3.0

Summaries a small model can actually use. On 13 real compaction points and 164 machine-graded questions, the local 27B answers 105 correctly from a vcc 0.3 summary, the same as from the full transcript, against 89 from stock Hermes's LLM summary and 70 from vcc 0.2.

- Files say what happened in plain words: `modified (patched)`, `written (created or overwritten)`, next to the tool's own verb.
- Commands Run lists each command's outcome in order, e.g. `failed (exit 1), then succeeded (exit 0)`; repeats collapse to `(3 times)`.
- New "Error Names Seen" section: named errors (`ImportError`, `ENOENT` ...) with the message where each first appeared.
- A carried-over non-vcc summary may use at most 15% of the budget (was 35%).
- The session log header explains the `→` / `←` notation.
- Merges read both the 0.3 and the 0.2 format, so upgrading mid-session is safe.
- `bench/headtohead.py`: stock Hermes vs vcc on real compaction points, with paraphrased questions and McNemar's exact test.
- `bench/proofs.py`: reference integrity counts only the `[#N]` pointers vcc wrote, not ones quoted from pasted text.
- Summary step costs 3% more than 0.2; still 1.6x faster than 0.1.

## 0.2.0

Same output as 0.1.0, byte for byte, in a lot less time. The engine's summary step is 38% faster across the benchmark corpus and about 2x faster on big sessions.

- Projector 2.4x faster: guarded text cleanup, a lowercase/anchored fast path for error detection (exact; falls back to the regex for the four characters that case-fold onto ASCII letters), dict fast paths, slotted dataclasses.
- `anchors.py`: Hermes's anchor index built by scanning only the lines each pattern can match. Checked against Hermes's builder at startup and on the first real compaction; any mismatch turns it off. New option `context.vcc.fast_anchors`.
- `bench/bench.py`: benchmark plus byte-identity harness over a frozen corpus of real and synthetic transcripts.
- Exhaustive tests over every Unicode code point for each text-handling shortcut.
- `bench/proofs.py`: faithfulness (every log line quotes its cited message), a negative control that a planted fabrication is rejected, reference integrity through Hermes's own `session_search`, and retention per compression ratio. Runs standalone with `--synthetic` and is enforced in CI.
- `bench/accuracy.py` + `bench/analyze.py`: model-level accuracy under FULL / vcc / vcc+lookup / stock-LLM-summary contexts, compared with McNemar's exact test.

## 0.1.0

First release.

- `vcc` context engine: Hermes's compressor with the summary LLM call replaced by a deterministic projection.
- `[#N]` message references in every log line, recoverable with `session_search(around_message_id=N)`.
- Error tracking: failures that were retried successfully land in Errors & Fixes, the rest in Blocked.
- Iterative merges keep the original goal and stay inside the budget across any number of compactions.
- Reads the `compression:` section itself (Hermes doesn't pass it to plugin engines); trigger follows the context window.
- Passthrough mode if the Hermes build is missing anything the engine relies on; per-pass fallback to the stock summary on any projection error.
