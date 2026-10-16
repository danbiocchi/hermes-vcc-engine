# Changelog

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
