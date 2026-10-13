# Changelog

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
