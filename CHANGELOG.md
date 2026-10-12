# Changelog

## 0.1.0

First release.

- `vcc` context engine: Hermes's compressor with the summary LLM call replaced by a deterministic projection.
- `[#N]` message references in every log line, recoverable with `session_search(around_message_id=N)`.
- Error tracking: failures that were retried successfully land in Errors & Fixes, the rest in Blocked.
- Iterative merges keep the original goal and stay inside the budget across any number of compactions.
- Reads the `compression:` section itself (Hermes doesn't pass it to plugin engines); trigger follows the context window.
- Passthrough mode if the Hermes build is missing anything the engine relies on; per-pass fallback to the stock summary on any projection error.
