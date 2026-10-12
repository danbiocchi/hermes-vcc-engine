# Catalog PR text (draft)

**Title:** plugin-catalog: add hermes-vcc-engine

Adds `hermes-vcc-engine`, a context engine (`context.engine: vcc`) that replaces the summary LLM call in compaction with a deterministic projection of the transcript.

**Why:** on local or slow models the stock summary call is the bottleneck. On a 27B local model it was 4 to 15 minutes per compaction with the session blocked. The projection takes milliseconds (77 ms on a 695k-token session, 22 ms on 79k) and keeps a `[#N]` reference to every compacted message, which the agent can expand with the built-in `session_search(around_message_id=N)`.

**How it plugs in:** `register(ctx)` calls `ctx.register_context_engine(VccEngine())`. `VccEngine` subclasses `ContextCompressor` and overrides only `_generate_summary`, reusing the stock post-processing (redaction, skill-marker reinjection, task grounding, lean sections, provenance check). No runtime rebinding of core. It falls back to the stock summary if a projection fails, and to full passthrough if a required base-class method is missing.

**Checks:** `hermes plugins validate` passes (security scan: safe, no core override). 49 tests, including full `compress()` runs against a real Hermes checkout with the LLM mocked to fail if called.

**Disclosure:**
- No network calls, no subprocesses, no file I/O beyond reading Hermes's own config through `hermes_cli.config.load_config_readonly()` (the `compression:` and `context.vcc` sections), since Hermes doesn't pass compression settings to plugin engines.
- No credentials, no telemetry, no self-update.
- Derived from the idea in VCC (lllyasviel/VCC, arXiv 2603.29678); independent implementation, no code copied.
