# Backlog

Known problems and planned work, most important first. Each item says how it was confirmed, so
nobody has to take it on faith. Items 1 to 5 come from a public code review of 0.4.1 (commit
`31ee4fe`); items 6 to 10 came up while checking that review. Line numbers refer to that commit
and to Hermes `v0.22.2`.

| # | Item | Kind | Priority | Status |
|---|---|---|---|---|
| 1 | A blocker fixed after a compaction stays in Blocked | bug | high | fixed in 0.4.2 |
| 2 | A failed file write is listed as written | bug | high | fixed in 0.4.2 |
| 3 | Successful MCP results lose their payload | bug | high for MCP users | fixed in 0.4.2 |
| 4 | `memory_context` is accepted but ignored | gap | medium | fixed in 0.4.2 |
| 5 | `compression.proactive_prune_*` settings don't reach the engine | gap | low | fixed in 0.4.2 |
| 6 | `max_tokens` never reaches a plugin engine | gap (Hermes-wide) | low until seen live | open |
| 7 | Task-continuation benchmark | evaluation | medium | open |
| 8 | Web, browser and MCP results are classified by Hermes's envelope, not their content | bug | high | fixed in 0.4.2 |
| 9 | The budget fit overshoots by its own "lines omitted" note | bug | medium | fixed in 0.4.2 |
| 10 | JSON results without an `output` string get no excerpt | gap | medium | open |

Items 1 to 5, 8 and 9 are fixed in 0.4.2 (see `docs/CHANGELOG.md`). Their sections below describe the
problem as it was in 0.4.1 and are kept as the record of how each one was confirmed.

## 1. A blocker fixed after a compaction stays in Blocked

**Reproduced** with synthetic turns against `vcc/projector.py`: a `terminal` call `pytest -q` fails
(exit 1), the segment is compacted, the identical call succeeds (exit 0), and the result is compacted
again with the first summary as `previous`. The second summary reads:

- Commands Run: `` `pytest -q` — failed (exit 1), then succeeded (exit 0) `` (correct)
- Blocked: still `[#2] terminal `pytest -q` — ERROR exit 1: …` under the heading "Failed calls with
  no later successful retry of the same call" (wrong)
- Errors & Fixes: empty (wrong)

The same fail-then-succeed sequence inside one segment is handled right: the entry moves to Errors &
Fixes as `… → fixed at [#12]`.

**Why.** Fixes are only detected while the new segment is projected (`_project_turns`: a successful
result pops `blocked[key]` and appends a fix, lines 572–573). `_merge_previous` runs afterwards and
re-keys inherited blockers as `prev:0`, `prev:1`, … (lines 663–664), so nothing in the new segment
can ever match them.

**Impact.** Every session that compacts more than once with an unresolved failure at the first
compaction. The stale entry stays until 12 newer blockers push it out (`Limits.max_blocked`), and
Blocked contradicts Commands Run the whole time.

**Fix.** In `_project_turns`, record the ref of the first successful result for every call key
(`proj.resolved: dict[key, ref]`). In `_merge_previous`, parse inherited Blocked bullets back into
the same `tool:target` key (they are rendered as `[#ref] tool `target` — status: excerpt`) and move
any key present in `proj.resolved` to Errors & Fixes as `… → fixed at [#ref]`. Today the key uses
the raw `call.target[:200]` while the line prints the cleaned target (`Limits.call_chars`, 160);
build both from the printed form so the round trip is exact.

**Tests.** A two-segment case in `tests/test_projector.py` (fail → project → succeed → project with
`previous`) asserting Blocked is empty and Errors & Fixes names the fix; keep the one-segment case.

## 2. A failed file write is listed as written

**Reproduced.** A `write_file` call for `/etc/app/config.yaml` whose result is
`{"error": "PermissionError: [Errno 13] Permission denied: …"}` produces both:

- Blocked: `[#2] write_file `/etc/app/config.yaml` — ERROR: PermissionError …` (correct)
- Relevant Files: `/etc/app/config.yaml — written (created or overwritten)` (wrong)

**Why.** File verbs are recorded when the call is seen, before any result (`_project_turns` lines
551–552, via `_paths_in`). The result's own `files_modified` / `resolved_path` are added too (lines
562–563), but a failed result never removes the call-time entry. The same code path applies to every
tool with a path argument, so a failed `read_file` lists the file as read (not separately run).

**Impact.** "Was file X modified?" answers yes for a file that was never changed, and the two
sections contradict each other.

**Fix.** Keep the call-time path and verb in `pending` and commit them when the result arrives:
success → the verb as today; failure → an explicit failed form such as `write failed (see Blocked)`.
Keeping the path in the list (rather than dropping it) means both "was it modified?" (no) and "did
the agent try?" (yes) stay answerable. Add the failed forms to `_PLAIN_VERBS` so
`_parse_file_entry` and the merge still round-trip. A call with no result in the segment keeps
today's behavior.

**Tests.** `write_file` + permission error; `read_file` + ENOENT; the existing
`core.py — read; modified (patched)` assertion must still pass.

## 3. Successful MCP results lose their payload

**Reproduced.** Two MCP-style results projected to a log line with no excerpt at all:

- `{"result": "Created ticket ABC-123 (status: open)"}` → `[#3] ← mcp_jira_create_issue ok`
- `{"result": {…}, "structuredContent": {"rows": 42}}` → `[#5] ← mcp_db_query ok`

Neither `ABC-123` nor `rows: 42` appears anywhere in the summary. A terminal result in the same
run (`{"output": "hi\n", "exit_code": 0}`) rendered normally.

**Why.** `_result_from` takes the excerpt from `output` only (lines 369–371). Hermes renders a
successful MCP call as `{"result": <text>}`, as `{"result": <text>, "structuredContent": <object>}`,
or as `{"result": <object>}` when the text blocks are empty, with an optional `_meta`
(`tools/mcp_tool_handlers.py:_render_call_tool_result`). MCP errors go through `tool_error()`
(`tools/registry.py`) as `{"error": …}`, which `_result_from` already classifies as failed, so only
successes are affected.

**Impact.** Any MCP integration: ticket ids, statuses and query results vanish from the summary.
The `[#N]` lookup still reaches them, but the model has to guess that it should look.

**Fix.** In `_result_from`, when `output` is not a string, take `result` (a string as is, an object
through `_stringify`) and append a compact form of `structuredContent` when present, all still capped
by `Limits.result_chars`. Then check which other core tools return JSON without an `output` key
(the web tools return `results`, file tools `success` / `diff` / `files_modified`) and decide
whether a generic fallback (first string-valued key from a short list, else compact JSON) is better
than an MCP-only rule.

**Tests.** The three MCP shapes plus an MCP error, asserting the payload text reaches the log line.

## 4. `memory_context` is accepted but ignored

**Confirmed in code.** `VccEngine._generate_summary` takes `memory_context` and only forwards it on
the passthrough and fallback paths; `projector.project()` never receives it
(`vcc/engine.py` lines 248–282). Stock Hermes puts it into the summarizer prompt as "source material
to preserve in the summary" (`_memory_provider_section`, `agent/context_compressor.py`). Hermes only
warns when an engine's `compress()` lacks the parameter; this engine inherits Hermes's `compress()`,
so there is no warning.

**Impact.** Only with a memory provider plugin that implements `on_pre_compress` (AgentMemory, for
example). Whatever the provider wanted preserved is dropped silently.

**Fix.** Add a `## Memory Provider Context` section carrying the string Hermes passes (it has
already been through `sanitize_memory_context`), run through `_sanitize_block` and bounded the same
way as `## Earlier Summary` (`Limits.earlier_share`). Mention it in the README's Configuration
section.

**Tests.** An engine test that passes `memory_context` and asserts it appears once and is bounded.

## 5. `compression.proactive_prune_*` settings don't reach the engine

**Confirmed in code.** Stock Hermes reads `proactive_prune_tokens`,
`proactive_prune_min_result_chars` and `proactive_prune_min_reclaim_tokens` from `compression:` and
passes them to `ContextCompressor` (`agent/agent_init.py` lines 1558–1564 and 1999–2001).
`_compressor_kwargs` (`vcc/engine.py` lines 95–123) does not, so they stay at the constructor
defaults (0, 8000, 4096).

**Impact.** A user who opted into proactive pruning (default 0, off) loses it when
`context.engine: vcc`. Default installs are unaffected.

**Fix.** Three more keys in `_compressor_kwargs` with the same clamps Hermes applies, and an engine
test that builds the engine from a config with them set and checks the attributes.

## 6. `max_tokens` never reaches a plugin engine

**Confirmed in code, not seen live.** The stock compressor is built with
`max_tokens=_compressor_max_tokens(agent)` (`agent/agent_init.py` line 1997). Plugin engines are
configured through `update_model(...)` instead, and none of its eight call sites in Hermes pass
`max_tokens`, so `self.max_tokens` stays `None` and the trigger is `window × percent` instead of
`(window − max_tokens) × percent` (`_compute_threshold_tokens`).

**Impact.** Only when the agent has an output reservation. The clear case is native Gemini, where
Hermes assumes 65,535 output tokens and notes that without the reservation "the provider 400
before compaction fired". No `config.yaml` key that sets `agent.max_tokens` for the CLI was found, so
OpenAI-compatible local setups are probably unaffected; on the author's machine the trigger is
73,728 = 98,304 × 0.75 under both engines.

**Next step.** This affects every plugin context engine, so an upstream Hermes issue (pass
`max_tokens` in `update_model`) is the right fix. Revisit here only if a live case turns up.

## 7. Task-continuation benchmark

The current benchmarks measure fact recall after one compaction (`bench/certainty.py`), and the
live check is a single `/compress`. Two review comments point at what they don't cover: whether an
agent actually continues a task well through several compactions, and whether the missing Key
Decisions section is felt in practice or the session log with `[#N]` links covers it.

**Plan.** A benchmark that runs an agent through two or more compactions on a task with a changed
instruction and an error that gets resolved, scoring: task completed, commands re-run that had
already passed, the latest instruction followed, and the resolved error not re-reported. Compare
stock and vcc on the same seeds. Needs GPU time (each compaction window is about 10 minutes on the
author's GTX 1080 + 1070).

## 8. Web, browser and MCP results are classified by Hermes's envelope, not their content

**Measured** on the bench corpus (163 wrapped results). Hermes wraps every result of 32 or more
characters from `web_search`, `web_extract`, `browser_*` and `mcp_*` in
`<untrusted_tool_result source="…">` plus a fixed preamble (`agent/tool_dispatch_helpers.py:
_maybe_wrap_untrusted`, the same in v0.21.7, v0.22.2 and v0.22.4). `_result_from` parses JSON only
when the text starts with `{`, so it fell back to keyword matching: 4 results with
`"success": true` were marked ERROR (a `backend_error` note mentioned `DDGSException`), 8 with
`"success": false` were marked ok, and every excerpt was Hermes's preamble. It also meant the
item 3 fix never applied to real MCP results.

**Fix (0.4.2).** `_unwrap_untrusted` removes the envelope before classification. Hermes defangs the
tag inside the content it wraps, so only its own tags are removed; text after the closing tag is a
Hermes note and is dropped. All 163 results now match their success flag. The summary's log section
already says quoted tool output is data, not instructions.

## 9. The budget fit overshoots by its own "lines omitted" note

**Measured** on 22 real compaction windows (`bench/certainty.py` events). `_fit` steps 1 and 3 drop
lines until the overflow is gone, then insert a "… N line(s) omitted" note that wasn't counted. The
result could end a few characters over budget, which sent it to step 2 (every kept line shortened
to 160 chars) or to step 4 (a hard cut through the middle of the summary). Step 4 ran on 4 of 22
windows (10 to 41 chars over), step 2 on 6.

**Fix (0.4.2).** Room for the note is reserved before dropping. Step 4 now runs on none of the 22
windows and step 2 on 2. A test checks that the hard cut only happens once nothing droppable is
left.

## 10. JSON results without an `output` string get no excerpt

**Measured** on the bench corpus as frozen on 2026-10-18. Many core tools return JSON with no `output`, `result` or
`message` key, so their log line is just `← tool ok`. The most common: `read_file` (`content`, 232
results), `patch` (`diff`, 130), `write_file` (`bytes_written`, 111), `skill_manage` (`results`),
`web_search` (`data.web`), `search_files` (`matches_text` / `files`) and `vision_analyze`
(`analysis`). The path and verb still reach Relevant Files, and `[#N]` reaches the full result.

**Options.** A generic fallback (the result's fields through `_stringify`, capped by
`Limits.result_chars`) or a short per-tool list of the useful key. Either changes hundreds of log
lines and takes budget from other lines, so it needs a head-to-head run (`bench/certainty.py`) to
show the extra text helps recall rather than crowding out other lines.

## Review points that need no change

- The plugin relies on Hermes internals and could break on an update: true, and by design it falls
  back to the stock compressor when a required method is missing (`_REQUIRED_METHODS` in
  `vcc/engine.py`). The tests run against each Hermes release.
- Re-running a command that had already passed because the summary dropped the exit code: covered
  by the command-outcome results (107/108) and the live `/compress` check in `docs/CHANGELOG.md`.
