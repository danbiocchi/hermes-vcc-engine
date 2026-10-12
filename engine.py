"""``vcc`` context engine: Hermes's own compressor with the summary LLM call swapped out.

Everything that is hard to get right stays Hermes's code: when to compact, which turns to
keep, tool-call/result pairing, the session-DB archive commit, cancellation and secret
redaction. The one thing replaced is ``_generate_summary``, the step that sends the
compacted turns to a model and waits for prose. Here it is a deterministic projection
that takes milliseconds (see ``projector.py``).

If this Hermes build doesn't look like the one the engine was written against, or the
projection fails for any reason, it hands the call back to the stock implementation.
Worst case you get the normal LLM summary, never a broken session.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

from agent.context_compressor import ContextCompressor

try:  # loaded as a package (Hermes plugin loader)
    from . import anchors, projector
except ImportError:  # loaded as a top-level module (tests, tooling)
    import anchors  # type: ignore[no-redef]
    import projector  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

ENGINE_NAME = "vcc"

# Base-class pieces the override reuses. Checked once at construction; any missing one
# switches the engine to passthrough (stock LLM summaries) with a single warning.
_REQUIRED_METHODS = (
    "_generate_summary", "_compression_cancelled", "_ground_historical_task_snapshot",
    "_augment_summary_lean", "_validate_summary_user_provenance", "_with_summary_prefix",
    "_transcript_has_real_user_turn", "_is_synthetic_compression_user_turn",
    "_clear_compression_failure_cooldown", "_clear_terminal_summary_failures",
    "_reset_consecutive_overload_aborts", "_compute_summary_budget", "update_model",
)
_REQUIRED_FUNCS = (
    "_redact_compaction_text", "_reinject_pruned_skill_markers", "_collect_ghosted_skill_names",
    "_extract_pruned_skill_names", "_MAX_PRUNED_SKILL_MARKERS",
)

# Option defaults (``context.vcc`` in config.yaml).
_DEFAULTS: dict[str, Any] = {
    "budget_ratio": 0.06,        # summary budget as a share of the context window
    "min_budget_tokens": 1500,
    "max_budget_tokens": 8000,
    "fallback_to_llm": True,     # on an unexpected projection error, use the stock summary
    "fast_anchors": True,        # build Hermes's anchor index with anchors.py (same output, faster)
}
_CHARS_PER_TOKEN = 4  # same rough ratio Hermes's own estimator uses

# anchors.build() must reproduce Hermes's anchor index byte for byte. It is checked against
# Hermes's builder once per process on an edge-case fixture and again on the first real
# compaction; any disagreement or error turns it off for the rest of the process.
_FAST_ANCHORS: dict[str, Any] = {"fixture_ok": None, "live_ok": False, "disabled": ""}


def _disable_fast_anchors(reason: str) -> None:
    if not _FAST_ANCHORS["disabled"]:
        _FAST_ANCHORS["disabled"] = reason
        logger.warning("vcc: fast anchor index off (%s); Hermes builds it as usual", reason)


def _load_hermes_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return cfg if isinstance(cfg, Mapping) else {}
    except Exception as exc:  # config is optional; defaults are fine
        logger.debug("vcc: could not read Hermes config (%s); using defaults", exc)
        return {}


def _section(cfg: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    node: Any = cfg
    for key in path:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(hi, max(lo, number))


def _compressor_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Map ``compression:`` to ContextCompressor kwargs.

    Hermes doesn't pass its compression settings to plugin engines, so read the same keys
    it reads for the built-in compressor and default them the same way.
    """
    comp = _section(cfg, "compression")
    cap = comp.get("threshold_tokens")
    try:
        cap = int(cap) if cap is not None and int(cap) > 0 else None
    except (TypeError, ValueError):
        cap = None
    thresholds = {}
    for key, value in _section(comp, "model_thresholds").items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "threshold_percent": _num(comp.get("threshold", 0.75), 0.75, 0.05, 0.98),
        "protect_first_n": int(_num(comp.get("protect_first_n", 3), 3, 0, 50)),
        "protect_last_n": int(_num(comp.get("protect_last_n", 20), 20, 1, 500)),
        "summary_target_ratio": _num(comp.get("target_ratio", 0.20), 0.20, 0.10, 0.80),
        "abort_on_summary_failure": bool(comp.get("abort_on_summary_failure", False)),
        "model_thresholds": thresholds,
        "threshold_tokens_cap": cap,
        "min_tail_user_messages": int(_num(comp.get("min_tail_user_messages", 1), 1, 1, 20)),
        "tail_mode": str(comp.get("tail_mode", "lean")).strip().lower() or "lean",
    }


def _engine_options(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = _section(cfg, "context", "vcc")
    return {
        "budget_ratio": _num(raw.get("budget_ratio", _DEFAULTS["budget_ratio"]), _DEFAULTS["budget_ratio"], 0.01, 0.30),
        "min_budget_tokens": int(_num(raw.get("min_budget_tokens", _DEFAULTS["min_budget_tokens"]),
                                      _DEFAULTS["min_budget_tokens"], 300, 50_000)),
        "max_budget_tokens": int(_num(raw.get("max_budget_tokens", _DEFAULTS["max_budget_tokens"]),
                                      _DEFAULTS["max_budget_tokens"], 500, 100_000)),
        "fallback_to_llm": bool(raw.get("fallback_to_llm", _DEFAULTS["fallback_to_llm"])),
        "fast_anchors": bool(raw.get("fast_anchors", _DEFAULTS["fast_anchors"])),
    }


def _missing_seams() -> list[str]:
    import agent.context_compressor as cc
    missing = [m for m in _REQUIRED_METHODS if not callable(getattr(ContextCompressor, m, None))]
    missing += [f for f in _REQUIRED_FUNCS if not hasattr(cc, f)]
    try:
        from agent.conversation_compression import _raise_if_stale_attempt  # noqa: F401
    except ImportError:
        missing.append("conversation_compression._raise_if_stale_attempt")
    return missing


class VccEngine(ContextCompressor):
    """ContextCompressor whose summaries are deterministic projections instead of LLM output."""

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None, **overrides: Any) -> None:
        cfg = _load_hermes_config() if config is None else config
        kwargs = _compressor_kwargs(cfg)
        kwargs.update(overrides)
        kwargs.setdefault("quiet_mode", True)
        super().__init__(model=kwargs.pop("model", ""), **kwargs)
        self._vcc_config = cfg
        self._vcc_options = _engine_options(cfg)
        self._vcc_overrides = dict(overrides)
        self._vcc_last_ms: Optional[float] = None
        self._vcc_runs = 0
        self._vcc_fallbacks = 0
        missing = _missing_seams()
        self._vcc_passthrough = bool(missing)
        if missing:
            logger.warning(
                "vcc: this Hermes build lacks %s; using stock LLM summaries until the plugin is updated",
                ", ".join(missing),
            )

    @property
    def name(self) -> str:
        return ENGINE_NAME

    # The summariser model never runs on this engine's normal path, so its context window
    # must not cap our trigger (the session-start feasibility probe installs it).
    @property
    def _aux_context_ceiling(self) -> Optional[int]:
        return None

    @_aux_context_ceiling.setter
    def _aux_context_ceiling(self, _value: Optional[int]) -> None:
        pass

    def clone_for_agent(self) -> "VccEngine":
        # A fresh instance per agent: no shared mutable state, nothing deepcopy could trip on.
        return type(self)(config=self._vcc_config, **self._vcc_overrides)

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status.update({
            "vcc_passthrough": self._vcc_passthrough,
            "vcc_runs": self._vcc_runs,
            "vcc_fallbacks": self._vcc_fallbacks,
            "vcc_last_ms": self._vcc_last_ms,
            "vcc_fast_anchors": (
                "off (config)" if not self._vcc_options["fast_anchors"]
                else f"off ({_FAST_ANCHORS['disabled']})" if _FAST_ANCHORS["disabled"]
                else "on" if _FAST_ANCHORS["live_ok"] else "pending first check"
            ),
        })
        return status

    def _vcc_fast_anchor_section(self, cc: Any, summary: str, turns: list[dict[str, Any]]) -> Optional[str]:
        """Hermes's anchor section for ``turns``, built faster. ``None`` leaves it to Hermes."""
        state = _FAST_ANCHORS
        if not self._vcc_options["fast_anchors"] or state["disabled"]:
            return None
        if getattr(self, "tail_mode", "lean") != "lean":
            return None  # Hermes adds no lean sections in legacy mode
        try:
            if state["fixture_ok"] is None:
                state["fixture_ok"] = anchors.verify(cc)
                if not state["fixture_ok"]:
                    _disable_fast_anchors("self-check against Hermes's builder failed")
                    return None
            if cc._LEAN_ANCHOR_HEADING in summary:
                return None  # Hermes would skip the section, so must we
            fast = anchors.build(turns, cc)
            if not state["live_ok"]:
                if fast != cc._build_anchor_index(turns):
                    _disable_fast_anchors("differed from Hermes's builder on a real compaction")
                    return None
                state["live_ok"] = True
            return fast or None
        except Exception as exc:  # the optimization must never be the reason a compaction fails
            _disable_fast_anchors(f"{type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ summary override

    def _vcc_budget_chars(self, turns: list[dict[str, Any]]) -> int:
        opts = self._vcc_options
        window = int(getattr(self, "context_length", 0) or 0)
        by_window = int(window * opts["budget_ratio"]) if window > 0 else opts["max_budget_tokens"]
        by_content = self._compute_summary_budget(turns)  # Hermes's own content-scaled budget
        tokens = min(by_window, by_content, opts["max_budget_tokens"])
        return max(opts["min_budget_tokens"], tokens) * _CHARS_PER_TOKEN

    def _vcc_is_real_user(self, message: Mapping[str, Any]) -> bool:
        try:
            return not self._is_synthetic_compression_user_turn(message)
        except Exception:
            return True

    def _generate_summary(
        self,
        turns_to_summarize: list[dict[str, Any]],
        focus_topic: Optional[str] = None,
        memory_context: str = "",
        bypass_cooldown: bool = False,
    ) -> Optional[str]:
        if self._vcc_passthrough:
            return super()._generate_summary(turns_to_summarize, focus_topic, memory_context, bypass_cooldown)

        import agent.context_compressor as cc
        from agent.auxiliary_client import AuxiliaryExplicitCancellation
        from agent.conversation_compression import _raise_if_stale_attempt

        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        started = time.perf_counter()
        try:
            previous = self._previous_summary or ""
            pruned_skills = list(dict.fromkeys(
                cc._collect_ghosted_skill_names(turns_to_summarize)
                + cc._extract_pruned_skill_names(previous)
            ))[: cc._MAX_PRUNED_SKILL_MARKERS]
            has_user_turn = getattr(self, "_summary_has_user_turn", None)
            if has_user_turn is None:
                has_user_turn = self._transcript_has_real_user_turn(turns_to_summarize)

            body = projector.project(
                turns_to_summarize,
                budget_chars=self._vcc_budget_chars(turns_to_summarize),
                session_id=getattr(self, "_session_id", "") or "",
                previous=cc._redact_compaction_text(previous) if previous else None,
                is_real_user=self._vcc_is_real_user,
                focus=focus_topic,
            )
            if not has_user_turn:
                body = _force_no_user_task(body)

            # Same post-processing, same order, as ContextCompressor._generate_summary.
            summary = cc._redact_compaction_text(body)
            summary = cc._reinject_pruned_skill_markers(summary, pruned_skills)
            summary = self._ground_historical_task_snapshot(summary, turns_to_summarize)
            anchor_section = self._vcc_fast_anchor_section(cc, summary, turns_to_summarize)
            if anchor_section:
                # _augment_summary_lean skips any section whose heading is already present, so
                # this replaces only Hermes's slow anchor scan; the rest is still its own code.
                summary += cc._redact_compaction_text(anchor_section)
            summary = self._augment_summary_lean(summary, turns_to_summarize)
            self._validate_summary_user_provenance(summary, has_user_turn)
        except AuxiliaryExplicitCancellation:
            raise
        except Exception as exc:
            self._vcc_fallbacks += 1
            if not self._vcc_options["fallback_to_llm"]:
                logger.warning("vcc: projection failed (%s); fallback disabled, no summary this pass", exc)
                return None
            logger.warning("vcc: projection failed (%s: %s); using the stock LLM summary for this pass",
                           type(exc).__name__, exc)
            return super()._generate_summary(turns_to_summarize, focus_topic, memory_context, bypass_cooldown)

        # A detached stale attempt must not publish onto shared compressor state.
        _raise_if_stale_attempt(self)
        self._previous_summary = summary
        self._clear_compression_failure_cooldown()
        self._summary_model_fallen_back = False
        self._last_summary_error = None
        self._clear_terminal_summary_failures()
        self._reset_consecutive_overload_aborts()

        self._vcc_runs += 1
        self._vcc_last_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("vcc: projected %d turns into %d chars in %.1f ms (no LLM call)",
                    len(turns_to_summarize), len(summary), self._vcc_last_ms)
        return self._with_summary_prefix(summary)


def _force_no_user_task(body: str) -> str:
    """With no user-authored turn anywhere, the task section must be Hermes's exact sentinel."""
    preamble, sections = projector.split_sections(body)
    sections[projector.TASK_HEADING] = projector.NO_USER_TASK
    parts = [preamble] if preamble else []
    parts += [f"{heading}\n{text}".rstrip() for heading, text in sections.items()]
    return "\n\n".join(parts)
