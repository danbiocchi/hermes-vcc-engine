"""The ``/vcc`` chat command: switch vcc on or off for hosted models.

A local model always gets the vcc summary. For a hosted model (any chat model that is not a local
network endpoint, the Claude Code CLI bridge included), ``context.vcc.only_local_models`` in
config.yaml decides, unless someone typed ``/vcc on`` or ``/vcc off`` in a chat.

That choice is one switch per Hermes profile, kept in ``<HERMES_HOME>/vcc_state.json``. A file, so
it reaches every process that runs chats (WebUI, gateway, CLI) and survives restarts. It is not
per chat: Hermes's WebUI calls plugin commands without saying which chat they came from. Engines
read the file at each compaction check, so a change applies from the next compaction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

STATE_FILE = "vcc_state.json"
_ON, _OFF, _DEFAULT = ("on", "yes", "true", "1"), ("off", "no", "false", "0"), ("default", "reset", "config", "auto")

# (path, mtime_ns, size) -> override, so the per-check read is one stat() while nothing changes.
_cache: dict[str, Any] = {"key": None, "value": None}


def state_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:  # outside Hermes (tests, tooling)
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home / STATE_FILE


def hosted_override() -> Optional[bool]:
    """``True``/``False`` after ``/vcc on``/``/vcc off``; ``None`` means the config decides."""
    path = state_path()
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _cache["key"] == key:
        return _cache["value"]
    value: Optional[bool] = None
    try:
        raw = json.loads(path.read_text(encoding="utf-8")).get("hosted_models")
        value = raw if isinstance(raw, bool) else None
    except (OSError, ValueError, AttributeError) as exc:  # unreadable: the config decides
        logger.debug("vcc: ignoring %s (%s)", path, exc)
    _cache.update(key=key, value=value)
    return value


def set_hosted_override(value: Optional[bool]) -> None:
    """Store the switch (``None`` clears it). Written atomically: a reader never sees half a file."""
    path = state_path()
    if value is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".vcc_state.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"hosted_models": value}, fh)
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def load_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return cfg if isinstance(cfg, Mapping) else {}
    except Exception as exc:  # config is optional; defaults are fine
        logger.debug("vcc: could not read Hermes config (%s); using defaults", exc)
        return {}


def config_default(cfg: Mapping[str, Any]) -> bool:
    """What hosted models get without a ``/vcc`` switch: vcc unless ``only_local_models`` is on."""
    section = cfg.get("context") if isinstance(cfg, Mapping) else None
    vcc = section.get("vcc") if isinstance(section, Mapping) else None
    only_local = bool(vcc.get("only_local_models", False)) if isinstance(vcc, Mapping) else False
    return not only_local


def hosted_mode(cfg: Mapping[str, Any]) -> tuple[bool, str]:
    """``(vcc on for hosted models?, where that comes from)``."""
    override = hosted_override()
    if override is None:
        return config_default(cfg), "config"
    return override, "/vcc"


def _word(on: bool) -> str:
    return "on" if on else "off"


def _status(cfg: Mapping[str, Any]) -> str:
    on, source = hosted_mode(cfg)
    default = config_default(cfg)
    lines = [
        f"vcc for hosted models (Claude, OpenRouter and other non-local models): {_word(on).upper()}"
        + (" (set with /vcc)" if source == "/vcc" else " (from config)"),
        "Local models always use vcc.",
        f"Config default for hosted models: {_word(default)}"
        + (" (context.vcc.only_local_models: true)" if not default else ""),
        "Off means Hermes writes the summary with its usual LLM call. "
        "Use /vcc on, /vcc off, or /vcc default. It applies to every chat from the next compaction.",
    ]
    return "\n".join(lines)


def handle(raw_args: str = "", **_context: Any) -> str:
    """``/vcc [on|off|default|status]``."""
    cfg = load_config()
    arg = (raw_args or "").strip().lower()
    if arg in ("", "status", "?", "help"):
        return _status(cfg)
    if arg in _ON + _OFF + _DEFAULT:
        value = True if arg in _ON else False if arg in _OFF else None
        try:
            set_hosted_override(value)
        except OSError as exc:
            logger.warning("vcc: could not save the /vcc switch to %s: %s", state_path(), exc)
            return f"vcc: could not save the switch ({type(exc).__name__}); nothing changed."
        logger.info("vcc: hosted models switched %s by /vcc",
                    "back to the config" if value is None else _word(value))
        return _status(cfg)
    return "Usage: /vcc [on|off|default]. Without an argument it shows the current setting.\n\n" + _status(cfg)
