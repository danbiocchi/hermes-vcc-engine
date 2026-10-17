"""Test setup.

Projector tests need nothing but this repo. Engine tests need a Hermes checkout: point
HERMES_AGENT_DIR at one (defaults to ~/.hermes/hermes-agent) and run with the same Python
Hermes uses. Without it they skip.

HERMES_HOME is pointed at a throwaway directory before Hermes is imported, so tests never
read or write a real profile.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vcc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERMES_DIR = Path(os.environ.get("HERMES_AGENT_DIR", Path.home() / ".hermes" / "hermes-agent")).expanduser()


def _hermes_site_packages() -> list[str]:
    """Ask Hermes's own interpreter which dependency tree it activates (read-only)."""
    if os.environ.get("HERMES_SITE_PACKAGES"):
        return os.environ["HERMES_SITE_PACKAGES"].split(os.pathsep)
    candidates = [os.environ.get("HERMES_PYTHON", "")]
    candidates += sorted(glob.glob(str(Path.home() / ".hermes/tools/python-*/bin/python3")), reverse=True)
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import hermes_bootstrap; "
        "print('\\n'.join(p for p in sys.path if p.endswith('site-packages')))"
    )
    for python in filter(None, candidates):
        try:
            out = subprocess.run([python, "-I", "-c", probe, str(HERMES_DIR)], capture_output=True,
                                 text=True, timeout=60, cwd=tempfile.gettempdir())
        except (OSError, subprocess.SubprocessError):
            continue
        paths = [line for line in out.stdout.splitlines() if line.strip()]
        if out.returncode == 0 and paths:
            return paths
    return []


HAVE_HERMES = False
if (HERMES_DIR / "agent" / "context_compressor.py").is_file():
    _site = _hermes_site_packages()
    # Sandbox the profile only after the probe, which needs the real install to resolve deps.
    _SANDBOX = Path(tempfile.mkdtemp(prefix="vcc-test-home-"))
    os.environ["HERMES_HOME"] = str(_SANDBOX)
    sys.path.insert(0, str(HERMES_DIR))
    sys.path.extend(p for p in _site if p not in sys.path)
    # Dependencies are already on sys.path. Importing the real bootstrap here would see the
    # sandboxed profile, provision a fresh runtime into it and re-exec the interpreter.
    import socket
    import types

    _stub = types.ModuleType("hermes_bootstrap")
    _stub._happy_eyeballs_create_connection = socket.create_connection  # the one name core imports
    sys.modules.setdefault("hermes_bootstrap", _stub)
    try:
        import agent.context_compressor  # noqa: F401

        HAVE_HERMES = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[vcc tests] Hermes not importable from {HERMES_DIR}: {exc}", file=sys.stderr)
else:
    _SANDBOX = Path(tempfile.mkdtemp(prefix="vcc-test-home-"))
    os.environ["HERMES_HOME"] = str(_SANDBOX)

requires_hermes = pytest.mark.skipif(not HAVE_HERMES, reason="needs a Hermes checkout (set HERMES_AGENT_DIR)")


@pytest.fixture()
def sandbox_home() -> Path:
    return _SANDBOX
