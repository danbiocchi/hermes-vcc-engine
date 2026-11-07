"""hermes-vcc-engine: a Hermes context engine that compacts without calling a model.

Enable with ``context.engine: vcc`` in config.yaml. See README.md.
"""

__version__ = "0.4.5"


def register(ctx) -> None:
    # Imported here so loading the package stays cheap and side-effect free.
    from .vcc import toggle
    from .vcc.engine import VccEngine

    ctx.register_context_engine(VccEngine())
    _register_command(ctx, toggle.handle)


def _register_command(ctx, handler) -> None:
    """``/vcc``: switch vcc on or off for hosted models from any chat (see vcc/toggle.py)."""
    register_command = getattr(ctx, "register_command", None)
    if not callable(register_command):
        return  # a Hermes without plugin commands: the config option still works
    try:  # a second load (e.g. Hermes probing installed engines) must not warn about a duplicate
        from hermes_cli.plugins import get_plugin_commands
        if "vcc" in (get_plugin_commands() or {}):
            return
    except Exception:
        pass
    register_command("vcc", handler,
                     description="Turn vcc on or off for hosted models (local models always use it)",
                     args_hint="[on|off|default]")
