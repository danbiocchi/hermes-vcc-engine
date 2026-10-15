"""hermes-vcc-engine: a Hermes context engine that compacts without calling a model.

Enable with ``context.engine: vcc`` in config.yaml. See README.md.
"""

__version__ = "0.3.0"


def register(ctx) -> None:
    # Imported here so loading the package stays cheap and side-effect free.
    from .engine import VccEngine

    ctx.register_context_engine(VccEngine())
