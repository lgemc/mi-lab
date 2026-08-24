import typer

from .commands import capture, data, model, probe, run, steer, viz
from .common import CONTEXT_SETTINGS, HelpfulGroup

"""
Root Typer application aggregating every command group. Each group lives in
its own module under commands/ and only formats what core returns, so anything
you can do from the shell you can also do from a notebook by calling the same
core functions.

Every command takes the same first argument: a config, given either as a name
in configs/ or as a path. That argument is the only thing that changes when an
experiment moves from a laptop model to a large one.

Run with: python -m src.cli <group> <command> [options]
"""

app = typer.Typer(
    help="A model-agnostic mechanistic interpretability lab: capture, probe, steer.",
    no_args_is_help=True,
    cls=HelpfulGroup,
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(model.app, name="model")
app.add_typer(capture.app, name="capture")
app.add_typer(data.app, name="data")
app.add_typer(probe.app, name="probe")
app.add_typer(steer.app, name="steer")
app.add_typer(run.app, name="run")
app.add_typer(viz.app, name="viz")

if __name__ == "__main__":
    app()
