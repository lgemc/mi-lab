from pathlib import Path
from typing import List

import typer

from ....core.config import presets
from ....viz import model as model_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    SHOW_OPTION,
    fracs_or_default,
    resolve_configs,
    save_chart,
)

"""
Charts of what a config resolves to: where a depth fraction lands, and how
two models compare in shape.

This is the group that makes 'layer 8 of a small model and layer 42 of a large
one are the same place' something you can look at rather than take on trust.

Run with: python -m src.cli viz model <command> [options]
"""

app = typer.Typer(help="What a config resolves to, and how models compare in shape.", cls=HelpfulGroup)

@app.command("ruler", cls=HelpfulCommand)
def model_ruler(
    config: List[str] = typer.Option([], "--config", "-c", help="Config to include; repeatable, defaults to all presets"),
    frac: List[float] = typer.Option([], "--frac", help="Depth fraction(s) to mark; defaults to six"),
    load: bool = typer.Option(True, help="Load checkpoints to resolve layer counts the configs do not state"),
    output: Path = typer.Option(Path("charts/depth-ruler.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the depth ruler: one fraction, one line, a different layer index per model

    The chart this framework exists to make. Every model gets a lane, every
    fraction is one vertical line, and the integers underneath are what a
    hardcoded layer index would have got wrong.
    """
    configs = resolve_configs(config or presets(), load)
    save_chart(model_viz.plot_depth_ruler(configs, fracs=fracs_or_default(frac)), output, show)

@app.command("shape", cls=HelpfulCommand)
def model_shape(
    config: List[str] = typer.Option([], "--config", "-c", help="Config to include; repeatable, defaults to all presets"),
    load: bool = typer.Option(True, help="Load checkpoints to resolve sizes the configs do not state"),
    output: Path = typer.Option(Path("charts/model-shape.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot residual stream width against depth, one point per model"""
    configs = resolve_configs(config or presets(), load)
    save_chart(model_viz.plot_shape(configs), output, show)

