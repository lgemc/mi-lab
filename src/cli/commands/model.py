from typing import List

import typer

from ...core.adapter import load_adapter
from ...core.config import load_config, presets
from ..common import HelpfulCommand, HelpfulGroup

"""
Inspect what a config resolves to: which checkpoint, and which absolute layer
a depth fraction lands on for that checkpoint.
"""

app = typer.Typer(help="Inspect configs and the layers their fractions resolve to.", cls=HelpfulGroup)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")

@app.command("list", cls=HelpfulCommand)
def list_configs():
    """List the configs shipped in configs/, without loading any checkpoint"""
    for name in presets():
        cfg = load_config(name)
        typer.echo(f"{name:<14} backend={cfg.backend:<14} {cfg.hf_name:<22} depth {cfg.probe_layer_frac}")

@app.command("info", cls=HelpfulCommand)
def info(
    config: str = CONFIG_ARGUMENT,
    load: bool = typer.Option(True, help="Load the checkpoint to resolve its sizes"),
):
    """Show a config, and the layer its probe fraction resolves to

    With --no-load nothing is downloaded, so any field the config does not
    state itself stays unknown.
    """
    cfg = load_adapter(config).cfg if load else load_config(config)
    for key, value in cfg.as_dict().items():
        typer.echo(f"{key}: {value}")
    if cfg.is_resolved:
        typer.echo(f"probe layer: {cfg.layer()} of {cfg.n_layers} (depth {cfg.probe_layer_frac})")
        typer.echo(f"depth sweep: {cfg.sweep(5)}")

@app.command("layer", cls=HelpfulCommand)
def layer(
    config: str = CONFIG_ARGUMENT,
    frac: List[float] = typer.Option([0.65], "--frac", help="Depth fraction(s) in [0, 1] to resolve; repeatable"),
):
    """Resolve depth fractions to absolute layer indices for this model"""
    cfg = load_adapter(config).cfg
    for fraction in frac:
        typer.echo(f"{fraction} -> layer {cfg.layer(fraction)} of {cfg.n_layers}")
