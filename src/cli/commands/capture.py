from typing import List, Optional

import torch
import typer

from ...core.adapter import load_adapter
from ...core.config import Position
from ...core.metrics import measure
from ..common import HelpfulCommand, HelpfulGroup

"""
Read activations out of a model, and generate from it without intervention.
Both commands print a shape and a cost, because a capture you cannot afford
to run is not a capture.
"""

app = typer.Typer(help="Capture activations, and generate unsteered baselines.", cls=HelpfulGroup)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")

@app.command("run", cls=HelpfulCommand)
def run(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to capture; repeatable"),
    frac: List[float] = typer.Option([0.65], "--frac", help="Depth fraction(s) to capture at; repeatable"),
    position: Position = typer.Option(Position.LAST, help="Token position(s) kept from each prompt"),
):
    """Capture the residual stream and report its shape, cost and per-layer norms

    The norms are the cheapest sanity check there is: a layer that captured
    zeros, or one whose activations do not differ from an early layer's, means
    the hook is not where you think it is.
    """
    adapter = load_adapter(config)
    layers = adapter.cfg.layers(list(frac))
    with measure(items=len(prompt)) as cost:
        activations = adapter.capture(prompt, layers=layers, position=position)

    typer.echo(f"model: {adapter.cfg.id} ({adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model})")
    typer.echo(f"prompts: {len(prompt)}  layers: {layers}  position: {position.value}")
    typer.echo(f"shape: {tuple(activations.shape)}  finite: {bool(torch.isfinite(activations).all())}")
    typer.echo(f"cost: {cost[0]}")
    for index, resolved in enumerate(layers):
        typer.echo(f"layer {resolved:>3}: mean norm {activations[:, index].norm(dim=-1).mean():.3f}")

@app.command("generate", cls=HelpfulCommand)
def generate(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue; repeatable"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
):
    """Greedily continue each prompt, with no intervention applied"""
    adapter = load_adapter(config)
    for text, completion in zip(prompt, adapter.generate(prompt, max_new_tokens=max_new_tokens)):
        typer.echo(f"{text}{typer.style(completion, bold=True)}")
