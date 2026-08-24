from typing import List, Optional

import torch
import typer

from ..core.adapter import load_adapter
from ..core.config import Position, load_config, presets as shipped_configs
from .common import CONTEXT_SETTINGS, HelpfulCommand, HelpfulGroup

"""
Root Typer application for the lab. Commands here only resolve a config, ask
core for a tensor, and format what comes back -- no experiment logic lives in
the CLI, so anything you can do from the shell you can also do from a notebook
by calling the same core functions.

Every command takes the same first argument: a config, given either as a
preset name or as a path to a YAML/JSON file. That argument is the only thing
that changes when you move an experiment from a laptop model to a large one.

Run with: python -m src.cli <command> [options]
"""

app = typer.Typer(
    help="A model-agnostic mechanistic interpretability lab: capture, steer, inspect.",
    no_args_is_help=True,
    cls=HelpfulGroup,
    context_settings=CONTEXT_SETTINGS,
)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")

@app.command("configs", cls=HelpfulCommand)
def configs():
    """List the configs shipped in configs/, without loading any checkpoint"""
    for name in shipped_configs():
        cfg = load_config(name)
        typer.echo(f"{name:<14} backend={cfg.backend:<14} {cfg.hf_name:<22} depth {cfg.probe_layer_frac}")

@app.command("info", cls=HelpfulCommand)
def info(config: str = CONFIG_ARGUMENT, load: bool = typer.Option(True, help="Load the checkpoint to resolve its sizes")):
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

@app.command("capture", cls=HelpfulCommand)
def capture(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to capture; repeatable"),
    frac: List[float] = typer.Option([0.65], "--frac", help="Depth fraction(s) to capture at; repeatable"),
    position: Position = typer.Option(Position.LAST, help="Token position(s) kept from each prompt"),
):
    """Capture the residual stream and report its shape and per-layer norms

    The norms are the cheapest sanity check there is: a layer that captured
    zeros, or one whose activations do not differ from an early layer's, means
    the hook is not where you think it is.
    """
    adapter = load_adapter(config)
    layers = adapter.cfg.layers(list(frac))
    activations = adapter.capture(prompt, layers=layers, position=position)

    typer.echo(f"model: {adapter.cfg.id} ({adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model})")
    typer.echo(f"prompts: {len(prompt)}  layers: {layers}  position: {position.value}")
    typer.echo(f"shape: {tuple(activations.shape)}  finite: {bool(torch.isfinite(activations).all())}")
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

@app.command("steer", cls=HelpfulCommand)
def steer(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue while steered; repeatable"),
    positive: List[str] = typer.Option(..., "--positive", help="Prompt exhibiting the behaviour; repeatable"),
    negative: List[str] = typer.Option(..., "--negative", help="Matched prompt lacking it; repeatable"),
    frac: float = typer.Option(0.65, help="Depth fraction to read the direction from and steer at"),
    strength: float = typer.Option(2.0, help="Intervention size, in mean activation norms"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
):
    """Steer generation along the difference-of-means direction between two prompt sets

    The direction is read at the same layer it is injected into, which is the
    cheapest steering vector that works and the one to beat before reaching for
    an SAE feature. Strength is measured in mean activation norms, so the same
    number means the same intervention on any model.
    """
    adapter = load_adapter(config)
    resolved = adapter.cfg.layer(frac)
    direction = (
        adapter.capture(positive, layers=[resolved]).mean(dim=0)
        - adapter.capture(negative, layers=[resolved]).mean(dim=0)
    )[0]

    typer.echo(f"steering layer {resolved} of {adapter.cfg.n_layers} at strength {strength}")
    baseline = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    with adapter.steer(resolved, direction, strength):
        steered = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    for text, before, after in zip(prompt, baseline, steered):
        typer.echo(f"\n{text}")
        typer.echo(f"  baseline: {before}")
        typer.echo(f"  steered : {typer.style(after, bold=True)}")

if __name__ == "__main__":
    app()
