from typing import List, Optional

import typer

from ...methods.probing import difference_of_means
from ...model.adapter import load_adapter
from ...share.loaders import open_probe
from ..common import HelpfulCommand, HelpfulGroup

"""
Intervene on generation by adding a direction to the residual stream, and show
the unsteered continuation next to it. A steering result you cannot see the
baseline of is not a result.
"""

app = typer.Typer(help="Steer generation along a direction read off the model.", cls=HelpfulGroup)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")

@app.command("contrast", cls=HelpfulCommand)
def contrast(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue while steered; repeatable"),
    positive: List[str] = typer.Option(..., "--positive", help="Prompt exhibiting the behaviour; repeatable"),
    negative: List[str] = typer.Option(..., "--negative", help="Matched prompt lacking it; repeatable"),
    frac: float = typer.Option(0.65, help="Depth fraction to read the direction from and steer at"),
    strength: float = typer.Option(1.0, help="Intervention size, in mean activation norms"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
):
    """Steer along the difference-of-means direction between two prompt sets

    The direction is read at the same layer it is injected into, which is the
    cheapest steering vector that works and the one to beat before reaching for
    an SAE feature. Strength is measured in mean activation norms, so the same
    number means the same intervention on any model.
    """
    adapter = load_adapter(config)
    resolved = adapter.cfg.layer(frac)
    activations = adapter.capture(list(positive) + list(negative), layers=[resolved])
    labels = [1] * len(positive) + [0] * len(negative)
    direction = difference_of_means(
        activations, labels, layer=resolved, model_id=adapter.cfg.id,
        n_layers=adapter.cfg.n_layers, dataset="contrast pairs",
    ).direction.float()

    typer.echo(f"steering layer {resolved} of {adapter.cfg.n_layers} at strength {strength}")
    baseline = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    with adapter.steer(resolved, direction, strength):
        steered = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    for text, before, after in zip(prompt, baseline, steered, strict=True):
        typer.echo(f"\n{text}")
        typer.echo(f"  baseline: {before}")
        typer.echo(f"  steered : {typer.style(after, bold=True)}")

@app.command("probe", cls=HelpfulCommand)
def from_probe(
    config: str = CONFIG_ARGUMENT,
    probe_path: str = typer.Option(..., "--probe", help="Path to a probe .pt, or to a shared .mia artifact"),
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue while steered; repeatable"),
    strength: float = typer.Option(2.0, help="Intervention size, in mean activation norms"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
):
    """Steer along a trained probe's direction, at the layer it was trained on

    A probe and a steering vector are the same object seen from two sides: the
    direction that reads a property off the residual stream is the direction
    that writes it back in. It is the probe's direction in activation space
    that gets injected, not its raw weight vector -- see LinearProbe.direction
    for why those are not the same thing.
    """
    adapter = load_adapter(config)
    probe = open_probe(probe_path)
    if probe.model_id != adapter.cfg.id:
        typer.echo(f"warning: probe was trained on '{probe.model_id}', steering '{adapter.cfg.id}'", err=True)

    if probe.method == "logistic":
        typer.echo(
            "note: a logistic probe is a discriminative direction and usually steers badly even when "
            "its AUC is the higher one; train with --method difference_of_means to steer",
            err=True,
        )
    typer.echo(f"steering layer {probe.layer} of {adapter.cfg.n_layers} at strength {strength}")
    baseline = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    with adapter.steer(probe.layer, probe.direction.float(), strength):
        steered = adapter.generate(prompt, max_new_tokens=max_new_tokens)
    for text, before, after in zip(prompt, baseline, steered, strict=True):
        typer.echo(f"\n{text}")
        typer.echo(f"  baseline: {before}")
        typer.echo(f"  steered : {typer.style(after, bold=True)}")
