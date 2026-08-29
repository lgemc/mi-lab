from pathlib import Path
from typing import List, Optional

import typer

from ....core.config import Position
from ....core.metrics import measure
from ....model.adapter import load_adapter
from ....viz import activations as act_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    CONFIG_ARGUMENT,
    DATA_OPTION,
    FRAC_OPTION,
    SEED_OPTION,
    SHOW_OPTION,
    fracs_or_default,
    label_names,
    load_dataset,
    prompts_or_dataset,
    save_chart,
    tokenizer_of,
)

"""
Charts of what came out of a capture: how big the residual stream is by
layer, how the classes sit apart in it, and how far two models drift.

The norms chart is the cheapest check that a hook is where you think it is.
A curve that does not grow with depth means the capture is reading something
other than the residual stream.

Run with: python -m src.cli viz activations <command> [options]
"""

app = typer.Typer(help="What came out of capture: norms, geometry, drift.", cls=HelpfulGroup)

@app.command("norms", cls=HelpfulCommand)
def act_norms(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(
        [], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"
    ),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-norms.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot residual stream norm by layer: the cheapest check that the hook is where you think

    Norms should climb with depth. A layer at zero captured nothing, and a
    flat line means every layer captured the same thing.
    """
    adapter = load_adapter(config)
    texts = prompts_or_dataset(adapter, prompt, data, size, seed)
    layers = adapter.cfg.layers(fracs_or_default(frac) or [index / 8 for index in range(9)])
    with measure(items=len(texts)) as cost:
        captured = adapter.capture(texts, layers=layers)
    typer.echo(f"captured {tuple(captured.shape)} at layers {layers}  ({cost[0]})")
    save_chart(act_viz.plot_layer_norms(captured, layers=layers), output, show)

@app.command("heatmap", cls=HelpfulCommand)
def act_heatmap(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to capture; repeatable"),
    example: int = typer.Option(0, help="Which prompt to draw"),
    dims: int = typer.Option(128, help="How many residual stream dimensions to show"),
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-heatmap.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot one prompt's residual stream as layers by dimensions

    A vertical stripe is an outlier dimension that fires at every depth, and
    those are the ones that dominate any unnormalized distance.
    """
    adapter = load_adapter(config)
    layers = adapter.cfg.layers(fracs_or_default(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture(list(prompt), layers=layers)
    save_chart(act_viz.plot_activation_heatmap(captured, layers=layers, example=example, dims=dims), output, show)

@app.command("tokens", cls=HelpfulCommand)
def act_tokens(
    config: str = CONFIG_ARGUMENT,
    prompt: str = typer.Option(..., "--prompt", "-p", help="The single prompt to draw"),
    frac: List[float] = FRAC_OPTION,
    normalize: str = typer.Option("layer", help="'layer' scales each row to its own max; 'none' shows raw norms"),
    output: Path = typer.Option(Path("charts/act-tokens.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot norm at every (layer, token position): where in the sequence the work happens

    Captures with position=all, which is the only way to see anything other
    than what the model ended up with at the last token.
    """
    adapter = load_adapter(config)
    tokenizer = tokenizer_of(adapter)
    layers = adapter.cfg.layers(fracs_or_default(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture([prompt], layers=layers, position=Position.ALL)
    tokens = [tokenizer.decode([identifier]).strip() or "_" for identifier in tokenizer(prompt)["input_ids"]]
    save_chart(act_viz.plot_token_layer_norms(captured, tokens, layers=layers, normalize=normalize), output, show)

@app.command("pca", cls=HelpfulCommand)
def act_pca(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(120, "--size", help="Examples to generate when using the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-pca.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot class separation in the top two components at every layer

    The probe sweep before the probe: the layer where the colours come apart
    is the layer the sweep will pick.
    """
    adapter = load_adapter(config)
    loaded = load_dataset(data, size, seed)
    layers = adapter.cfg.layers(fracs_or_default(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture(loaded.texts, layers=layers)
    figure = act_viz.plot_layer_pca(captured, loaded.labels, layers=layers, label_names=label_names(loaded))
    save_chart(figure, output, show)

@app.command("similarity", cls=HelpfulCommand)
def act_similarity(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(
        [], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"
    ),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    output: Path = typer.Option(Path("charts/act-similarity.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot cosine similarity between every pair of layers: how fast the stream turns over

    Captures every layer of the model, not a sweep, since the point is the
    shape of the whole band.
    """
    adapter = load_adapter(config)
    texts = prompts_or_dataset(adapter, prompt, data, size, seed)
    layers = list(range(adapter.cfg.n_layers))
    captured = adapter.capture(texts, layers=layers)
    save_chart(act_viz.plot_layer_similarity(captured, layers=layers), output, show)

@app.command("drift", cls=HelpfulCommand)
def act_drift(
    config: str = CONFIG_ARGUMENT,
    against: str = typer.Option(
        ..., "--against", help="The config to compare it with, same prompts, same layers"
    ),
    prompt: List[str] = typer.Option(
        [], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"
    ),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-drift.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot per-layer agreement between two models' activations on the same prompts

    Cosine and relative L2 together, because a quantization can leave a
    direction untouched while changing its magnitude, and those are different
    findings. The two configs must have the same depth for the comparison to
    mean anything.
    """
    reference = load_adapter(config)
    compared = load_adapter(against)
    if reference.cfg.n_layers != compared.cfg.n_layers:
        raise typer.BadParameter(
            f"'{reference.cfg.id}' has {reference.cfg.n_layers} layers and "
            f"'{compared.cfg.id}' has {compared.cfg.n_layers}; drift is only defined layer-for-layer"
        )
    texts = prompts_or_dataset(reference, prompt, data, size, seed)
    fracs = fracs_or_default(frac) or [index / 8 for index in range(9)]
    layers = reference.cfg.layers(fracs)
    figure = act_viz.plot_drift(
        reference.capture(texts, layers=layers),
        compared.capture(texts, layers=layers),
        layers=layers,
        reference_name=reference.cfg.id,
        other_name=compared.cfg.id,
    )
    save_chart(figure, output, show)

