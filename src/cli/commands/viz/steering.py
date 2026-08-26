from pathlib import Path
from typing import List, Optional

import typer

from ....methods.probing import LinearProbe, difference_of_means
from ....methods.steering import random_control, strength_sweep
from ....model.adapter import load_adapter
from ....viz import steering as steer_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    CONFIG_ARGUMENT,
    SHOW_OPTION,
    save_chart,
)

"""
The steering chart is a curve, never one generation at one strength.

Effect and fluency are drawn together because the ceiling is where they
cross, and the norm-matched random control is drawn beside them because a
direction that does not beat it has demonstrated nothing.

Run with: python -m src.cli viz steering <command> [options]
"""

app = typer.Typer(help="What steering did, as a curve rather than one generation.", cls=HelpfulGroup)

@app.command("sweep", cls=HelpfulCommand)
def steer_sweep(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue while steered; repeatable"),
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe to steer along and score with"),
    positive: List[str] = typer.Option([], "--positive", help="Prompt exhibiting the behaviour; repeatable"),
    negative: List[str] = typer.Option([], "--negative", help="Matched prompt lacking it; repeatable"),
    frac: float = typer.Option(0.65, help="Depth to read the direction from and steer at"),
    strength: List[float] = typer.Option([], "--strength", help="Strengths to try; defaults to 0 through 3"),
    control: bool = typer.Option(True, help="Also sweep a random vector of the same norm, as the control"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
    output: Path = typer.Option(Path("charts/steer-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot effect and fluency against steering strength, with a random control

    The ceiling is where effect is still climbing and fluency has started to
    fall, and one generation at one strength cannot show it. Give either a
    saved probe, or contrast pairs to build a direction from.
    """
    adapter = load_adapter(config)
    strengths = list(strength) or [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    if probe_path:
        probe = LinearProbe.load(probe_path)
        layer, direction = probe.layer, probe.direction.float()
    elif positive and negative:
        layer = adapter.cfg.layer(frac)
        captured = adapter.capture(list(positive) + list(negative), layers=[layer])
        probe = difference_of_means(captured, [1] * len(positive) + [0] * len(negative),
                                    layer=layer, model_id=adapter.cfg.id,
                                    n_layers=adapter.cfg.n_layers, dataset="contrast pairs")
        direction = probe.direction.float()
    else:
        raise typer.BadParameter("give either --probe, or both --positive and --negative")

    typer.echo(f"steering layer {layer} of {adapter.cfg.n_layers} over strengths {strengths}")
    sweeps = {
        "direction": strength_sweep(adapter, list(prompt), layer, direction, strengths,
                                    probe=probe, max_new_tokens=max_new_tokens),
    }
    if control:
        sweeps["random control"] = strength_sweep(
            adapter, list(prompt), layer, random_control(direction), strengths,
            probe=probe, max_new_tokens=max_new_tokens)

    for point in sweeps["direction"]:
        typer.echo(f"  strength {point.strength:>4.1f}  effect {point.effect:>7.3f}  "
                   f"fluency {point.fluency:.2f}  {point.completions[0][:60]!r}")

    if control:
        save_chart(steer_viz.plot_control_comparison(sweeps), output, show)
    else:
        save_chart(steer_viz.plot_strength_sweep(sweeps["direction"]), output, show)

