from pathlib import Path
from typing import Optional

import typer

from ....core.adapter import load_adapter, require_circuits
from ....core.circuits import classify_heads, direct_logit_attribution, discover, patch_heads, patch_residual, verify
from ....core.ioi import CORRUPTIONS, FRAMES, build_ioi
from ....viz import circuits as circuit_viz
from ....viz.dashboard import Panel, Section, render
from ....viz.style import save_figure
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    CONFIG_ARGUMENT,
    SEED_OPTION,
    SHOW_OPTION,
    save_chart,
)

"""
Charts of the circuit study: what each head writes towards the answer, what
each head causes, where the answer gets decided, and whether the set found
survives being checked.

`dashboard` measures every panel off one dataset and one pair of baselines,
which is what stops being true the moment the charts are made one command at
a time.

Run with: python -m src.cli viz circuit <command> [options]
"""

app = typer.Typer(help="Which heads do the task, correlationally and causally.", cls=HelpfulGroup)

IOI_SIZE_OPTION = typer.Option(16, "--size", help="Clean/corrupted prompt pairs to build")
FRAME_OPTION = typer.Option(0, "--frame", help=f"Which of the {len(FRAMES)} sentence frames to use")
CORRUPTION_OPTION = typer.Option(
    "abc", "--corruption", help=f"How the clean prompt is broken; one of {sorted(CORRUPTIONS)}"
)

def _ioi(config: str, size: int, seed: int, frame: int, corruption: str):
    """Load a circuit-capable adapter and build the IOI dataset every chart here starts from"""
    adapter = require_circuits(load_adapter(config))
    dataset = build_ioi(adapter, size=size, seed=seed, frame=frame, corruption=corruption)
    typer.echo(f"{dataset.name}: {len(dataset)} pairs on {adapter.cfg.id}, ABBA share {dataset.balance:.0%}")
    return adapter, dataset

@app.command("attribution", cls=HelpfulCommand)
def circuit_attribution(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    output: Path = typer.Option(Path("charts/circuit-attribution.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw what each head wrote directly towards the answer, as a layer-by-head grid"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    save_chart(circuit_viz.plot_attribution(direct_logit_attribution(adapter, dataset)), output, show)

@app.command("patch", cls=HelpfulCommand)
def circuit_patch(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    site: str = typer.Option(
        "heads", "--site", help="'heads' for a layer-by-head grid, 'residual' for layer by position"
    ),
    output: Optional[Path] = typer.Option(None, help="Where to save the chart; defaults to the site's own name"),
    show: bool = SHOW_OPTION,
):
    """Draw what patching recovered, either per head or across the sentence

    'residual' is the chart that tells the story: the signal sits on the
    repeated name early and moves to the end of the sentence partway up.
    """
    if site not in ("heads", "residual"):
        raise typer.BadParameter(f"unknown site '{site}'; use 'heads' or 'residual'")
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    target = output or Path(f"charts/circuit-patch-{site}.png")
    if site == "heads":
        save_chart(circuit_viz.plot_head_effects(patch_heads(adapter, dataset)), target, show)
    else:
        save_chart(circuit_viz.plot_patch_grid(patch_residual(adapter, dataset)), target, show)

@app.command("roles", cls=HelpfulCommand)
def circuit_roles(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.3, help="Attention weight a head needs before it gets named"),
    output: Path = typer.Option(Path("charts/circuit-roles.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw how much attention each head spends on each of the task's four movements"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    save_chart(circuit_viz.plot_roles(classify_heads(adapter, dataset), threshold=threshold), output, show)

@app.command("compare", cls=HelpfulCommand)
def circuit_compare(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    output: Path = typer.Option(Path("charts/circuit-compare.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw direct attribution against causal effect, one point per head

    The chart the rest of this group exists to set up: the heads off the
    diagonal are the ones neither method would have found alone.
    """
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    attribution = direct_logit_attribution(adapter, dataset)
    effects = patch_heads(adapter, dataset)
    labelled = dict(effects.ranked(4))
    save_chart(circuit_viz.plot_attribution_against_effect(attribution, effects, labels=labelled), output, show)

@app.command("verify", cls=HelpfulCommand)
def circuit_verify(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.8, help="Recovery the growing circuit has to reach before the search stops"),
    max_heads: int = typer.Option(12, help="Most heads the search is allowed to add"),
    output: Path = typer.Option(Path("charts/circuit-verify.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Grow a circuit and draw whether it is enough, needed, and free of passengers"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    found = discover(adapter, dataset, threshold=threshold, max_heads=max_heads)
    typer.echo(f"  {found}")
    save_chart(circuit_viz.plot_verification(verify(adapter, dataset, found)), output, show)

@app.command("dashboard", cls=HelpfulCommand)
def circuit_dashboard(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.8, help="Recovery the growing circuit has to reach before the search stops"),
    max_heads: int = typer.Option(12, help="Most heads the search is allowed to add"),
    out_dir: Path = typer.Option(Path("charts"), "--out-dir", help="Where the PNGs and the HTML page go"),
    show: bool = typer.Option(False, "--show", help="Print the path when finished"),
):
    """Run the whole IOI battery once and assemble it into one HTML page

    Everything measured off a single dataset and a single set of baselines, so
    the charts on the page are comparable with each other -- which is the
    thing that stops being true the moment they are made one command at a time.
    """
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(name: str, target) -> Path:
        path = out_dir / f"{name}.png"
        save_figure(target, path)
        typer.echo(f"  drew {path}")
        return path

    typer.echo("behaviour and attribution...")
    attribution = direct_logit_attribution(adapter, dataset)
    roles = classify_heads(adapter, dataset)
    behaviour = Section("What each head writes", [
        Panel("Direct logit attribution", "Exact, cheap, and blind to everything but the direct path to the logits.",
              draw("circuit-attribution", circuit_viz.plot_attribution(attribution))),
        Panel("Strongest writers", "The same grid as a ranking, for the heads worth naming.",
              draw("circuit-top-heads", circuit_viz.plot_top_heads(
                  attribution.top(6) + attribution.top(4, negative=True), legend="logits towards the answer"))),
        Panel("Where each head looks", "Attention names candidates; only patching promotes one to a finding.",
              draw("circuit-roles", circuit_viz.plot_roles(roles)), wide=True),
    ])

    typer.echo("patching (one forward pass per site)...")
    effects = patch_heads(adapter, dataset)
    grid = patch_residual(adapter, dataset)
    causal = Section("What each head causes", [
        Panel("Head patching", "Restore one head from the clean run and see how much of the answer returns.",
              draw("circuit-patch-heads", circuit_viz.plot_head_effects(effects))),
        Panel("Written against caused", "The heads off the diagonal are the ones neither method finds alone.",
              draw("circuit-compare", circuit_viz.plot_attribution_against_effect(
                  attribution, effects, labels=dict(effects.ranked(4))))),
        Panel("Residual stream by position", "Where in the sentence, and at what depth, the answer gets decided.",
              draw("circuit-patch-residual", circuit_viz.plot_patch_grid(grid)), wide=True),
    ])

    typer.echo("circuit...")
    found = discover(adapter, dataset, threshold=threshold, max_heads=max_heads, effects=effects)
    report = verify(adapter, dataset, found)
    typer.echo(f"  {report}")
    checked = Section("The circuit", [
        Panel("Growth", "A curve that jumps and flattens found a circuit; one that climbs evenly did not.",
              draw("circuit-growth", circuit_viz.plot_circuit_growth(found))),
        Panel("Verification", "Enough on its own, needed by the clean run, and free of passengers.",
              draw("circuit-verify", circuit_viz.plot_verification(report))),
    ])

    page = render(
        [behaviour, causal, checked],
        out_dir / "circuit.html",
        title=f"The IOI circuit in {adapter.cfg.id}",
        subtitle="Which heads identify the indirect object, asked correlationally and then causally.",
        provenance=[
            f"{len(dataset)} clean/corrupted pairs, '{corruption}' corruption, seed {seed}",
            f"logit difference {report.baselines.clean:+.3f} clean / {report.baselines.corrupted:+.3f} corrupted",
            f"circuit: {found}",
            f"faithfulness {report.faithfulness:.2f}, necessity {report.necessity:.2f}",
        ],
    )
    typer.echo(f"Wrote {page}")
    if show:
        typer.echo(f"open {page.resolve()}")

