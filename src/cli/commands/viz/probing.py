from pathlib import Path
from typing import List, Optional, Tuple

import typer

from ....methods.probing import LinearProbe, difference_of_means, evaluate, measure_scoring_cost, sweep, train_probe
from ....model.adapter import load_adapter
from ....viz import probing as probe_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    CONFIG_ARGUMENT,
    DATA_OPTION,
    FRAC_OPTION,
    SEED_OPTION,
    SHOW_OPTION,
    SIZE_OPTION,
    fracs_or_default,
    label_names,
    load_dataset,
    save_chart,
)

"""
Charts of what the probe found and what it cost to run.

The sweep is the one to read first: which layer wins is a fact about this
model, and that the winner sits in the middle rather than at either end is
the part expected to survive a model swap.

run_sweep and probe_at are shared with the dashboard, which fits the same
probes to put them on one page.

Run with: python -m src.cli viz probing <command> [options]
"""

app = typer.Typer(help="What the probe found, and what it cost.", cls=HelpfulGroup)

def run_sweep(adapter, loaded, test_frac: float, seed: int, fracs, method: str):
    """Run one depth sweep and hand back its reports"""
    train_set, test_set = loaded.split(test_frac=test_frac, seed=seed)
    return sweep(adapter, train_set, test_set, fracs=fracs, method=method,
                 **({"seed": seed} if method == "logistic" else {}))

def probe_at(adapter, loaded, frac: float, test_frac: float, seed: int, method: str):
    """Fit one probe at one depth and return it with the held-out activations"""
    train_set, test_set = loaded.split(test_frac=test_frac, seed=seed)
    layer = adapter.cfg.layer(frac)
    provenance = {"model_id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers, "layer": layer, "dataset": loaded.name}
    train_activations = adapter.capture(train_set.texts, layers=[layer])
    test_activations = adapter.capture(test_set.texts, layers=[layer])
    fit = difference_of_means if method == "difference_of_means" else train_probe
    extra = {} if method == "difference_of_means" else {"seed": seed}
    probe = fit(train_activations, train_set.labels, **provenance, **extra)
    probe.metrics.update(evaluate(probe, test_activations, test_set.labels))
    return probe, test_activations, test_set

def _load_or_fit(adapter, probe_path, data, size, seed, frac, test_frac, method):
    """Use a saved probe if one was given, otherwise fit a fresh one at --frac"""
    loaded = load_dataset(data, size, seed)
    if probe_path:
        probe = LinearProbe.load(probe_path)
        if probe.model_id != adapter.cfg.id:
            typer.echo(f"warning: probe was trained on '{probe.model_id}', scoring '{adapter.cfg.id}'", err=True)
        _, test_set = loaded.split(test_frac=test_frac, seed=seed)
        return probe, adapter.capture(test_set.texts, layers=[probe.layer]), test_set, loaded
    probe, test_activations, test_set = probe_at(adapter, loaded, frac, test_frac, seed, method)
    return probe, test_activations, test_set, loaded

@app.command("sweep", cls=HelpfulCommand)
def probe_sweep(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out for the reported numbers"),
    frac: List[float] = FRAC_OPTION,
    method: List[str] = typer.Option(["logistic"], "--method", help="Probe method; repeat it to compare methods"),
    metric: str = typer.Option("auc", help="'auc' or 'accuracy'"),
    output: Path = typer.Option(Path("charts/probe-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot probe score by depth, one line per method

    Drawn against depth rather than layer index, so two models land on the
    same axis. Pass --method twice to put logistic and difference-of-means on
    one chart, which is what makes 'the best reader is the worst writer' a
    picture instead of a claim.
    """
    adapter = load_adapter(config)
    loaded = load_dataset(data, size, seed)
    fracs = fracs_or_default(frac)
    sweeps = {name: run_sweep(adapter, loaded, test_frac, seed, fracs, name) for name in method}
    for name, reports in sweeps.items():
        best = max(reports, key=lambda report: report.auc)
        typer.echo(f"{name:<20} best layer {best.layer} at depth {best.frac:.2f}, AUC {best.auc:.3f}")
    chart = probe_viz.plot_method_sweep(sweeps, metric=metric) if len(sweeps) > 1 \
        else probe_viz.plot_layer_sweep(next(iter(sweeps.values())), metric=metric)
    save_chart(chart, output, show)

@app.command("roc", cls=HelpfulCommand)
def probe_roc(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    output: Path = typer.Option(Path("charts/probe-roc.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the ROC curve whose area the sweep reports as one number"""
    adapter = load_adapter(config)
    probe, test_activations, test_set, _ = _load_or_fit(
        adapter, probe_path, data, size, seed, frac, test_frac, method)
    save_chart(probe_viz.plot_roc(probe, test_activations, test_set.labels), output, show)

@app.command("scores", cls=HelpfulCommand)
def probe_scores(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    output: Path = typer.Option(Path("charts/probe-scores.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the probe's score distribution per class: the margin AUC throws away"""
    adapter = load_adapter(config)
    probe, test_activations, test_set, loaded = _load_or_fit(
        adapter, probe_path, data, size, seed, frac, test_frac, method)
    save_chart(probe_viz.plot_score_distribution(
        probe, test_activations, test_set.labels, label_names=label_names(loaded)), output, show)

@app.command("weights", cls=HelpfulCommand)
def probe_weights(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    k: int = typer.Option(20, "-k", "--k", help="How many top dimensions to name"),
    output: Path = typer.Option(Path("charts/probe-weights.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how concentrated the probe's direction is, and which dimensions carry it"""
    adapter = load_adapter(config)
    probe, _, _, _ = _load_or_fit(adapter, probe_path, data, size, seed, frac, test_frac, method)
    save_chart(probe_viz.plot_weight_spectrum(probe, k=k), output, show)

def _parse_point(raw: str) -> Tuple[str, float, float]:
    """Parse a 'name,ms_per_item,score' triple given on the command line"""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(f"expected 'name,ms_per_item,score', got '{raw}'")
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except ValueError as error:
        raise typer.BadParameter(f"'{raw}' has a non-numeric cost or score") from error

@app.command("pareto", cls=HelpfulCommand)
def probe_pareto(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit every method at"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: List[str] = typer.Option(
        ["logistic", "difference_of_means"], "--method", help="Methods to measure; repeatable"
    ),
    point: List[str] = typer.Option(
        [], "--point", help="An externally measured method as 'name,ms_per_item,auc'; repeatable"
    ),
    output: Path = typer.Option(Path("charts/probe-pareto.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot measured cost against AUC, with the Pareto frontier drawn

    The probe methods are fitted and timed here. Anything mi-lab cannot run --
    an LLM judge, a hosted classifier -- goes in through --point with the cost
    and score you measured elsewhere, rather than being estimated silently.
    """
    adapter = load_adapter(config)
    loaded = load_dataset(data, size, seed)
    points = []
    for name in method:
        probe, test_activations, _ = probe_at(adapter, loaded, frac, test_frac, seed, name)
        cost = measure_scoring_cost(probe, test_activations)
        points.append((name, cost.ms_per_item, probe.metrics["auc"]))
        typer.echo(f"{name:<20} AUC {probe.metrics['auc']:.3f}  {cost.ms_per_item * 1000:.1f} us per activation")
    points.extend(_parse_point(raw) for raw in point)
    save_chart(probe_viz.plot_pareto(points), output, show)

