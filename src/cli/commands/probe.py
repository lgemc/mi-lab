from typing import List, Optional

import typer

from ...data.dataset import synthetic
from ...data.prompts import load_labeled
from ...methods.probing import (
    difference_of_means,
    evaluate,
    measure_scoring_cost,
    sweep,
    train_probe,
)
from ...model.adapter import load_adapter
from ...share.loaders import open_probe
from ..common import HelpfulCommand, HelpfulGroup

"""
Train linear probes on captured activations and report what they cost.

Every command here ends in numbers rather than a plot: an AUC, the AUC of the
training-free baseline it has to beat, the size of the artifact, and the
marginal time to score one activation.
"""

app = typer.Typer(help="Train, evaluate and sweep linear probes.", cls=HelpfulGroup)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")
DATA_OPTION = typer.Option(None, "--data", help="A .prompts or .jsonl dataset; omit for the synthetic toy set")

def _load(data: Optional[str], size: int, seed: int):
    """Load a dataset from disk, or fall back to the built-in toy set"""
    dataset = load_labeled(data) if data else synthetic(n=size, seed=seed)
    typer.echo(f"dataset: {dataset.name}  n={len(dataset)}  positives={dataset.positives} ({dataset.balance:.0%})")
    return dataset

@app.command("train", cls=HelpfulCommand)
def train(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    frac: float = typer.Option(0.65, help="Depth fraction to read activations from"),
    size: int = typer.Option(200, help="Examples to generate when using the synthetic set"),
    test_frac: float = typer.Option(0.3, help="Fraction held out for the reported numbers"),
    seed: int = typer.Option(0, help="Seed for the split and the optimizer"),
    method: str = typer.Option("logistic", help="'logistic' to read, 'difference_of_means' to steer"),
    out: Optional[str] = typer.Option(None, "--out", help="Where to save the probe named by --method"),
):
    """Train a probe at one layer and report AUC, baseline, size and cost

    The difference-of-means baseline is printed next to the trained probe on
    purpose. If the optimizer is not beating a one-pass mean subtraction, the
    honest report is that the property is linearly obvious and the training
    added nothing.
    """
    adapter = load_adapter(config)
    dataset = _load(data, size, seed)
    train_set, test_set = dataset.split(test_frac=test_frac, seed=seed)
    layer = adapter.cfg.layer(frac)

    train_activations = adapter.capture(train_set.texts, layers=[layer])
    test_activations = adapter.capture(test_set.texts, layers=[layer])

    provenance = {"model_id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers, "layer": layer, "dataset": dataset.name}
    probe = train_probe(train_activations, train_set.labels, seed=seed, **provenance)
    baseline = difference_of_means(train_activations, train_set.labels, **provenance)

    trained = evaluate(probe, test_activations, test_set.labels)
    reference = evaluate(baseline, test_activations, test_set.labels)
    probe.metrics.update(trained)
    cost = measure_scoring_cost(probe, test_activations)

    typer.echo(f"layer {layer} of {adapter.cfg.n_layers} (depth {frac})  train={len(train_set)} test={len(test_set)}")
    typer.echo(f"logistic probe     AUC {trained['auc']:.3f}  accuracy {trained['accuracy']:.3f}")
    typer.echo(
        f"difference of means AUC {reference['auc']:.3f}"
        f"  accuracy {reference['accuracy']:.3f}  (baseline, no training)"
    )
    typer.echo(f"artifact: {probe.n_bytes / 1024:.1f} KiB   scoring: {cost.ms_per_item * 1000:.1f} us per activation")
    if trained["auc"] <= reference["auc"]:
        typer.echo("note: the trained probe did not beat the untrained baseline here", err=True)
    if out:
        saved = probe if method == "logistic" else baseline
        saved.save(out)
        typer.echo(f"saved {saved.method} probe to {out}")

@app.command("sweep", cls=HelpfulCommand)
def sweep_layers(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    frac: List[float] = typer.Option([], "--frac", help="Depth fraction(s) to try; defaults to nine evenly spaced"),
    size: int = typer.Option(200, help="Examples to generate when using the synthetic set"),
    test_frac: float = typer.Option(0.3, help="Fraction held out for the reported numbers"),
    seed: int = typer.Option(0, help="Seed for the split and the optimizer"),
    method: str = typer.Option("logistic", help="'logistic' or 'difference_of_means'"),
    out: Optional[str] = typer.Option(None, "--out", help="Where to save the best layer's probe"),
):
    """Train a probe at every depth and show where the signal lives

    Which layer wins is a fact about this model. That the winner sits in the
    middle rather than at either end is the part expected to survive a model
    swap -- if it does not, that is a finding, not a bug.
    """
    adapter = load_adapter(config)
    dataset = _load(data, size, seed)
    train_set, test_set = dataset.split(test_frac=test_frac, seed=seed)

    reports = sweep(adapter, train_set, test_set, fracs=frac or None, method=method, seed=seed)
    best = max(reports, key=lambda report: report.auc)

    typer.echo(f"{'layer':>6} {'depth':>6} {'AUC':>7} {'acc':>7}")
    for report in reports:
        marker = "  <- best" if report is best else ""
        typer.echo(
            f"{report.layer:>6} {report.frac:>6.2f} {report.auc:>7.3f} "
            f"{report.metrics['accuracy']:>7.3f}{marker}"
        )
    typer.echo(f"best: layer {best.layer} of {adapter.cfg.n_layers} at depth {best.frac:.2f}, AUC {best.auc:.3f}")
    if out:
        best.probe.save(out)
        typer.echo(f"saved probe to {out}")

@app.command("score", cls=HelpfulCommand)
def score(
    config: str = CONFIG_ARGUMENT,
    probe_path: str = typer.Option(..., "--probe", help="Path to a probe .pt, or to a shared .mia artifact"),
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to score; repeatable"),
):
    """Apply a saved probe to new prompts

    The probe carries the layer it was trained on, so nothing here needs to be
    told where to look.
    """
    adapter = load_adapter(config)
    probe = open_probe(probe_path)
    if probe.model_id != adapter.cfg.id:
        typer.echo(f"warning: probe was trained on '{probe.model_id}', scoring '{adapter.cfg.id}'", err=True)

    activations = adapter.capture(prompt, layers=[probe.layer])
    typer.echo(f"probe: layer {probe.layer}, trained on {probe.model_id} / {probe.dataset} ({probe.method})")
    for text, value in zip(prompt, probe.score(activations), strict=True):
        verdict = "positive" if value > 0 else "negative"
        typer.echo(f"{value:>8.3f}  {verdict:<8}  {text}")
