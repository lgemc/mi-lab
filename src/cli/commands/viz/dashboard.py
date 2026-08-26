from pathlib import Path
from typing import List, Optional

import typer

from ....core.adapter import load_adapter
from ....core.probing import measure_scoring_cost
from ....viz import activations as act_viz
from ....viz import dataset as dataset_viz
from ....viz import model as model_viz
from ....viz import probing as probe_viz
from ....viz.dashboard import Panel, Section, render
from ....viz.style import save_figure
from .common import (
    CONFIG_ARGUMENT,
    DATA_OPTION,
    FRAC_OPTION,
    SEED_OPTION,
    fracs_or_default,
    label_names,
    load_dataset,
)
from .probing import probe_at, run_sweep

"""
One run's charts assembled into a single self-contained HTML page.

It is a command rather than a group, and it lives in its own module because it
is the one chart that spans every other group: it fits the same probes the
probing group fits and draws the same panels the others draw, off one dataset
and one capture, so the numbers on the page are comparable with each other.

__init__.py registers it on the root app. Doing that here instead would mean
importing the root app from the package that imports this module, which is a
cycle.
"""

def dashboard(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(120, "--size", help="Examples to generate when using the synthetic set"),
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    frac: List[float] = FRAC_OPTION,
    probe_frac: float = typer.Option(0.65, help="Depth to fit the single-layer probe at"),
    out_dir: Path = typer.Option(Path("charts"), "--out-dir", help="Where the PNGs and index.html go"),
    show: bool = typer.Option(False, "--show", help="Print the path when finished"),
):
    """Run the whole battery for one model and dataset and assemble it into one HTML page

    Everything the lab can currently see about a model, in the order you would
    ask the questions: what is in the data, where the model puts it, whether a
    probe can read it, and what it cost. The page embeds its own images, so it
    can be moved anywhere afterwards.
    """
    adapter = load_adapter(config)
    loaded = load_dataset(data, size, seed)
    fracs = fracs_or_default(frac) or [index / 8 for index in range(9)]
    layers = adapter.cfg.layers(fracs)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(name: str, target) -> Path:
        path = out_dir / f"{name}.png"
        save_figure(target, path)
        typer.echo(f"  drew {path}")
        return path

    typer.echo("dataset...")
    data_section = Section("The data", [
        Panel("Class balance", "An AUC quoted without this can be a probe that learned the base rate.",
              draw("dataset-balance", dataset_viz.plot_class_balance(loaded))),
        Panel("Length by class", "If the classes differ in length, the probe can read the length instead.",
              draw("dataset-lengths", dataset_viz.plot_length_distribution(loaded))),
        Panel("Bag-of-words baseline", "Tokens that separate the classes on their own. A probe has to beat this.",
              draw("dataset-tokens", dataset_viz.plot_token_log_odds(loaded))),
        Panel("Split leakage", "How close each test example is to its nearest training one.",
              draw("dataset-leakage", dataset_viz.plot_split_leakage(*loaded.split(test_frac=test_frac, seed=seed)))),
    ])

    typer.echo("model...")
    model_section = Section("The model", [
        Panel("Depth ruler", "One fraction, one line, a different layer index per model.",
              draw("depth-ruler", model_viz.plot_depth_ruler([adapter.cfg], fracs=fracs)), wide=True),
    ])

    typer.echo(f"capturing {len(loaded)} prompts at layers {layers}...")
    captured = adapter.capture(loaded.texts, layers=layers)
    all_layers = list(range(adapter.cfg.n_layers))
    everything = adapter.capture(loaded.texts[:40], layers=all_layers)

    act_section = Section("The activations", [
        Panel("Norm by layer", "Should climb with depth. Flat or zero means the hook is not where you think.",
              draw("act-norms", act_viz.plot_layer_norms(captured, layers=layers))),
        Panel("Layer similarity", "How fast the residual stream turns over. A bright diagonal band is healthy.",
              draw("act-similarity", act_viz.plot_layer_similarity(everything, layers=all_layers))),
        Panel("One prompt's stream", "Vertical stripes are outlier dimensions that fire at every depth.",
              draw("act-heatmap", act_viz.plot_activation_heatmap(captured, layers=layers))),
        Panel("Class separation by layer", "The probe sweep before the probe: where the colours come apart.",
              draw("act-pca", act_viz.plot_layer_pca(captured, loaded.labels, layers=layers,
                                                     label_names=label_names(loaded))), wide=True),
    ])

    typer.echo("probing...")
    sweeps = {name: run_sweep(adapter, loaded, test_frac, seed, fracs, name)
              for name in ("logistic", "difference_of_means")}
    probe, test_activations, test_set = probe_at(adapter, loaded, probe_frac, test_frac, seed, "logistic")
    cost = measure_scoring_cost(probe, test_activations)
    points = [(name, measure_scoring_cost(reports[0].probe, test_activations).ms_per_item,
               max(report.auc for report in reports))
              for name, reports in sweeps.items()]

    probe_section = Section("The probe", [
        Panel("Depth sweep, both methods", "Where the signal lives, and which method reads it best at each depth.",
              draw("probe-sweep", probe_viz.plot_method_sweep(sweeps)), wide=True),
        Panel("ROC", "The curve the AUC is the area under.",
              draw("probe-roc", probe_viz.plot_roc(probe, test_activations, test_set.labels))),
        Panel("Score separation", "The margin AUC throws away. Touching humps mean a fragile threshold.",
              draw("probe-scores", probe_viz.plot_score_distribution(
                  probe, test_activations, test_set.labels, label_names=label_names(loaded)))),
        Panel("Direction spectrum", "Whether a few residual dimensions carry the property or all of them do.",
              draw("probe-weights", probe_viz.plot_weight_spectrum(probe)), wide=True),
        Panel("Cost against AUC", "Measured, not estimated. Add external methods with viz probe pareto --point.",
              draw("probe-pareto", probe_viz.plot_pareto(points))),
    ])

    best = max(sweeps["logistic"], key=lambda report: report.auc)
    path = render(
        [data_section, model_section, act_section, probe_section],
        out_dir / "index.html",
        title=f"mi-lab: {adapter.cfg.id}",
        subtitle=f"{loaded.name} - {len(loaded)} examples, {loaded.balance:.0%} positive",
        provenance=[
            f"{adapter.cfg.id} via {adapter.cfg.backend}: {adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model}, {adapter.cfg.dtype}",
            f"best probe: layer {best.layer} at depth {best.frac:.2f}, AUC {best.auc:.3f}",
            f"probe artifact {probe.n_bytes / 1024:.1f} KiB, scoring {cost.ms_per_item * 1000:.1f} us per activation",
        ],
    )
    typer.echo(f"\nWrote {path}")
    if show:
        typer.echo(f"open file://{path.resolve()}")
