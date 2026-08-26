from pathlib import Path

import typer

from ....core.run import Run, find_runs
from ....viz import runs as runs_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    SHOW_OPTION,
    save_chart,
)

"""
Charts over a directory of runs: what a pile of them adds up to, and how one
metric moved as a parameter was swept.

Run with: python -m src.cli viz runs <command> [options]
"""

app = typer.Typer(help="What a pile of runs adds up to.", cls=HelpfulGroup)

@app.command("compare", cls=HelpfulCommand)
def runs_compare(
    root: str = typer.Argument("runs", help="Directory holding run directories"),
    metric: str = typer.Option("best_auc", help="Metric to compare"),
    group_by: str = typer.Option("experiment", "--group-by", help="'experiment', or a dotted path into the run's params"),
    output: Path = typer.Option(Path("charts/runs-compare.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot one metric across runs: the scaling contract, checked rather than quoted

    --group-by model.config puts one experiment's result on two models side by
    side, which is the comparison the course's scaling table is about.
    """
    found = find_runs(root)
    if not found:
        raise typer.BadParameter(f"no runs under {root}")
    save_chart(runs_viz.plot_metric_across_runs(found, metric=metric, group_by=group_by), output, show)

@app.command("sweep", cls=HelpfulCommand)
def runs_sweep(
    directory: str = typer.Argument(..., help="A run directory produced by a probe_sweep"),
    output: Path = typer.Option(Path("charts/run-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Redraw a finished sweep's depth curve from the metrics it recorded, with no recompute"""
    save_chart(runs_viz.plot_sweep_from_run(Run.load(directory)), output, show)

