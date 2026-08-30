from pathlib import Path
from typing import Optional

import typer

from ....data.tasks import build_task, task_names
from ....methods.comparison import compare_techniques, consistency, discover_across, specificity
from ....model.adapter import load_adapter, require_circuits
from ....viz import comparison as comparison_viz
from ....viz.style import save_figure
from ...common import HelpfulCommand, HelpfulGroup
from .common import CONFIG_ARGUMENT, SEED_OPTION, SHOW_OPTION, save_chart

"""
Charts of the comparison: which technique agrees with which, what each one
cost, whether one technique finds the same circuit twice, and what a circuit
costs the tasks it does not claim to explain.

`dashboard` measures every panel off one comparison, one consistency run and
one cross-task sweep, so the panels on the page are about each other. Made one
command at a time they are four charts of four different measurements that
happen to share a colour scheme.

Run with: python -m src.cli viz compare <command> [options]
"""

app = typer.Typer(help="Compare the ways of finding a circuit, drawn.", cls=HelpfulGroup)

TASK_OPTION = typer.Option("ioi", "--task", help=f"Which task to measure on; one of {task_names()}")
SIZE_OPTION = typer.Option(16, "--size", help="Clean/corrupted prompt pairs to build")
COUNT_OPTION = typer.Option(8, "--count", help="Heads each technique is allowed to select")
METHODS_OPTION = typer.Option(None, "--methods", help="Comma-separated techniques; every one of them by default")

def _compare(config: str, task: str, size: int, seed: int, count: int, methods: Optional[str], check: bool = True):
    """Load the model, build the task and run every technique on it"""
    adapter = require_circuits(load_adapter(config))
    built = build_task(task, adapter, size=size, seed=seed)
    typer.echo(f"{built.name}: {len(built)} pairs on {adapter.cfg.id}")
    chosen = [name.strip() for name in methods.split(",")] if methods else None
    return adapter, built, compare_techniques(
        adapter, built, methods=chosen, count=count, check=check, seed=seed
    )

@app.command("agreement", cls=HelpfulCommand)
def agreement(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    methods: Optional[str] = METHODS_OPTION,
    which: str = typer.Option("overlap", "--which", help="'overlap' for the selected circuits, 'order' for the ranks"),
    output: Path = typer.Option(Path("charts/compare-agreement.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw how much the techniques agree, as a technique-by-technique matrix"""
    _, _, comparison = _compare(config, task, size, seed, count, methods, check=False)
    save_chart(comparison_viz.plot_agreement(comparison, which=which), output, show)

@app.command("cost", cls=HelpfulCommand)
def cost(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    methods: Optional[str] = METHODS_OPTION,
    reference: str = typer.Option("patching", "--reference", help="The technique the others are approximating"),
    output: Path = typer.Option(Path("charts/compare-cost.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw what each technique cost against how much of the reference it recovered"""
    _, _, comparison = _compare(config, task, size, seed, count, methods, check=False)
    save_chart(comparison_viz.plot_cost_against_agreement(comparison, reference=reference), output, show)

@app.command("scores", cls=HelpfulCommand)
def scores(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    methods: Optional[str] = METHODS_OPTION,
    output: Path = typer.Option(Path("charts/compare-scores.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw faithfulness, necessity and incompleteness for every technique's circuit"""
    _, _, comparison = _compare(config, task, size, seed, count, methods)
    save_chart(comparison_viz.plot_scores(comparison), output, show)

@app.command("consistency", cls=HelpfulCommand)
def consistency_chart(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    method: str = typer.Option("eap", "--method", help="Technique to run once per example"),
    output: Path = typer.Option(Path("charts/compare-consistency.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw how often each head turned up when the technique was run one example at a time"""
    adapter = require_circuits(load_adapter(config))
    built = build_task(task, adapter, size=size, seed=seed)
    found = consistency(adapter, built, method=method, count=count)
    typer.echo(str(found))
    save_chart(comparison_viz.plot_consistency(found), output, show)

@app.command("specificity", cls=HelpfulCommand)
def specificity_chart(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    method: str = typer.Option("eap", "--method", help="Technique used to find every task's circuit"),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma-separated tasks; every one of them by default"),
    output: Path = typer.Option(Path("charts/compare-specificity.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw what every task's circuit costs every task, as a damage matrix"""
    adapter = require_circuits(load_adapter(config))
    chosen = [name.strip() for name in tasks.split(",")] if tasks else task_names()
    built = {name: build_task(name, adapter, size=size, seed=seed) for name in chosen}
    circuits = discover_across(adapter, built, method=method, count=count)
    save_chart(comparison_viz.plot_specificity(specificity(adapter, built, circuits, seed=seed)), output, show)

@app.command("dashboard", cls=HelpfulCommand)
def dashboard(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    method: str = typer.Option("eap", "--method", help="Technique used for consistency and for the other tasks"),
    output: Path = typer.Option(Path("charts/compare"), help="Directory to write the panels into"),
    show: bool = SHOW_OPTION,
):
    """Draw the whole comparison off one set of measurements

    Every panel is measured from the same comparison, the same per-example run
    and the same cross-task sweep, which is what stops being true the moment
    the charts are made one command at a time -- and a page whose panels
    disagree about the data behind them is worse than no page.
    """
    adapter, built, comparison = _compare(config, task, size, seed, count, None)
    recurrence = consistency(adapter, built, method=method, count=count)
    tasks = {name: build_task(name, adapter, size=size, seed=seed) for name in task_names()}
    circuits = discover_across(adapter, tasks, method=method, count=count)
    across = specificity(adapter, tasks, circuits, seed=seed)

    panels = (
        ("agreement", comparison_viz.plot_agreement(comparison, which="overlap")),
        ("order", comparison_viz.plot_agreement(comparison, which="order")),
        ("cost", comparison_viz.plot_cost_against_agreement(comparison)),
        ("scores", comparison_viz.plot_scores(comparison)),
        ("consistency", comparison_viz.plot_consistency(recurrence)),
        ("specificity", comparison_viz.plot_specificity(across)),
    )
    for name, chart in panels:
        typer.echo(f"Wrote {save_figure(chart, output / f'{name}.png')}")
    if show:
        from ....viz.style import show_figure

        show_figure(output / "cost.png")
