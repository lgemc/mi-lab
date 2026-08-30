from pathlib import Path
from typing import Optional

import typer

from ...data.tasks import TASKS, build_task, task_names
from ...methods.circuits import behaviour
from ...methods.comparison import compare_techniques, consistency, discover_across, specificity
from ...methods.discovery import TECHNIQUES, technique_names
from ...model.adapter import load_adapter, require_circuits
from ...share import storage
from ...share.converters.comparison import from_comparison
from ..common import HelpfulCommand, HelpfulGroup

"""
Compare the ways of finding a circuit, rather than reporting one.

`ioi` answers "which heads do this task". This group answers the three
questions that come after it: do the techniques agree about which heads,
is it the same circuit on every example, and would the circuit have cost
another task any less.

The commands are in the order their bills go up. `list` needs no model,
`techniques` needs one forward pass per head for the expensive half of the
methods, `consistency` needs that again per example, and `specificity` needs
a circuit per task before it can ablate any of them.

Run with: python -m src.cli compare <command> [options]
"""

app = typer.Typer(
    help="Compare circuit-finding techniques: agreement, consistency, specificity.", cls=HelpfulGroup
)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")
TASK_OPTION = typer.Option("ioi", "--task", help=f"Which task to measure on; one of {task_names()}")
SIZE_OPTION = typer.Option(16, "--size", help="Clean/corrupted prompt pairs to build")
SEED_OPTION = typer.Option(0, "--seed", help="Seed for the prompts and for the random control")
COUNT_OPTION = typer.Option(8, "--count", help="Heads each technique is allowed to select")

def _prepare(config: str, task: str, size: int, seed: int, quiet: bool = False):
    """Load the model and build the task every command in this group starts from"""
    adapter = require_circuits(load_adapter(config))
    built = build_task(task, adapter, size=size, seed=seed)
    if not quiet:
        typer.echo(f"{adapter.cfg.id}: {adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads")
        typer.echo(f"task {built.name}: n={len(built)}")
        typer.echo(f"  clean    : {built.clean[0]}")
        typer.echo(f"  corrupted: {built.corrupted[0]}")
    return adapter, built

@app.command("list", cls=HelpfulCommand)
def show_list():
    """List the tasks and the techniques, with what each one costs

    No model is loaded. The cost column is the argument: a technique that
    agrees with patching at a constant number of passes is the only kind that
    survives a model with two hundred heads per layer.
    """
    typer.echo("tasks:")
    for name in task_names():
        typer.echo(f"  {name:<14} {TASKS[name].description}")
    typer.echo("\ntechniques:")
    for name in technique_names():
        technique = TECHNIQUES[name]
        typer.echo(f"  {name:<14} {technique.description}")
        typer.echo(f"  {'':<14} in {technique.units}, {technique.cost}")

@app.command("tasks", cls=HelpfulCommand)
def show_tasks(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
):
    """Check that the model does each task before asking which heads do it

    A task the model fails is a task whose circuit is a circuit for failing.
    The span is the other number to read: everything a patching study reports
    is a fraction of it.
    """
    adapter = require_circuits(load_adapter(config))
    typer.echo(f"{adapter.cfg.id}: {adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads\n")
    typer.echo(f"{'task':<14}{'n':>4}{'accuracy':>10}{'clean':>9}{'corrupted':>11}{'span':>8}")
    for name in task_names():
        task = build_task(name, adapter, size=size, seed=seed)
        clean = behaviour(adapter, task)
        corrupted = behaviour(adapter, task, prompts=task.corrupted)
        typer.echo(
            f"{name:<14}{len(task):>4}{clean.accuracy:>9.0%}{clean.logit_difference:>+9.2f}"
            f"{corrupted.logit_difference:>+11.2f}{clean.logit_difference - corrupted.logit_difference:>+8.2f}"
        )
        typer.echo(f"  {task.clean[0]}")

@app.command("techniques", cls=HelpfulCommand)
def compare(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    methods: Optional[str] = typer.Option(
        None, "--methods", help="Comma-separated techniques to run; every one of them by default"
    ),
    check: bool = typer.Option(True, help="Verify each circuit as well as ranking the heads"),
    samples: int = typer.Option(6, "--samples", help="Subsets drawn per circuit for the completeness check"),
):
    """Run every technique on one task and score all of them the same way

    Every circuit is the same size, because faithfulness climbs with the head
    count and a comparison at different sizes compares sizes. Read the two
    matrices together: techniques can order the tail identically and still
    disagree about the top, which is where the circuit actually is.
    """
    adapter, built = _prepare(config, task, size, seed)
    chosen = [name.strip() for name in methods.split(",")] if methods else None
    result = compare_techniques(
        adapter, built, methods=chosen, count=count, check=check, samples=samples, seed=seed
    )

    typer.echo(f"\n{'technique':<13}{'faithful':>10}{'necessity':>11}{'incomplete':>12}{'passes':>8}{'seconds':>9}")
    for one in result.results:
        typer.echo(
            f"{one.method:<13}{one.faithfulness:>+10.3f}{one.necessity:>+11.3f}{one.incompleteness:>12.3f}"
            f"{one.ranking.passes:>8}{one.ranking.seconds:>9.1f}"
        )
    typer.echo("")
    for one in result.results:
        heads = one.heads or [head for head, _ in one.ranking.ranked(count)]
        typer.echo(f"  {one.method:<13}" + ", ".join(f"L{layer}H{head}" for layer, head in heads))

    for which, label in (("overlap", "circuit overlap (intersection over union)"), ("order", "rank correlation")):
        typer.echo(f"\n{label}:")
        typer.echo(f"{'':<13}" + "".join(f"{name:>13}" for name in result.methods))
        for name, row in zip(result.methods, result.matrix(which), strict=True):
            typer.echo(f"{name:<13}" + "".join(f"{value:>13.2f}" for value in row))

@app.command("consistency", cls=HelpfulCommand)
def show_consistency(
    config: str = CONFIG_ARGUMENT,
    task: str = TASK_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    method: str = typer.Option("eap", "--method", help="Technique to run once per example"),
    presence: float = typer.Option(
        0.5, "--presence", help="Share of examples a head must appear in before it counts as shared"
    ),
    examples: Optional[int] = typer.Option(
        None, "--examples", help="Per-example circuits to find; all of them by default"
    ),
):
    """Find a circuit per example and see how much of each one the examples share

    A circuit found on a batch is an average, and an average can be made of
    components that no single example used. Reuse near the chance line means
    exactly that, and it is not visible from any batch-level number.
    """
    adapter, built = _prepare(config, task, size, seed)
    found = consistency(adapter, built, method=method, count=count, presence=presence, examples=examples)
    typer.echo(f"\n{found}")

    typer.echo("\nhow often each head was selected:")
    for (layer, head), share in sorted(found.frequency.items(), key=lambda item: (-item[1], item[0])):
        marker = "  shared" if share >= presence else ""
        typer.echo(f"  L{layer:<2} H{head:<2}  {share:>5.0%}{marker}")

@app.command("specificity", cls=HelpfulCommand)
def show_specificity(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    count: int = COUNT_OPTION,
    method: str = typer.Option("eap", "--method", help="Technique used to find every task's circuit"),
    tasks: Optional[str] = typer.Option(None, "--tasks", help="Comma-separated tasks; every one of them by default"),
    save: Optional[Path] = typer.Option(
        None, "--save", help=f"Write the whole comparison as a shareable {storage.SUFFIX} artifact"
    ),
):
    """Ablate every task's circuit on every task, and a random one as the floor

    The diagonal is the number a circuit paper reports. The off-diagonal is
    the one that says whether the circuit was ever about its task: if removing
    the IOI circuit costs the other tasks about as much, then it is machinery
    the model uses for everything and the label was never earned.
    """
    adapter = require_circuits(load_adapter(config))
    chosen = [name.strip() for name in tasks.split(",")] if tasks else task_names()
    built = {name: build_task(name, adapter, size=size, seed=seed) for name in chosen}
    circuits = discover_across(adapter, built, method=method, count=count)

    typer.echo(f"{adapter.cfg.id}: circuits of {count} heads found by '{method}'\n")
    for name, heads in circuits.items():
        typer.echo(f"  {name:<14}" + ", ".join(f"L{layer}H{head}" for layer, head in heads))

    found = specificity(adapter, built, circuits, seed=seed)
    typer.echo("\ndamage (row = measured on, column = circuit ablated), as a share of the clean logit difference:")
    typer.echo(f"{'':<14}" + "".join(f"{name:>14}" for name in found.tasks) + f"{'random':>14}")
    for name, row in zip(found.tasks, found.matrix(), strict=True):
        typer.echo(
            f"{name:<14}" + "".join(f"{value:>14.2f}" for value in row)
            + f"{found.control[name].damage:>14.2f}"
        )
    typer.echo("")
    for name in found.tasks:
        typer.echo(
            f"  {name:<14} own {found.own(name):+.2f}  others {found.others(name):+.2f}  "
            f"margin {found.margin(name):+.2f}"
        )
    share = count / (adapter.cfg.n_layers * adapter.cfg.n_heads)
    typer.echo(f"\ncircuit overlap between tasks (two independent picks of this size share {share / (2 - share):.2f}):")
    for (left, right), value in found.overlap.items():
        typer.echo(f"  {left:<14} vs {right:<14} {value:.2f}")

    if save is not None:
        primary = chosen[0]
        comparison = compare_techniques(
            adapter, built[primary], methods=["attribution", "patching", "eap", "random"],
            count=count, seed=seed,
        )
        recurrence = consistency(adapter, built[primary], method=method, count=count)
        artifact = from_comparison(
            adapter.cfg, built[primary], comparison, consistency=recurrence, specificity=found,
            task_key=primary,
            tokens=built[primary].token_labels(adapter), landmarks=built[primary].landmarks(adapter),
        )
        storage.save(artifact, str(save))
        typer.echo(f"\nwrote {save}: {artifact}")
