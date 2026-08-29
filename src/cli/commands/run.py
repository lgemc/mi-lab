from pathlib import Path
from typing import List, Optional

import typer

from ...experiment.run import Run, find_runs
from ...experiment.runner import EXPERIMENTS, run_directory, run_experiment
from ...experiment.spec import compose_spec, groups, load_spec
from ..common import HelpfulCommand, HelpfulGroup

"""
Compose experiments from specs/, run them, and read back what they produced.

Nothing here takes an experiment parameter as a flag: everything a run depends
on comes from the composed spec, and --set is how you change any of it without
a flag having to exist. That is what keeps a run reproducible from the
directory it wrote.

Sweeps live in the Hydra entry point instead, because --multirun needs argv:
    python -m src.app --multirun model=gpt2-small,pythia-70m
"""

app = typer.Typer(help="Compose and run experiments, and inspect the runs they produce.", cls=HelpfulGroup)

PRESET_OPTION = typer.Option(None, "--preset", "-e", help="A named bundle from specs/preset/")
SET_OPTION = typer.Option(
    [], "--set", "-s",
    help="Hydra override: 'model=pythia-70m' swaps a group, 'method.lr=0.1' one key; repeatable",
)

@app.command("groups", cls=HelpfulCommand)
def list_groups():
    """List the config groups in specs/ and the options in each"""
    for name, options in groups().items():
        typer.echo(f"{name:<8} {', '.join(options)}")

@app.command("kinds", cls=HelpfulCommand)
def kinds():
    """List the experiment kinds a spec may ask for"""
    for name, function in sorted(EXPERIMENTS.items()):
        summary = (function.__doc__ or "").strip().splitlines()[0]
        typer.echo(f"{name:<14} {summary}")

@app.command("show", cls=HelpfulCommand)
def show(preset: Optional[str] = PRESET_OPTION, set_: List[str] = SET_OPTION):
    """Compose a spec and print it, without running anything

    The hash is the reproducibility key: it covers everything that determines
    the result, so two specs that print the same hash will produce the same
    numbers, and changing only output paths will not change it.
    """
    spec = compose_spec(preset=preset, overrides=set_)
    typer.echo(f"experiment: {spec.experiment}  kind: {spec.kind}  hash: {spec.spec_hash}")
    for section, values in spec.as_dict().items():
        if isinstance(values, dict):
            typer.echo(f"{section}:")
            for key, value in values.items():
                typer.echo(f"  {key}: {value}")
        else:
            typer.echo(f"{section}: {values}")

def _report(spec, run, root: Optional[str]) -> None:
    """Print what a finished run measured and where it landed"""
    directory = run_directory(spec, run, root)
    typer.echo(f"{run.status} in {run.duration_seconds:.1f}s -> {directory}")
    for key, value in sorted(run.metrics.items()):
        typer.echo(f"  {key}: {value:g}")
    for ref in run.produced:
        typer.echo(f"  produced {ref.kind}: {ref.id}")

@app.command("exec", cls=HelpfulCommand)
def execute(
    preset: Optional[str] = PRESET_OPTION,
    set_: List[str] = SET_OPTION,
    root: Optional[str] = typer.Option(
        None, "--root", help="Where to write the run; defaults to the spec's output.root"
    ),
):
    """Compose an experiment and run it"""
    spec = compose_spec(preset=preset, overrides=set_)
    typer.echo(f"running '{spec.experiment}' ({spec.kind}) hash {spec.spec_hash}")
    _report(spec, run_experiment(spec, root=root), root)

@app.command("replay", cls=HelpfulCommand)
def replay(
    directory: str = typer.Argument(..., help="A run directory, or a path to a spec.yaml"),
    root: Optional[str] = typer.Option(None, "--root", help="Where to write the new run"),
):
    """Re-run the exact spec a previous run recorded

    This reads the run's own self-contained spec.yaml rather than recomposing
    from specs/, so a run stays reproducible from its own directory even after
    the group files it was originally composed from have changed. A matching
    hash afterwards means the rerun really was the same experiment.
    """
    path = Path(directory)
    spec_path = path if path.is_file() else path / "spec.yaml"
    spec = load_spec(str(spec_path))
    typer.echo(f"replaying '{spec.experiment}' ({spec.kind}) hash {spec.spec_hash}")
    _report(spec, run_experiment(spec, root=root), root)

@app.command("list", cls=HelpfulCommand)
def list_runs(root: str = typer.Argument("outputs", help="Directory holding the experiment directories")):
    """List the runs under a root, newest first"""
    runs = find_runs(root)
    if not runs:
        typer.echo(f"no runs under {root}")
        return
    typer.echo(f"{'run':<30} {'status':<10} {'experiment':<28} best")
    for run in runs:
        headline = run.metrics.get("best_auc", run.metrics.get("auc"))
        summary = f"AUC {headline:.3f}" if headline is not None else (run.error or "")
        typer.echo(f"{run.run_id:<30} {run.status:<10} {run.experiment:<28} {summary}")

@app.command("info", cls=HelpfulCommand)
def info(directory: str = typer.Argument(..., help="A run directory")):
    """Show everything one run recorded"""
    run = Run.load(directory)
    typer.echo(f"{run.run_id}  {run.status}  hash {run.spec_hash}")
    typer.echo(f"experiment: {run.experiment} ({run.kind})")
    typer.echo(f"started: {run.created_at}  duration: {run.duration_seconds}s")
    if run.error:
        typer.echo(f"error: {run.error}")
    typer.echo("metrics:")
    for key, value in sorted(run.metrics.items()):
        typer.echo(f"  {key}: {value:g}")
    for ref in run.produced:
        typer.echo(f"produced {ref.kind}: {ref.id}")
