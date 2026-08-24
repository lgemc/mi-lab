from pathlib import Path
from typing import List, Optional

import typer

from ...core.run import Run, find_runs
from ...core.runner import EXPERIMENTS, run_experiment
from ...core.spec import load_spec
from ..common import HelpfulCommand, HelpfulGroup

"""
Execute experiments described as specs, and read back what they produced.

Nothing here takes an experiment parameter as a flag: everything the run
depends on lives in the spec, and --set is how you change one without editing
the file. That is what keeps a run reproducible from the directory it wrote.
"""

app = typer.Typer(help="Run experiments from specs, and inspect the runs they produce.", cls=HelpfulGroup)

SET_OPTION = typer.Option(
    [], "--set", "-s",
    help="Override any spec key by dotted path, e.g. -s model.config=pythia-70m; repeatable",
)

@app.command("kinds", cls=HelpfulCommand)
def kinds():
    """List the experiment kinds a spec may ask for"""
    for name, function in sorted(EXPERIMENTS.items()):
        summary = (function.__doc__ or "").strip().splitlines()[0]
        typer.echo(f"{name:<14} {summary}")

@app.command("show", cls=HelpfulCommand)
def show(
    spec_path: Optional[str] = typer.Argument(None, help="Spec file; omit to see the defaults"),
    set_: List[str] = SET_OPTION,
):
    """Resolve a spec and print it, without running anything

    The hash is the reproducibility key: it covers everything that determines
    the result, so two specs that print the same hash will produce the same
    numbers, and changing only output paths will not change it.
    """
    spec = load_spec(spec_path, overrides=set_)
    typer.echo(f"experiment: {spec.experiment}  kind: {spec.kind}  hash: {spec.spec_hash}")
    for section, values in spec.as_dict().items():
        if isinstance(values, dict):
            typer.echo(f"{section}:")
            for key, value in values.items():
                typer.echo(f"  {key}: {value}")
        else:
            typer.echo(f"{section}: {values}")

@app.command("exec", cls=HelpfulCommand)
def execute(
    spec_path: Optional[str] = typer.Argument(None, help="Spec file; omit to run the defaults"),
    set_: List[str] = SET_OPTION,
    root: Optional[str] = typer.Option(None, "--root", help="Where to write the run; defaults to the spec's output.root"),
):
    """Run an experiment and report what it measured"""
    spec = load_spec(spec_path, overrides=set_)
    typer.echo(f"running '{spec.experiment}' ({spec.kind}) hash {spec.spec_hash}")
    run = run_experiment(spec, root=root)

    directory = Path(root or spec.output.root) / run.run_id
    typer.echo(f"{run.status} in {run.duration_seconds:.1f}s -> {directory}")
    for key, value in sorted(run.metrics.items()):
        typer.echo(f"  {key}: {value:g}")
    for ref in run.produced:
        typer.echo(f"  produced {ref.kind}: {ref.id}")

@app.command("list", cls=HelpfulCommand)
def list_runs(root: str = typer.Argument("runs", help="Directory holding run directories")):
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
