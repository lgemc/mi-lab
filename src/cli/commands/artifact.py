from pathlib import Path
from typing import Optional

import typer

from ...methods.probing import LinearProbe, ProbeError
from ...share import storage
from ...share.converters.probe import from_probe
from ...share.schema import Artifact, ArtifactError
from ..common import HelpfulCommand, HelpfulGroup

"""
Read, check and package the shareable form of a result.

Nothing here loads a model, which is the point of the format: whether an
artifact is worth downloading, which model it belongs to and what it claims
are all answerable from the card alone. `show` is what you run on someone
else's directory before deciding to spend a checkpoint on it.

Run with: python -m src.cli artifact <command> [options]
"""

app = typer.Typer(help="Read, check and package shareable interpretability artifacts.", cls=HelpfulGroup)

PATH_ARGUMENT = typer.Argument(..., help=f"Path to a {storage.SUFFIX} artifact directory")

def _open(path: str) -> Artifact:
    """Load an artifact or exit with the reason it could not be read"""
    try:
        return storage.load(path)
    except ArtifactError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

@app.command("show", cls=HelpfulCommand)
def show(
    path: str = PATH_ARGUMENT,
    top: int = typer.Option(10, "--top", help="Circuit nodes to print, strongest causal effect first"),
):
    """Print an artifact's card: what it measured, on what, and against which baseline

    The span is printed before the metrics on purpose. Every fractional number
    below it is a share of that span, and a recovery quoted over a corruption
    that barely moved the model is the one mistake this format exists to make
    visible.
    """
    artifact = _open(path)
    model = artifact.model
    typer.echo(f"{artifact.kind}  {typer.style(artifact.id, bold=True)}  ({artifact.version}, {artifact.created_at})")
    typer.echo(
        f"model    : {model.id} ({model.hf_name}), {model.n_layers} layers "
        f"x {model.n_heads} heads, {model.dtype}"
    )

    site = artifact.site
    depths = ", ".join(f"{frac:.2f}" for frac in site.fracs[:6]) + (" ..." if len(site.fracs) > 6 else "")
    typer.echo(f"site     : {site.component} at layers {site.layers[:6]}{' ...' if len(site.layers) > 6 else ''}"
               f" (depth {depths}), position {site.position}")
    typer.echo(f"method   : {artifact.method}")

    if artifact.task:
        count = artifact.task.get("n")
        typer.echo(f"data     : {artifact.task.get('name', 'unnamed')}" + (f", n={count}" if count else ""))
    if artifact.span is not None:
        span = artifact.span
        typer.echo(
            f"baseline : {span.metric} {span.clean:+.3f} clean / {span.corrupted:+.3f} corrupted "
            f"(span {span.span:+.3f})"
        )
    # the definition is printed under the number rather than beside it, because a
    # metric name is not a definition and two artifacts reporting 'faithfulness 0.9'
    # are only comparable if they measured the same thing
    for key, metric in artifact.metrics.items():
        typer.echo(f"  {key:<22} {metric.value:+.4f}  [{metric.units}]")
        typer.echo(typer.style(f"  {'':<22} {metric.definition}", dim=True))

    if artifact.identifiability:
        typer.echo(f"assumed  : {', '.join(artifact.identifiability)}")
    controls = artifact.controls
    if not controls.empty:
        typer.echo("controls :")
        for control in controls.cross_task:
            typer.echo(f"  cross-task  {control.name:<18} {control.metric} {control.value:+.4f}")
        for control in controls.random_baseline:
            typer.echo(f"  random      {control.name:<18} {control.metric} {control.value:+.4f}")

    typer.echo("tensors  :")
    for name, payload in artifact.tensors.items():
        axes = " x ".join(f"{axis}={size}" for axis, size in
                          zip(payload.axes, payload.values.shape, strict=True)) or "scalar"
        typer.echo(f"  {name:<22} {axes}  [{payload.units}]")

    heads = artifact.circuit_heads
    if heads:
        typer.echo(f"circuit  : {len(heads)} nodes, {len(artifact.edges)} measured edges")
        for node in heads[:top]:
            role = f"  {node.role}" if node.role else ""
            attribution = node.scores.get("attribution", float("nan"))
            causal = node.scores.get("causal", float("nan"))
            typer.echo(f"  {node.id:<8} causal {causal:+.3f}  attribution {attribution:+.3f}{role}")

    provenance = artifact.provenance
    commit = provenance.get("git_commit") or "unknown"
    dirty = " (uncommitted changes)" if provenance.get("git_dirty") else ""
    typer.echo(f"made by  : {provenance.get('tool', '?')} at {commit}{dirty}, torch {provenance.get('torch', '?')}")
    typer.echo(f"size     : {artifact.n_bytes / 1024:.1f} KiB")
    if artifact.notes:
        typer.echo(f"\n{artifact.notes}")

@app.command("check", cls=HelpfulCommand)
def check(path: str = PATH_ARGUMENT):
    """Validate an artifact and name what a reader could not trust about it

    Exits non-zero if it is unusable, so this is the thing to run over a
    directory of downloads before any of them is applied to a model.
    """
    artifact = _open(path)
    typer.echo(f"{artifact}: readable, card and tensors agree")

    warnings = []
    if artifact.provenance.get("git_dirty"):
        warnings.append("made from a tree with uncommitted changes, so its commit does not name the code that ran")
    if not artifact.provenance.get("git_commit"):
        warnings.append("no commit recorded, so the code that produced it cannot be recovered")
    if artifact.span is not None and abs(artifact.span.span) < 1e-3:
        warnings.append(
            f"the baseline span is {artifact.span.span:.2e}, so every fraction in it divides by almost nothing"
        )
    if artifact.kind == "circuit" and not artifact.edges:
        warnings.append("no edges were measured, so this names the parts of a circuit and not its wiring")
    if artifact.kind == "circuit" and not artifact.controls.cross_task:
        warnings.append(
            "no cross-task ablation was run, so every number here is within-task; circuits at this level "
            "overlap heavily and this one is not shown to be particular to the task it names"
        )
    if artifact.kind == "steering_vector" and not artifact.identifiability:
        warnings.append(
            "no structural assumption was imposed when this direction was fit, so it is one of a class of "
            "directions with indistinguishable behaviour rather than the direction for what it claims"
        )
    if artifact.kind == "steering_vector" and not artifact.controls.random_baseline:
        warnings.append(
            "no norm-matched random direction was recorded, so nothing here separates this direction's "
            "effect from what any vector of that size does at that layer"
        )
    if artifact.model.n_layers is None or artifact.model.d_model is None:
        warnings.append("the model's sizes are not recorded, so a payload cannot be checked before it is applied")

    for warning in warnings:
        typer.echo(f"  warning: {warning}")
    if not warnings:
        typer.echo("  nothing to flag")

@app.command("list", cls=HelpfulCommand)
def list_artifacts(root: str = typer.Argument("outputs", help="Directory to search for artifacts")):
    """Every artifact under a directory, one line each"""
    found = storage.find_artifacts(root)
    if not found:
        typer.echo(f"no {storage.SUFFIX} artifacts under {root}")
        raise typer.Exit(code=1)
    for artifact in found:
        typer.echo(f"{artifact.kind:<16} {artifact.id:<40} {artifact.model.id:<14} {artifact.n_bytes / 1024:>8.1f} KiB")

@app.command("pack", cls=HelpfulCommand)
def pack(
    probe_path: str = typer.Argument(..., help="Path to a probe written by 'probe train' or 'probe sweep'"),
    out: Optional[Path] = typer.Option(
        None, "--out", help=f"Where to write it; defaults to the same name{storage.SUFFIX}"
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Identifier to give it; defaults to dataset-model-layer"),
):
    """Wrap a probe .pt as a shareable artifact

    The probe file this repository writes is already self-contained; what it
    lacks is the checkpoint it belongs to spelled out, its provenance, and a
    card a stranger can read without torch. That is what packing adds.
    """
    try:
        probe = LinearProbe.load(probe_path)
    except ProbeError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    target = out or Path(probe_path).with_suffix(storage.SUFFIX)
    try:
        artifact = from_probe(probe, name=name)
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    storage.save(artifact, str(target))
    typer.echo(f"{artifact}\nwrote {target}")
