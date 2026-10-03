from pathlib import Path
from typing import List, Optional

import typer

from ...share.hub import CROSSCODER_PREFIX, PREFIX, HubError, Upload, crosscoder_plan, plan, push
from ...telemetry.results import root as results_root
from ..common import HelpfulCommand, HelpfulGroup

"""
Publish trained checkpoints to the Hugging Face Hub.

Both commands print what they are about to upload before they upload it, and
with `--dry-run` print only that -- gigabytes are worth one look at the names
first. Nothing here loads a model; it reads files and sends them.

Run with: python -m src.cli hub <command> [options]
"""

app = typer.Typer(help="Publish trained checkpoints to the Hugging Face Hub.", cls=HelpfulGroup)

REPO = typer.Option("lgmc/mi-lab", "--repo", help="Hub model repository to commit to")
REVISION = typer.Option(None, "--revision", help="Branch to commit to (default: main)")
DRY_RUN = typer.Option(False, "--dry-run", help="Print the plan and upload nothing")


def _publish(build, repo: str, revision: Optional[str], dry_run: bool, message: str) -> None:
    """Print the plan `build` returns, then commit it unless this is a dry run"""
    try:
        uploads: List[Upload] = build()
    except HubError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    total = 0
    for upload in uploads:
        total += upload.size
        shards = f"  ({len(upload.sources)} shards, merged)" if len(upload.sources) > 1 else ""
        typer.echo(f"{upload.size / 2**30:6.2f} GiB  {upload.path_in_repo}  <-  {upload.sources[0]}{shards}")
    typer.echo(f"{total / 2**30:6.2f} GiB  total, {len(uploads)} files -> {repo}")
    if dry_run:
        return
    typer.echo(push(uploads, repo, message=message, revision=revision))


@app.command("models", cls=HelpfulCommand)
def models(
    runs: Path = typer.Argument(..., help="Self-Distillation's runs/ directory (holds seq-<model>, sft-seq-<model>)"),
    repo: str = REPO,
    prefix: str = typer.Option(PREFIX, "--prefix", help="Folder inside the repository"),
    revision: Optional[str] = REVISION,
    dry_run: bool = DRY_RUN,
):
    """Upload the final model of every SDFT and SFT stage, one commit for all of them

    Each stage is named by the sequence that produced it, so
    `seq-Qwen3-0.6B/2-science` lands at
    `<prefix>/qwen3-0.6b/sdft/tool-use-then-science.safetensors`.
    """
    _publish(lambda: plan(runs, prefix=prefix), repo, revision, dry_run, "Add SDFT and SFT stage weights")


@app.command("crosscoders", cls=HelpfulCommand)
def crosscoders(
    results: Path = typer.Argument(None, help="Results root holding the crosscoders (default: MI_LAB_RESULTS)"),
    repo: str = REPO,
    prefix: str = typer.Option(CROSSCODER_PREFIX, "--prefix", help="Folder inside the repository"),
    revision: Optional[str] = REVISION,
    dry_run: bool = DRY_RUN,
):
    """Upload every trained crosscoder with its report, one folder per checkpoint pair

    A crosscoder trained on the base model and SDFT's last stage lands at
    `<prefix>/qwen3-0.6b/sdft/base--tool-use-then-science/crosscoder.safetensors`,
    with `crosscoder.json` beside it. The checkpoint names come from the configs
    the crosscoder records, so run this from a checkout that has them.
    """
    root = results or results_root()
    _publish(lambda: crosscoder_plan(root, prefix=prefix), repo, revision, dry_run, "Add crosscoders")
