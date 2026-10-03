from pathlib import Path
from typing import Optional

import typer

from ...share.hub import PREFIX, HubError, plan, push
from ..common import HelpfulCommand, HelpfulGroup

"""
Publish trained checkpoints to the Hugging Face Hub.

`push` prints what it is about to upload before it uploads it, and with
`--dry-run` prints only that -- several gigabytes is worth one look at the
names first. Nothing here loads a model; it reads weight files and sends them.

Run with: python -m src.cli hub <command> [options]
"""

app = typer.Typer(help="Publish trained checkpoints to the Hugging Face Hub.", cls=HelpfulGroup)


@app.command("push", cls=HelpfulCommand)
def push_runs(
    runs: Path = typer.Argument(..., help="Self-Distillation's runs/ directory (holds seq-<model>, sft-seq-<model>)"),
    repo: str = typer.Option("lgmc/mi-lab", "--repo", help="Hub model repository to commit to"),
    prefix: str = typer.Option(PREFIX, "--prefix", help="Folder inside the repository"),
    revision: Optional[str] = typer.Option(None, "--revision", help="Branch to commit to (default: main)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and upload nothing"),
):
    """Upload the final model of every SDFT and SFT stage, one commit for all of them

    Each stage is named by the sequence that produced it, so
    `seq-Qwen3-0.6B/2-science` lands at
    `<prefix>/qwen3-0.6b/sdft/tool-use-then-science.safetensors`.
    """
    try:
        uploads = plan(runs, prefix=prefix)
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
    typer.echo(push(uploads, repo, revision=revision))
