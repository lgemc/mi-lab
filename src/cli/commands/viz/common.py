from pathlib import Path
from typing import List, Optional, Sequence

import typer

from ....core.adapter import load_adapter
from ....core.config import ConfigError, load_config
from ....core.dataset import synthetic
from ....core.prompts import load_labeled
from ....viz.style import save_figure, show_figure

"""
The options and helpers every viz group shares.

They live here rather than being repeated because a chart group differing on
what --size means, or on whether --data falls back to the toy set, is a
difference nobody notices until two charts disagree about the data they were
drawn from.

Nothing in this module imports src.viz's chart modules: the matplotlib backend
is chosen in __init__.py before any of them load, and a helper that pulled one
in early would take that choice away.
"""

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")
DATA_OPTION = typer.Option(None, "--data", help="A .prompts or .jsonl dataset; omit for the synthetic toy set")
SIZE_OPTION = typer.Option(200, "--size", help="Examples to generate when using the synthetic set")
SEED_OPTION = typer.Option(0, "--seed", help="Seed for the dataset and the split")
SHOW_OPTION = typer.Option(False, "--show", help="Render the chart inline in the terminal after saving")
FRAC_OPTION = typer.Option([], "--frac", help="Depth fraction(s) to use; defaults to nine evenly spaced")

def save_chart(target, path: Path, show: bool):
    """Save a chart and optionally render it inline in the terminal"""
    typer.echo(f"Wrote {save_figure(target, path)}")
    if show:
        show_figure(path)

def load_dataset(data: Optional[str], size: int, seed: int, quiet: bool = False):
    """Load a dataset from disk, or fall back to the built-in toy set"""
    loaded = load_labeled(data) if data else synthetic(n=size, seed=seed)
    if not quiet:
        typer.echo(f"dataset: {loaded.name}  n={len(loaded)}  positives={loaded.positives} ({loaded.balance:.0%})")
    return loaded

def label_names(loaded) -> Sequence[str]:
    """A dataset's own names for its two classes, so charts label them the way the file does"""
    return loaded.label_names

def fracs_or_default(frac: List[float]) -> Optional[List[float]]:
    """Use the fractions given, or let the caller's default stand"""
    return list(frac) or None

def prompts_or_dataset(adapter, prompt: List[str], data: Optional[str], size: int, seed: int) -> List[str]:
    """Prompts given on the command line, or the dataset's texts if none were"""
    if prompt:
        return list(prompt)
    return load_dataset(data, size, seed).texts

def resolve_configs(names: Sequence[str], load: bool) -> List:
    """Turn config names into configs carrying n_layers, loading checkpoints if needed

    A config that states its own sizes never needs a download. One that does
    not is resolved by loading its adapter, and if that fails -- an
    unimplemented backend, a checkpoint that will not fit -- it is reported and
    skipped rather than taking the whole chart down with it.
    """
    resolved = []
    for name in names:
        cfg = load_config(name)
        if cfg.is_resolved:
            resolved.append(cfg)
            continue
        if not load:
            typer.echo(f"skipping '{name}': no n_layers in the config and --no-load was given", err=True)
            continue
        try:
            resolved.append(load_adapter(cfg).cfg)
        except (ConfigError, OSError, ValueError) as error:
            typer.echo(f"skipping '{name}': {error}", err=True)
    if not resolved:
        raise typer.BadParameter("no config could be resolved to a layer count")
    return resolved

def tokenizer_of(adapter):
    """The adapter's tokenizer, if this backend exposes one

    Not part of the ModelAdapter protocol, so anything needing token strings
    asks for it and says clearly when a backend cannot provide it.
    """
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is None:
        raise typer.BadParameter(
            f"the '{adapter.cfg.backend}' backend exposes no tokenizer, so token labels are unavailable"
        )
    return tokenizer

