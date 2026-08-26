from pathlib import Path
from typing import Optional

import typer

from ....core.adapter import load_adapter
from ....viz import dataset as dataset_viz
from ...common import HelpfulCommand, HelpfulGroup
from .common import (
    DATA_OPTION,
    SEED_OPTION,
    SHOW_OPTION,
    SIZE_OPTION,
    load_dataset,
    save_chart,
    tokenizer_of,
)

"""
Charts of the dataset itself, drawn before any model loads.

Every mistake these catch -- a class that is really a topic, a duplicate
across the split, a pair whose halves drifted apart -- is a mistake that is
already in the file, and no amount of probing afterwards will report it.

Run with: python -m src.cli viz dataset <command> [options]
"""

app = typer.Typer(help="What is in the data, before anything is trained on it.", cls=HelpfulGroup)

@app.command("balance", cls=HelpfulCommand)
def dataset_balance(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    output: Path = typer.Option(Path("charts/dataset-balance.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how many examples each class has"""
    loaded = load_dataset(data, size, seed)
    save_chart(dataset_viz.plot_class_balance(loaded), output, show)

@app.command("lengths", cls=HelpfulCommand)
def dataset_lengths(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    config: Optional[str] = typer.Option(None, "--config", help="Count real tokens with this model's tokenizer"),
    output: Path = typer.Option(Path("charts/dataset-lengths.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot example length per class, to catch a probe that can read the length

    With --config the counts are the model's own tokens; without it they are
    words, which is enough to see a gap but is not what the model saw.
    """
    loaded = load_dataset(data, size, seed)
    tokenize = None
    if config:
        tokenizer = tokenizer_of(load_adapter(config))
        tokenize = lambda text: tokenizer(text)["input_ids"]
    save_chart(dataset_viz.plot_length_distribution(loaded, tokenize=tokenize), output, show)

@app.command("tokens", cls=HelpfulCommand)
def dataset_tokens(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    k: int = typer.Option(12, "-k", "--k", help="Tokens to show per class"),
    output: Path = typer.Option(Path("charts/dataset-tokens.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the tokens most predictive of each class: the bag-of-words baseline, drawn

    If a handful of words separate the classes, a probe reaching AUC 1.0 has
    not shown the model represents the property.
    """
    loaded = load_dataset(data, size, seed)
    save_chart(dataset_viz.plot_token_log_odds(loaded, k=k), output, show)

@app.command("leakage", cls=HelpfulCommand)
def dataset_leakage(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    output: Path = typer.Option(Path("charts/dataset-leakage.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how similar each test example is to its nearest training example"""
    train_set, test_set = load_dataset(data, size, seed).split(test_frac=test_frac, seed=seed)
    save_chart(dataset_viz.plot_split_leakage(train_set, test_set), output, show)

