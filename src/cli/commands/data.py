from dataclasses import replace
from pathlib import Path

import typer

from ...data.dataset import DatasetError, load_csv, save_jsonl, synthetic
from ...data.prompts import SUFFIXES, load_labeled, save_prompts
from ..common import HelpfulCommand, HelpfulGroup

"""
Read, check and convert the datasets a probe is trained on.

Nothing here loads a model. That is the point: a dataset is wrong or right
before any activation is captured, and the mistakes that matter -- a class
that is really a topic, a duplicate across the split, a pair whose halves
drifted apart -- are all visible in the file.
"""

app = typer.Typer(help="Inspect and convert datasets.", cls=HelpfulGroup)

PATH_ARGUMENT = typer.Argument(..., help="Path to a .prompts or .jsonl dataset")
SOURCE_ARGUMENT = typer.Argument(..., help="Path to a .prompts, .jsonl or .csv dataset")

@app.command("check", cls=HelpfulCommand)
def check(
    path: str = PATH_ARGUMENT,
    test_frac: float = typer.Option(0.3, help="Split to preview, so the numbers you will train on are the ones shown"),
    seed: int = typer.Option(0, help="Seed for the previewed split"),
    show: int = typer.Option(0, "--show", help="Print this many examples, quoted so whitespace is visible"),
):
    """Read a dataset, report what is in it, and name what will go wrong

    Exits non-zero if the file is unusable, so this is the thing to run in a
    pre-commit hook over a directory of hand-edited sets.
    """
    data = load_labeled(path)
    negative, positive = data.label_names
    typer.echo(f"{data.name}: {len(data)} examples, {data.positives} {positive} / {len(data) - data.positives} {negative}")
    typer.echo(f"balance: {data.balance:.0%} positive")

    if data.groups is not None:
        units = data.units
        sizes = sorted({len(unit) for unit in units})
        typer.echo(f"groups: {len(units)} kept whole by a split, sizes {sizes}")

    lengths = [len(text) for text in data.texts]
    typer.echo(f"length: {min(lengths)}-{max(lengths)} characters, median {sorted(lengths)[len(lengths) // 2]}")

    problems = []
    if data.duplicates:
        problems.append(f"{len(data.duplicates)} duplicated texts, which leak across a split: {data.duplicates[:3]}")
    if not 0.3 <= data.balance <= 0.7:
        problems.append(f"balance is {data.balance:.0%}, so a probe can score well by learning the base rate")
    if any(text != text.strip() for text in data.texts):
        problems.append("some prompts start or end with whitespace, which changes how they tokenize")

    train, test = data.split(test_frac=test_frac, seed=seed)
    leaked = set(train.texts) & set(test.texts)
    typer.echo(f"split at {test_frac}: train {len(train)} ({train.balance:.0%}) / test {len(test)} ({test.balance:.0%})")
    if leaked:
        problems.append(f"{len(leaked)} texts appear on both sides of the split")

    width = max(len(name) for name in data.label_names)
    for example in range(min(show, len(data))):
        name = data.label_names[data.labels[example]]
        typer.echo(f"  {name:>{width}}  {data.texts[example]!r}")

    for problem in problems:
        typer.echo(typer.style(f"warning: {problem}", fg=typer.colors.YELLOW), err=True)
    if not problems:
        typer.echo("no problems found")

@app.command("convert", cls=HelpfulCommand)
def convert(
    path: str = SOURCE_ARGUMENT,
    out: str = typer.Option(..., "--out", help="Destination; the suffix picks the format"),
    text_field: str = typer.Option("text", help="Column or field holding the prompt"),
    label_field: str = typer.Option("label", help="Column or field holding the label"),
    group_field: str = typer.Option(None, "--group-field", help="CSV column whose shared value makes rows one group"),
    labels: str = typer.Option(None, "--labels", help="Two comma-separated names for label 0 and label 1"),
    name: str = typer.Option(None, "--name", help="Name for the dataset; defaults to the filename"),
):
    """Convert between .prompts, .jsonl and a downloaded .csv

    JSONL has nowhere to put label names or groups, so converting to it drops
    them. That direction is for handing data to something that is not this
    framework; the other direction is the one to run once on a download and
    then keep.

    A downloaded contrast set is the reason --group-field exists. Rows sharing
    a subject are a pair, and a pair split in half makes the AUC measure the
    one word that differs -- so name the column that identifies the pair, and
    check the result with `data check` before training anything on it.
    """
    if Path(path).suffix.lower() == ".csv":
        data = load_csv(path, text_field=text_field, label_field=label_field, group_field=group_field)
    else:
        data = load_labeled(path, text_field=text_field, label_field=label_field)
    if labels:
        names = tuple(part.strip() for part in labels.split(","))
        if len(names) != 2:
            raise DatasetError(f"--labels wants two comma-separated names, got {labels!r}")
        data = replace(data, label_names=names)
    if name:
        data = data.rename(name)

    suffix = Path(out).suffix.lower()
    if suffix in SUFFIXES:
        save_prompts(data, out)
    elif suffix == ".jsonl":
        if data.groups is not None:
            typer.echo("note: groups do not survive JSONL, so a split of the result can break a pair", err=True)
        save_jsonl(data, out, text_field=text_field, label_field=label_field)
    else:
        raise DatasetError(
            f"cannot tell what format '{out}' should be from its suffix; "
            f"use {sorted((*SUFFIXES, '.jsonl'))}"
        )
    groups = "ungrouped" if data.groups is None else f"{len(data.units)} groups"
    typer.echo(f"wrote {len(data)} examples ({groups}) to {out}")

@app.command("synthetic", cls=HelpfulCommand)
def write_synthetic(
    out: str = typer.Option(..., "--out", help="Where to write the generated .prompts file"),
    size: int = typer.Option(200, help="How many examples to generate"),
    seed: int = typer.Option(0, help="Seed for the templates"),
):
    """Write the built-in toy set out as a file, as a worked example of the format

    Useful for two things: seeing what the format looks like at size, and
    having something to edit into a real set rather than starting from an
    empty file.
    """
    data = synthetic(n=size, seed=seed)
    save_prompts(data, out)
    typer.echo(f"wrote {len(data)} examples to {out}")
