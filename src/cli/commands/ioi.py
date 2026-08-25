import typer

from ...core.adapter import load_adapter, require_circuits
from ...core.circuits import classify_heads, direct_logit_attribution, discover, patch_heads, patch_residual, verify
from ...core.ioi import CORRUPTIONS, FRAMES, build_ioi, evaluate
from ..common import HelpfulCommand, HelpfulGroup

"""
Replicate the Indirect Object Identification circuit from the command line.

The commands are the argument in order: does the model do the task, what does
each head write towards the answer, where does the information sit, which
heads actually cause it, and does the set they add up to survive being checked.
Each one prints numbers and nothing else -- the same functions drawn rather
than tabulated live under `viz circuit`.

Patching costs one forward pass per site, so the default dataset is small on
purpose. Widen it once a result looks worth trusting, not before.

Run with: python -m src.cli ioi <command> [options]
"""

app = typer.Typer(help="Replicate the IOI circuit: attribute, patch, discover, verify.", cls=HelpfulGroup)

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")
SIZE_OPTION = typer.Option(16, "--size", help="Clean/corrupted prompt pairs to build")
SEED_OPTION = typer.Option(0, "--seed", help="Seed for the names, places and objects")
FRAME_OPTION = typer.Option(0, "--frame", help=f"Which of the {len(FRAMES)} sentence frames to use")
CORRUPTION_OPTION = typer.Option(
    "abc", "--corruption", help=f"How the clean prompt is broken; one of {sorted(CORRUPTIONS)}"
)

def _prepare(config: str, size: int, seed: int, frame: int, corruption: str, quiet: bool = False):
    """Load the model and build the dataset every command in this group starts from"""
    adapter = require_circuits(load_adapter(config))
    dataset = build_ioi(adapter, size=size, seed=seed, frame=frame, corruption=corruption)
    if not quiet:
        typer.echo(f"{adapter.cfg.id}: {adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads")
        typer.echo(f"dataset: {dataset.name}  n={len(dataset)}  ABBA share {dataset.balance:.0%}")
        typer.echo(f"  clean    : {dataset.examples[0].clean}")
        typer.echo(f"  corrupted: {dataset.examples[0].corrupted}")
        typer.echo(f"  answer   : {dataset.examples[0].io} (not {dataset.examples[0].subject})")
    return adapter, dataset

@app.command("dataset", cls=HelpfulCommand)
def show_dataset(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    show: int = typer.Option(4, "--show", help="Print this many example pairs"),
):
    """Build the prompts and show what the model is actually being asked

    Prints the token positions too. Those are what a patching heatmap is
    indexed by, and a chart whose columns you cannot name is a chart you
    cannot read.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption, quiet=True)
    typer.echo(f"{dataset.name}: {len(dataset)} pairs, ABBA share {dataset.balance:.0%}")
    typer.echo(f"frame: {dataset.frame}\n")

    for example in dataset.examples[:show]:
        typer.echo(f"[{example.order}] {example.clean}")
        typer.echo(f"           {example.corrupted}")
        typer.echo(f"           -> {typer.style(example.io, bold=True)}, not {example.subject}\n")

    landmarks = {position: name for name, position in dataset.landmarks(adapter).items()}
    typer.echo("positions:")
    for index, token in enumerate(dataset.token_labels(adapter)):
        marker = f"  <- {landmarks[index]}" if index in landmarks else ""
        typer.echo(f"  {index:>3}  {token!r}{marker}")

@app.command("evaluate", cls=HelpfulCommand)
def evaluate_task(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
):
    """Report accuracy and the logit span the corruption opens

    The span is the number to look at. Everything the rest of this group
    reports is a fraction of it, so a span near zero means the later commands
    will hand back confident noise.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption)
    report = evaluate(adapter, dataset)
    typer.echo(f"\n{report}")
    if abs(report.span) < 0.5:
        typer.echo(
            "warning: the corruption barely moved the answer, so patching has almost nothing to recover",
            err=True,
        )

@app.command("attribute", cls=HelpfulCommand)
def attribute(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    top: int = typer.Option(6, "--top", help="Heads to list at each end"),
):
    """Split the answer into what each head and MLP wrote towards it

    One forward pass, every component. The line to check first is the
    unattributed remainder: it is the receipt on the decomposition, and it is
    zero when every place a model writes into its residual stream is one this
    framework knows how to hook.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption)
    result = direct_logit_attribution(adapter, dataset)

    typer.echo(f"\nmeasured logit difference : {result.measured:+.3f}")
    typer.echo(f"attributed to components  : {result.total:+.3f}")
    typer.echo(f"unattributed remainder    : {result.residual:+.2e}")
    typer.echo(f"  embedding {result.embedding:+.3f}   biases and final shift {result.offset:+.3f}")

    typer.echo(f"\ntowards the answer (top {top}):")
    for (layer, head), value in result.top(top):
        typer.echo(f"  L{layer:<2} H{head:<2}  {value:+.3f}")
    typer.echo(f"\nagainst the answer (top {top}):")
    for (layer, head), value in result.top(top, negative=True):
        typer.echo(f"  L{layer:<2} H{head:<2}  {value:+.3f}")

    typer.echo("\nMLP by layer:")
    for layer, value in enumerate(result.mlps.tolist()):
        typer.echo(f"  L{layer:<2}      {value:+.3f}")

@app.command("heads", cls=HelpfulCommand)
def heads(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.3, help="Attention weight a head needs before it gets named"),
):
    """Name each head after the attention movement it spends itself on

    Attention says where a head looked, which makes this a list of candidates
    and not a finding. `ioi patch` is what turns one into the other.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption)
    roles = classify_heads(adapter, dataset)
    assigned = roles.assign(threshold=threshold)
    if not assigned:
        typer.echo(f"\nno head puts {threshold:g} of its attention on any of these movements")
        return

    typer.echo("")
    for role in roles.roles:
        named = [head for head, name in sorted(assigned.items()) if name == role]
        listed = ", ".join(f"L{layer}H{head}" for layer, head in named) or "none"
        typer.echo(f"{role:>16}: {listed}")

@app.command("patch", cls=HelpfulCommand)
def patch(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    site: str = typer.Option(
        "heads", "--site", help="'heads' for one head at a time, 'residual' for layer by position"
    ),
    top: int = typer.Option(10, "--top", help="Heads to list, largest absolute effect first"),
):
    """Restore one activation at a time from the clean run and see what comes back

    This is the causal half of the study and the expensive one: a forward pass
    per site. 'heads' asks which head does the task; 'residual' asks where in
    the sentence, and at which depth, the answer becomes decided.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption)
    if site == "heads":
        effects = patch_heads(adapter, dataset)
        typer.echo(f"\nspan to recover: {effects.baselines.span:+.3f} logits")
        for (layer, head), value in effects.ranked(top):
            typer.echo(f"  L{layer:<2} H{head:<2}  {value:+.3f}")
        return
    if site != "residual":
        raise typer.BadParameter(f"unknown site '{site}'; use 'heads' or 'residual'")

    grid = patch_residual(adapter, dataset)
    landmarks = {position: name for name, position in grid.landmarks.items()}
    typer.echo(f"\nspan to recover: {grid.baselines.span:+.3f} logits")
    header = "".join(f"{landmarks.get(index, index)!s:>7}" for index in range(len(grid.tokens)))
    typer.echo(f"{'layer':>6}{header}")
    for layer, row in enumerate(grid.effects.tolist()):
        typer.echo(f"{layer:>6}" + "".join(f"{value:>7.2f}" for value in row))
    best_layer, best_position, best_value = grid.best()
    typer.echo(f"\nbest single site: layer {best_layer}, position {best_position} "
               f"({grid.tokens[best_position]!r}) recovers {best_value:.2f}")

@app.command("circuit", cls=HelpfulCommand)
def circuit(
    config: str = CONFIG_ARGUMENT,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.8, help="Recovery the growing circuit has to reach before the search stops"),
    max_heads: int = typer.Option(12, help="Most heads the search is allowed to add"),
    tolerance: float = typer.Option(0.05, help="Recovery a head has to be worth before it counts as load-bearing"),
    roles: bool = typer.Option(True, help="Name each head found after its attention movement"),
):
    """Grow a circuit, then check whether it is enough, needed, and free of passengers

    The three checks are printed together because any one of them alone is
    misleading. A faithful circuit that is not necessary is one route among
    several; a necessary one full of passengers is a bigger claim than the
    evidence supports.
    """
    adapter, dataset = _prepare(config, size, seed, frame, corruption)
    effects = patch_heads(adapter, dataset)
    found = discover(adapter, dataset, threshold=threshold, max_heads=max_heads, effects=effects)
    typer.echo(f"\n{found}")
    for step, ((layer, head), score) in enumerate(zip(found.heads, found.scores, strict=True), start=1):
        typer.echo(f"  {step:>2}. + L{layer}H{head:<2}  recovery {score:+.3f}")

    report = verify(adapter, dataset, found)
    typer.echo(f"\nfaithfulness (this set alone restores the answer) : {report.faithfulness:.3f}")
    typer.echo(f"necessity    (the clean run needs it)              : {report.necessity:.3f}")
    typer.echo("minimality   (recovery lost by dropping each head) :")
    named = classify_heads(adapter, dataset).assign() if roles else {}
    for (layer, head), drop in report.minimality.items():
        role = f"  [{named[(layer, head)]}]" if (layer, head) in named else ""
        verdict = "load-bearing" if drop >= tolerance else "spare"
        typer.echo(f"  L{layer:<2} H{head:<2}  {drop:+.3f}  {verdict}{role}")

    spare = report.spare(tolerance=tolerance)
    if spare:
        listed = ", ".join(f"L{layer}H{head}" for layer, head in spare)
        typer.echo(f"\n{len(spare)} head(s) the circuit does not need: {listed}")
