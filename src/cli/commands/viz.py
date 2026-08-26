from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import typer

matplotlib.use("Agg")  # the CLI writes files; a notebook importing src.viz keeps its own backend

from ...core.adapter import load_adapter, require_circuits
from ...core.circuits import classify_heads, direct_logit_attribution, discover, patch_heads, patch_residual, verify
from ...core.config import ConfigError, Position, load_config, presets
from ...core.dataset import synthetic
from ...core.ioi import CORRUPTIONS, FRAMES, build_ioi
from ...core.metrics import measure
from ...core.probing import LinearProbe, difference_of_means, evaluate, measure_scoring_cost, sweep, train_probe
from ...core.prompts import load_labeled
from ...core.run import Run, find_runs
from ...core.steering import random_control, strength_sweep
from ...viz import activations as act_viz
from ...viz import circuits as circuit_viz
from ...viz import dataset as dataset_viz
from ...viz import model as model_viz
from ...viz import probing as probe_viz
from ...viz import runs as runs_viz
from ...viz import steering as steer_viz
from ...viz.dashboard import Panel, Section, render
from ...viz.style import save_figure, show_figure
from ..common import HelpfulCommand, HelpfulGroup

"""
Render the charts in viz/ and write them to disk. Each command mirrors one viz
function: the command loads what that function needs and saves the figure it
returns, and does no plotting of its own.

Sub-groups follow the shape of the questions rather than the shape of the
code -- what is in the data, what is in the model, what is in the
activations, what the probe found, what steering did, which heads do the task,
what the runs say.

Every command takes --output and --show, and --show renders the chart inline
in the terminal for anyone working over ssh.

Run with: python -m src.cli viz <group> <command> [options]
"""

app = typer.Typer(help="Draw the dataset, the model, the activations, the probes and the runs.", cls=HelpfulGroup)

dataset_app = typer.Typer(help="What is in the data, before anything is trained on it.", cls=HelpfulGroup)
model_app = typer.Typer(help="What a config resolves to, and how models compare in shape.", cls=HelpfulGroup)
act_app = typer.Typer(help="What came out of capture: norms, geometry, drift.", cls=HelpfulGroup)
probe_app = typer.Typer(help="What the probe found, and what it cost.", cls=HelpfulGroup)
steer_app = typer.Typer(help="What steering did, as a curve rather than one generation.", cls=HelpfulGroup)
circuit_app = typer.Typer(help="Which heads do the task, correlationally and causally.", cls=HelpfulGroup)
runs_app = typer.Typer(help="What a pile of runs adds up to.", cls=HelpfulGroup)

app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(act_app, name="act")
app.add_typer(probe_app, name="probe")
app.add_typer(steer_app, name="steer")
app.add_typer(circuit_app, name="circuit")
app.add_typer(runs_app, name="runs")

CONFIG_ARGUMENT = typer.Argument(..., help="Name of a config in configs/, or a path to a YAML/JSON config")
DATA_OPTION = typer.Option(None, "--data", help="A .prompts or .jsonl dataset; omit for the synthetic toy set")
SIZE_OPTION = typer.Option(200, "--size", help="Examples to generate when using the synthetic set")
SEED_OPTION = typer.Option(0, "--seed", help="Seed for the dataset and the split")
SHOW_OPTION = typer.Option(False, "--show", help="Render the chart inline in the terminal after saving")
FRAC_OPTION = typer.Option([], "--frac", help="Depth fraction(s) to use; defaults to nine evenly spaced")

def _save(target, path: Path, show: bool):
    """Save a chart and optionally render it inline in the terminal"""
    typer.echo(f"Wrote {save_figure(target, path)}")
    if show:
        show_figure(path)

def _dataset(data: Optional[str], size: int, seed: int, quiet: bool = False):
    """Load a dataset from disk, or fall back to the built-in toy set"""
    loaded = load_labeled(data) if data else synthetic(n=size, seed=seed)
    if not quiet:
        typer.echo(f"dataset: {loaded.name}  n={len(loaded)}  positives={loaded.positives} ({loaded.balance:.0%})")
    return loaded

def _label_names(loaded) -> Sequence[str]:
    """A dataset's own names for its two classes, so charts label them the way the file does"""
    return loaded.label_names

def _fracs(frac: List[float]) -> Optional[List[float]]:
    """Use the fractions given, or let the caller's default stand"""
    return list(frac) or None

def _prompts(adapter, prompt: List[str], data: Optional[str], size: int, seed: int) -> List[str]:
    """Prompts given on the command line, or the dataset's texts if none were"""
    if prompt:
        return list(prompt)
    return _dataset(data, size, seed).texts

def _resolve_configs(names: Sequence[str], load: bool) -> List:
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

def _tokenizer_of(adapter):
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

# --------------------------------------------------------------------------- dataset

@dataset_app.command("balance", cls=HelpfulCommand)
def dataset_balance(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    output: Path = typer.Option(Path("charts/dataset-balance.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how many examples each class has"""
    loaded = _dataset(data, size, seed)
    _save(dataset_viz.plot_class_balance(loaded), output, show)

@dataset_app.command("lengths", cls=HelpfulCommand)
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
    loaded = _dataset(data, size, seed)
    tokenize = None
    if config:
        tokenizer = _tokenizer_of(load_adapter(config))
        tokenize = lambda text: tokenizer(text)["input_ids"]
    _save(dataset_viz.plot_length_distribution(loaded, tokenize=tokenize), output, show)

@dataset_app.command("tokens", cls=HelpfulCommand)
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
    loaded = _dataset(data, size, seed)
    _save(dataset_viz.plot_token_log_odds(loaded, k=k), output, show)

@dataset_app.command("leakage", cls=HelpfulCommand)
def dataset_leakage(
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    output: Path = typer.Option(Path("charts/dataset-leakage.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how similar each test example is to its nearest training example"""
    train_set, test_set = _dataset(data, size, seed).split(test_frac=test_frac, seed=seed)
    _save(dataset_viz.plot_split_leakage(train_set, test_set), output, show)

# --------------------------------------------------------------------------- model

@model_app.command("ruler", cls=HelpfulCommand)
def model_ruler(
    config: List[str] = typer.Option([], "--config", "-c", help="Config to include; repeatable, defaults to all presets"),
    frac: List[float] = typer.Option([], "--frac", help="Depth fraction(s) to mark; defaults to six"),
    load: bool = typer.Option(True, help="Load checkpoints to resolve layer counts the configs do not state"),
    output: Path = typer.Option(Path("charts/depth-ruler.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the depth ruler: one fraction, one line, a different layer index per model

    The chart this framework exists to make. Every model gets a lane, every
    fraction is one vertical line, and the integers underneath are what a
    hardcoded layer index would have got wrong.
    """
    configs = _resolve_configs(config or presets(), load)
    _save(model_viz.plot_depth_ruler(configs, fracs=_fracs(frac)), output, show)

@model_app.command("shape", cls=HelpfulCommand)
def model_shape(
    config: List[str] = typer.Option([], "--config", "-c", help="Config to include; repeatable, defaults to all presets"),
    load: bool = typer.Option(True, help="Load checkpoints to resolve sizes the configs do not state"),
    output: Path = typer.Option(Path("charts/model-shape.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot residual stream width against depth, one point per model"""
    configs = _resolve_configs(config or presets(), load)
    _save(model_viz.plot_shape(configs), output, show)

# --------------------------------------------------------------------------- activations

@act_app.command("norms", cls=HelpfulCommand)
def act_norms(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option([], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-norms.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot residual stream norm by layer: the cheapest check that the hook is where you think

    Norms should climb with depth. A layer at zero captured nothing, and a
    flat line means every layer captured the same thing.
    """
    adapter = load_adapter(config)
    texts = _prompts(adapter, prompt, data, size, seed)
    layers = adapter.cfg.layers(_fracs(frac) or [index / 8 for index in range(9)])
    with measure(items=len(texts)) as cost:
        captured = adapter.capture(texts, layers=layers)
    typer.echo(f"captured {tuple(captured.shape)} at layers {layers}  ({cost[0]})")
    _save(act_viz.plot_layer_norms(captured, layers=layers), output, show)

@act_app.command("heatmap", cls=HelpfulCommand)
def act_heatmap(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to capture; repeatable"),
    example: int = typer.Option(0, help="Which prompt to draw"),
    dims: int = typer.Option(128, help="How many residual stream dimensions to show"),
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-heatmap.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot one prompt's residual stream as layers by dimensions

    A vertical stripe is an outlier dimension that fires at every depth, and
    those are the ones that dominate any unnormalized distance.
    """
    adapter = load_adapter(config)
    layers = adapter.cfg.layers(_fracs(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture(list(prompt), layers=layers)
    _save(act_viz.plot_activation_heatmap(captured, layers=layers, example=example, dims=dims), output, show)

@act_app.command("tokens", cls=HelpfulCommand)
def act_tokens(
    config: str = CONFIG_ARGUMENT,
    prompt: str = typer.Option(..., "--prompt", "-p", help="The single prompt to draw"),
    frac: List[float] = FRAC_OPTION,
    normalize: str = typer.Option("layer", help="'layer' scales each row to its own max; 'none' shows raw norms"),
    output: Path = typer.Option(Path("charts/act-tokens.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot norm at every (layer, token position): where in the sequence the work happens

    Captures with position=all, which is the only way to see anything other
    than what the model ended up with at the last token.
    """
    adapter = load_adapter(config)
    tokenizer = _tokenizer_of(adapter)
    layers = adapter.cfg.layers(_fracs(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture([prompt], layers=layers, position=Position.ALL)
    tokens = [tokenizer.decode([identifier]).strip() or "_" for identifier in tokenizer(prompt)["input_ids"]]
    _save(act_viz.plot_token_layer_norms(captured, tokens, layers=layers, normalize=normalize), output, show)

@act_app.command("pca", cls=HelpfulCommand)
def act_pca(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(120, "--size", help="Examples to generate when using the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-pca.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot class separation in the top two components at every layer

    The probe sweep before the probe: the layer where the colours come apart
    is the layer the sweep will pick.
    """
    adapter = load_adapter(config)
    loaded = _dataset(data, size, seed)
    layers = adapter.cfg.layers(_fracs(frac) or [index / 8 for index in range(9)])
    captured = adapter.capture(loaded.texts, layers=layers)
    figure = act_viz.plot_layer_pca(captured, loaded.labels, layers=layers, label_names=_label_names(loaded))
    _save(figure, output, show)

@act_app.command("similarity", cls=HelpfulCommand)
def act_similarity(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option([], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    output: Path = typer.Option(Path("charts/act-similarity.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot cosine similarity between every pair of layers: how fast the stream turns over

    Captures every layer of the model, not a sweep, since the point is the
    shape of the whole band.
    """
    adapter = load_adapter(config)
    texts = _prompts(adapter, prompt, data, size, seed)
    layers = list(range(adapter.cfg.n_layers))
    captured = adapter.capture(texts, layers=layers)
    _save(act_viz.plot_layer_similarity(captured, layers=layers), output, show)

@act_app.command("drift", cls=HelpfulCommand)
def act_drift(
    config: str = CONFIG_ARGUMENT,
    against: str = typer.Option(..., "--against", help="The config to compare it with, same prompts, same layers"),
    prompt: List[str] = typer.Option([], "--prompt", "-p", help="Prompt to capture; repeatable, defaults to the dataset"),
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(40, "--size", help="Examples to use when falling back to the synthetic set"),
    seed: int = SEED_OPTION,
    frac: List[float] = FRAC_OPTION,
    output: Path = typer.Option(Path("charts/act-drift.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot per-layer agreement between two models' activations on the same prompts

    Cosine and relative L2 together, because a quantization can leave a
    direction untouched while changing its magnitude, and those are different
    findings. The two configs must have the same depth for the comparison to
    mean anything.
    """
    reference = load_adapter(config)
    compared = load_adapter(against)
    if reference.cfg.n_layers != compared.cfg.n_layers:
        raise typer.BadParameter(
            f"'{reference.cfg.id}' has {reference.cfg.n_layers} layers and "
            f"'{compared.cfg.id}' has {compared.cfg.n_layers}; drift is only defined layer-for-layer"
        )
    texts = _prompts(reference, prompt, data, size, seed)
    fracs = _fracs(frac) or [index / 8 for index in range(9)]
    layers = reference.cfg.layers(fracs)
    figure = act_viz.plot_drift(
        reference.capture(texts, layers=layers),
        compared.capture(texts, layers=layers),
        layers=layers,
        reference_name=reference.cfg.id,
        other_name=compared.cfg.id,
    )
    _save(figure, output, show)

# --------------------------------------------------------------------------- probing

def _sweep(adapter, loaded, test_frac: float, seed: int, fracs, method: str):
    """Run one depth sweep and hand back its reports"""
    train_set, test_set = loaded.split(test_frac=test_frac, seed=seed)
    return sweep(adapter, train_set, test_set, fracs=fracs, method=method,
                 **({"seed": seed} if method == "logistic" else {}))

def _probe_at(adapter, loaded, frac: float, test_frac: float, seed: int, method: str):
    """Fit one probe at one depth and return it with the held-out activations"""
    train_set, test_set = loaded.split(test_frac=test_frac, seed=seed)
    layer = adapter.cfg.layer(frac)
    provenance = {"model_id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers, "layer": layer, "dataset": loaded.name}
    train_activations = adapter.capture(train_set.texts, layers=[layer])
    test_activations = adapter.capture(test_set.texts, layers=[layer])
    fit = difference_of_means if method == "difference_of_means" else train_probe
    extra = {} if method == "difference_of_means" else {"seed": seed}
    probe = fit(train_activations, train_set.labels, **provenance, **extra)
    probe.metrics.update(evaluate(probe, test_activations, test_set.labels))
    return probe, test_activations, test_set

def _load_or_fit(adapter, probe_path, data, size, seed, frac, test_frac, method):
    """Use a saved probe if one was given, otherwise fit a fresh one at --frac"""
    loaded = _dataset(data, size, seed)
    if probe_path:
        probe = LinearProbe.load(probe_path)
        if probe.model_id != adapter.cfg.id:
            typer.echo(f"warning: probe was trained on '{probe.model_id}', scoring '{adapter.cfg.id}'", err=True)
        _, test_set = loaded.split(test_frac=test_frac, seed=seed)
        return probe, adapter.capture(test_set.texts, layers=[probe.layer]), test_set, loaded
    probe, test_activations, test_set = _probe_at(adapter, loaded, frac, test_frac, seed, method)
    return probe, test_activations, test_set, loaded

@probe_app.command("sweep", cls=HelpfulCommand)
def probe_sweep(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out for the reported numbers"),
    frac: List[float] = FRAC_OPTION,
    method: List[str] = typer.Option(["logistic"], "--method", help="Probe method; repeat it to compare methods"),
    metric: str = typer.Option("auc", help="'auc' or 'accuracy'"),
    output: Path = typer.Option(Path("charts/probe-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot probe score by depth, one line per method

    Drawn against depth rather than layer index, so two models land on the
    same axis. Pass --method twice to put logistic and difference-of-means on
    one chart, which is what makes 'the best reader is the worst writer' a
    picture instead of a claim.
    """
    adapter = load_adapter(config)
    loaded = _dataset(data, size, seed)
    fracs = _fracs(frac)
    sweeps = {name: _sweep(adapter, loaded, test_frac, seed, fracs, name) for name in method}
    for name, reports in sweeps.items():
        best = max(reports, key=lambda report: report.auc)
        typer.echo(f"{name:<20} best layer {best.layer} at depth {best.frac:.2f}, AUC {best.auc:.3f}")
    chart = probe_viz.plot_method_sweep(sweeps, metric=metric) if len(sweeps) > 1 \
        else probe_viz.plot_layer_sweep(next(iter(sweeps.values())), metric=metric)
    _save(chart, output, show)

@probe_app.command("roc", cls=HelpfulCommand)
def probe_roc(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    output: Path = typer.Option(Path("charts/probe-roc.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the ROC curve whose area the sweep reports as one number"""
    adapter = load_adapter(config)
    probe, test_activations, test_set, _ = _load_or_fit(
        adapter, probe_path, data, size, seed, frac, test_frac, method)
    _save(probe_viz.plot_roc(probe, test_activations, test_set.labels), output, show)

@probe_app.command("scores", cls=HelpfulCommand)
def probe_scores(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    output: Path = typer.Option(Path("charts/probe-scores.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot the probe's score distribution per class: the margin AUC throws away"""
    adapter = load_adapter(config)
    probe, test_activations, test_set, loaded = _load_or_fit(
        adapter, probe_path, data, size, seed, frac, test_frac, method)
    _save(probe_viz.plot_score_distribution(
        probe, test_activations, test_set.labels, label_names=_label_names(loaded)), output, show)

@probe_app.command("weights", cls=HelpfulCommand)
def probe_weights(
    config: str = CONFIG_ARGUMENT,
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe; omit to fit one at --frac"),
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit at when no probe is given"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: str = typer.Option("logistic", help="Method to fit when no probe is given"),
    k: int = typer.Option(20, "-k", "--k", help="How many top dimensions to name"),
    output: Path = typer.Option(Path("charts/probe-weights.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot how concentrated the probe's direction is, and which dimensions carry it"""
    adapter = load_adapter(config)
    probe, _, _, _ = _load_or_fit(adapter, probe_path, data, size, seed, frac, test_frac, method)
    _save(probe_viz.plot_weight_spectrum(probe, k=k), output, show)

def _parse_point(raw: str) -> Tuple[str, float, float]:
    """Parse a 'name,ms_per_item,score' triple given on the command line"""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(f"expected 'name,ms_per_item,score', got '{raw}'")
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except ValueError as error:
        raise typer.BadParameter(f"'{raw}' has a non-numeric cost or score") from error

@probe_app.command("pareto", cls=HelpfulCommand)
def probe_pareto(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = SIZE_OPTION,
    seed: int = SEED_OPTION,
    frac: float = typer.Option(0.65, help="Depth to fit every method at"),
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    method: List[str] = typer.Option(["logistic", "difference_of_means"], "--method", help="Methods to measure; repeatable"),
    point: List[str] = typer.Option([], "--point", help="An externally measured method as 'name,ms_per_item,auc'; repeatable"),
    output: Path = typer.Option(Path("charts/probe-pareto.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot measured cost against AUC, with the Pareto frontier drawn

    The probe methods are fitted and timed here. Anything mi-lab cannot run --
    an LLM judge, a hosted classifier -- goes in through --point with the cost
    and score you measured elsewhere, rather than being estimated silently.
    """
    adapter = load_adapter(config)
    loaded = _dataset(data, size, seed)
    points = []
    for name in method:
        probe, test_activations, test_set = _probe_at(adapter, loaded, frac, test_frac, seed, name)
        cost = measure_scoring_cost(probe, test_activations)
        points.append((name, cost.ms_per_item, probe.metrics["auc"]))
        typer.echo(f"{name:<20} AUC {probe.metrics['auc']:.3f}  {cost.ms_per_item * 1000:.1f} us per activation")
    points.extend(_parse_point(raw) for raw in point)
    _save(probe_viz.plot_pareto(points), output, show)

# --------------------------------------------------------------------------- steering

@steer_app.command("sweep", cls=HelpfulCommand)
def steer_sweep(
    config: str = CONFIG_ARGUMENT,
    prompt: List[str] = typer.Option(..., "--prompt", "-p", help="Prompt to continue while steered; repeatable"),
    probe_path: Optional[str] = typer.Option(None, "--probe", help="A saved probe to steer along and score with"),
    positive: List[str] = typer.Option([], "--positive", help="Prompt exhibiting the behaviour; repeatable"),
    negative: List[str] = typer.Option([], "--negative", help="Matched prompt lacking it; repeatable"),
    frac: float = typer.Option(0.65, help="Depth to read the direction from and steer at"),
    strength: List[float] = typer.Option([], "--strength", help="Strengths to try; defaults to 0 through 3"),
    control: bool = typer.Option(True, help="Also sweep a random vector of the same norm, as the control"),
    max_new_tokens: Optional[int] = typer.Option(None, help="Tokens to generate; defaults to the config"),
    output: Path = typer.Option(Path("charts/steer-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot effect and fluency against steering strength, with a random control

    The ceiling is where effect is still climbing and fluency has started to
    fall, and one generation at one strength cannot show it. Give either a
    saved probe, or contrast pairs to build a direction from.
    """
    adapter = load_adapter(config)
    strengths = list(strength) or [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    if probe_path:
        probe = LinearProbe.load(probe_path)
        layer, direction = probe.layer, probe.direction.float()
    elif positive and negative:
        layer = adapter.cfg.layer(frac)
        captured = adapter.capture(list(positive) + list(negative), layers=[layer])
        probe = difference_of_means(captured, [1] * len(positive) + [0] * len(negative),
                                    layer=layer, model_id=adapter.cfg.id,
                                    n_layers=adapter.cfg.n_layers, dataset="contrast pairs")
        direction = probe.direction.float()
    else:
        raise typer.BadParameter("give either --probe, or both --positive and --negative")

    typer.echo(f"steering layer {layer} of {adapter.cfg.n_layers} over strengths {strengths}")
    sweeps = {
        "direction": strength_sweep(adapter, list(prompt), layer, direction, strengths,
                                    probe=probe, max_new_tokens=max_new_tokens),
    }
    if control:
        sweeps["random control"] = strength_sweep(
            adapter, list(prompt), layer, random_control(direction), strengths,
            probe=probe, max_new_tokens=max_new_tokens)

    for point in sweeps["direction"]:
        typer.echo(f"  strength {point.strength:>4.1f}  effect {point.effect:>7.3f}  "
                   f"fluency {point.fluency:.2f}  {point.completions[0][:60]!r}")

    if control:
        _save(steer_viz.plot_control_comparison(sweeps), output, show)
    else:
        _save(steer_viz.plot_strength_sweep(sweeps["direction"]), output, show)

# --------------------------------------------------------------------------- circuit

IOI_SIZE_OPTION = typer.Option(16, "--size", help="Clean/corrupted prompt pairs to build")
FRAME_OPTION = typer.Option(0, "--frame", help=f"Which of the {len(FRAMES)} sentence frames to use")
CORRUPTION_OPTION = typer.Option(
    "abc", "--corruption", help=f"How the clean prompt is broken; one of {sorted(CORRUPTIONS)}"
)

def _ioi(config: str, size: int, seed: int, frame: int, corruption: str):
    """Load a circuit-capable adapter and build the IOI dataset every chart here starts from"""
    adapter = require_circuits(load_adapter(config))
    dataset = build_ioi(adapter, size=size, seed=seed, frame=frame, corruption=corruption)
    typer.echo(f"{dataset.name}: {len(dataset)} pairs on {adapter.cfg.id}, ABBA share {dataset.balance:.0%}")
    return adapter, dataset

@circuit_app.command("attribution", cls=HelpfulCommand)
def circuit_attribution(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    output: Path = typer.Option(Path("charts/circuit-attribution.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw what each head wrote directly towards the answer, as a layer-by-head grid"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    _save(circuit_viz.plot_attribution(direct_logit_attribution(adapter, dataset)), output, show)

@circuit_app.command("patch", cls=HelpfulCommand)
def circuit_patch(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    site: str = typer.Option(
        "heads", "--site", help="'heads' for a layer-by-head grid, 'residual' for layer by position"
    ),
    output: Optional[Path] = typer.Option(None, help="Where to save the chart; defaults to the site's own name"),
    show: bool = SHOW_OPTION,
):
    """Draw what patching recovered, either per head or across the sentence

    'residual' is the chart that tells the story: the signal sits on the
    repeated name early and moves to the end of the sentence partway up.
    """
    if site not in ("heads", "residual"):
        raise typer.BadParameter(f"unknown site '{site}'; use 'heads' or 'residual'")
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    target = output or Path(f"charts/circuit-patch-{site}.png")
    if site == "heads":
        _save(circuit_viz.plot_head_effects(patch_heads(adapter, dataset)), target, show)
    else:
        _save(circuit_viz.plot_patch_grid(patch_residual(adapter, dataset)), target, show)

@circuit_app.command("roles", cls=HelpfulCommand)
def circuit_roles(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.3, help="Attention weight a head needs before it gets named"),
    output: Path = typer.Option(Path("charts/circuit-roles.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw how much attention each head spends on each of the task's four movements"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    _save(circuit_viz.plot_roles(classify_heads(adapter, dataset), threshold=threshold), output, show)

@circuit_app.command("compare", cls=HelpfulCommand)
def circuit_compare(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    output: Path = typer.Option(Path("charts/circuit-compare.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Draw direct attribution against causal effect, one point per head

    The chart the rest of this group exists to set up: the heads off the
    diagonal are the ones neither method would have found alone.
    """
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    attribution = direct_logit_attribution(adapter, dataset)
    effects = patch_heads(adapter, dataset)
    labelled = dict(effects.ranked(4))
    _save(circuit_viz.plot_attribution_against_effect(attribution, effects, labels=labelled), output, show)

@circuit_app.command("verify", cls=HelpfulCommand)
def circuit_verify(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.8, help="Recovery the growing circuit has to reach before the search stops"),
    max_heads: int = typer.Option(12, help="Most heads the search is allowed to add"),
    output: Path = typer.Option(Path("charts/circuit-verify.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Grow a circuit and draw whether it is enough, needed, and free of passengers"""
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    found = discover(adapter, dataset, threshold=threshold, max_heads=max_heads)
    typer.echo(f"  {found}")
    _save(circuit_viz.plot_verification(verify(adapter, dataset, found)), output, show)

@circuit_app.command("dashboard", cls=HelpfulCommand)
def circuit_dashboard(
    config: str = CONFIG_ARGUMENT,
    size: int = IOI_SIZE_OPTION,
    seed: int = SEED_OPTION,
    frame: int = FRAME_OPTION,
    corruption: str = CORRUPTION_OPTION,
    threshold: float = typer.Option(0.8, help="Recovery the growing circuit has to reach before the search stops"),
    max_heads: int = typer.Option(12, help="Most heads the search is allowed to add"),
    out_dir: Path = typer.Option(Path("charts"), "--out-dir", help="Where the PNGs and the HTML page go"),
    show: bool = typer.Option(False, "--show", help="Print the path when finished"),
):
    """Run the whole IOI battery once and assemble it into one HTML page

    Everything measured off a single dataset and a single set of baselines, so
    the charts on the page are comparable with each other -- which is the
    thing that stops being true the moment they are made one command at a time.
    """
    adapter, dataset = _ioi(config, size, seed, frame, corruption)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(name: str, target) -> Path:
        path = out_dir / f"{name}.png"
        save_figure(target, path)
        typer.echo(f"  drew {path}")
        return path

    typer.echo("behaviour and attribution...")
    attribution = direct_logit_attribution(adapter, dataset)
    roles = classify_heads(adapter, dataset)
    behaviour = Section("What each head writes", [
        Panel("Direct logit attribution", "Exact, cheap, and blind to everything but the direct path to the logits.",
              draw("circuit-attribution", circuit_viz.plot_attribution(attribution))),
        Panel("Strongest writers", "The same grid as a ranking, for the heads worth naming.",
              draw("circuit-top-heads", circuit_viz.plot_top_heads(
                  attribution.top(6) + attribution.top(4, negative=True), legend="logits towards the answer"))),
        Panel("Where each head looks", "Attention names candidates; only patching promotes one to a finding.",
              draw("circuit-roles", circuit_viz.plot_roles(roles)), wide=True),
    ])

    typer.echo("patching (one forward pass per site)...")
    effects = patch_heads(adapter, dataset)
    grid = patch_residual(adapter, dataset)
    causal = Section("What each head causes", [
        Panel("Head patching", "Restore one head from the clean run and see how much of the answer returns.",
              draw("circuit-patch-heads", circuit_viz.plot_head_effects(effects))),
        Panel("Written against caused", "The heads off the diagonal are the ones neither method finds alone.",
              draw("circuit-compare", circuit_viz.plot_attribution_against_effect(
                  attribution, effects, labels=dict(effects.ranked(4))))),
        Panel("Residual stream by position", "Where in the sentence, and at what depth, the answer gets decided.",
              draw("circuit-patch-residual", circuit_viz.plot_patch_grid(grid)), wide=True),
    ])

    typer.echo("circuit...")
    found = discover(adapter, dataset, threshold=threshold, max_heads=max_heads, effects=effects)
    report = verify(adapter, dataset, found)
    typer.echo(f"  {report}")
    checked = Section("The circuit", [
        Panel("Growth", "A curve that jumps and flattens found a circuit; one that climbs evenly did not.",
              draw("circuit-growth", circuit_viz.plot_circuit_growth(found))),
        Panel("Verification", "Enough on its own, needed by the clean run, and free of passengers.",
              draw("circuit-verify", circuit_viz.plot_verification(report))),
    ])

    page = render(
        [behaviour, causal, checked],
        out_dir / "circuit.html",
        title=f"The IOI circuit in {adapter.cfg.id}",
        subtitle="Which heads identify the indirect object, asked correlationally and then causally.",
        provenance=[
            f"{len(dataset)} clean/corrupted pairs, '{corruption}' corruption, seed {seed}",
            f"logit difference {report.baselines.clean:+.3f} clean / {report.baselines.corrupted:+.3f} corrupted",
            f"circuit: {found}",
            f"faithfulness {report.faithfulness:.2f}, necessity {report.necessity:.2f}",
        ],
    )
    typer.echo(f"Wrote {page}")
    if show:
        typer.echo(f"open {page.resolve()}")

# --------------------------------------------------------------------------- runs

@runs_app.command("compare", cls=HelpfulCommand)
def runs_compare(
    root: str = typer.Argument("runs", help="Directory holding run directories"),
    metric: str = typer.Option("best_auc", help="Metric to compare"),
    group_by: str = typer.Option("experiment", "--group-by", help="'experiment', or a dotted path into the run's params"),
    output: Path = typer.Option(Path("charts/runs-compare.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Plot one metric across runs: the scaling contract, checked rather than quoted

    --group-by model.config puts one experiment's result on two models side by
    side, which is the comparison the course's scaling table is about.
    """
    found = find_runs(root)
    if not found:
        raise typer.BadParameter(f"no runs under {root}")
    _save(runs_viz.plot_metric_across_runs(found, metric=metric, group_by=group_by), output, show)

@runs_app.command("sweep", cls=HelpfulCommand)
def runs_sweep(
    directory: str = typer.Argument(..., help="A run directory produced by a probe_sweep"),
    output: Path = typer.Option(Path("charts/run-sweep.png"), help="Where to save the chart"),
    show: bool = SHOW_OPTION,
):
    """Redraw a finished sweep's depth curve from the metrics it recorded, with no recompute"""
    _save(runs_viz.plot_sweep_from_run(Run.load(directory)), output, show)

# --------------------------------------------------------------------------- dashboard

@app.command("dashboard", cls=HelpfulCommand)
def dashboard(
    config: str = CONFIG_ARGUMENT,
    data: Optional[str] = DATA_OPTION,
    size: int = typer.Option(120, "--size", help="Examples to generate when using the synthetic set"),
    seed: int = SEED_OPTION,
    test_frac: float = typer.Option(0.3, help="Fraction held out"),
    frac: List[float] = FRAC_OPTION,
    probe_frac: float = typer.Option(0.65, help="Depth to fit the single-layer probe at"),
    out_dir: Path = typer.Option(Path("charts"), "--out-dir", help="Where the PNGs and index.html go"),
    show: bool = typer.Option(False, "--show", help="Print the path when finished"),
):
    """Run the whole battery for one model and dataset and assemble it into one HTML page

    Everything the lab can currently see about a model, in the order you would
    ask the questions: what is in the data, where the model puts it, whether a
    probe can read it, and what it cost. The page embeds its own images, so it
    can be moved anywhere afterwards.
    """
    adapter = load_adapter(config)
    loaded = _dataset(data, size, seed)
    fracs = _fracs(frac) or [index / 8 for index in range(9)]
    layers = adapter.cfg.layers(fracs)
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(name: str, target) -> Path:
        path = out_dir / f"{name}.png"
        save_figure(target, path)
        typer.echo(f"  drew {path}")
        return path

    typer.echo("dataset...")
    data_section = Section("The data", [
        Panel("Class balance", "An AUC quoted without this can be a probe that learned the base rate.",
              draw("dataset-balance", dataset_viz.plot_class_balance(loaded))),
        Panel("Length by class", "If the classes differ in length, the probe can read the length instead.",
              draw("dataset-lengths", dataset_viz.plot_length_distribution(loaded))),
        Panel("Bag-of-words baseline", "Tokens that separate the classes on their own. A probe has to beat this.",
              draw("dataset-tokens", dataset_viz.plot_token_log_odds(loaded))),
        Panel("Split leakage", "How close each test example is to its nearest training one.",
              draw("dataset-leakage", dataset_viz.plot_split_leakage(*loaded.split(test_frac=test_frac, seed=seed)))),
    ])

    typer.echo("model...")
    model_section = Section("The model", [
        Panel("Depth ruler", "One fraction, one line, a different layer index per model.",
              draw("depth-ruler", model_viz.plot_depth_ruler([adapter.cfg], fracs=fracs)), wide=True),
    ])

    typer.echo(f"capturing {len(loaded)} prompts at layers {layers}...")
    captured = adapter.capture(loaded.texts, layers=layers)
    all_layers = list(range(adapter.cfg.n_layers))
    everything = adapter.capture(loaded.texts[:40], layers=all_layers)

    act_section = Section("The activations", [
        Panel("Norm by layer", "Should climb with depth. Flat or zero means the hook is not where you think.",
              draw("act-norms", act_viz.plot_layer_norms(captured, layers=layers))),
        Panel("Layer similarity", "How fast the residual stream turns over. A bright diagonal band is healthy.",
              draw("act-similarity", act_viz.plot_layer_similarity(everything, layers=all_layers))),
        Panel("One prompt's stream", "Vertical stripes are outlier dimensions that fire at every depth.",
              draw("act-heatmap", act_viz.plot_activation_heatmap(captured, layers=layers))),
        Panel("Class separation by layer", "The probe sweep before the probe: where the colours come apart.",
              draw("act-pca", act_viz.plot_layer_pca(captured, loaded.labels, layers=layers,
                                                     label_names=_label_names(loaded))), wide=True),
    ])

    typer.echo("probing...")
    sweeps = {name: _sweep(adapter, loaded, test_frac, seed, fracs, name)
              for name in ("logistic", "difference_of_means")}
    probe, test_activations, test_set = _probe_at(adapter, loaded, probe_frac, test_frac, seed, "logistic")
    cost = measure_scoring_cost(probe, test_activations)
    points = [(name, measure_scoring_cost(reports[0].probe, test_activations).ms_per_item,
               max(report.auc for report in reports))
              for name, reports in sweeps.items()]

    probe_section = Section("The probe", [
        Panel("Depth sweep, both methods", "Where the signal lives, and which method reads it best at each depth.",
              draw("probe-sweep", probe_viz.plot_method_sweep(sweeps)), wide=True),
        Panel("ROC", "The curve the AUC is the area under.",
              draw("probe-roc", probe_viz.plot_roc(probe, test_activations, test_set.labels))),
        Panel("Score separation", "The margin AUC throws away. Touching humps mean a fragile threshold.",
              draw("probe-scores", probe_viz.plot_score_distribution(
                  probe, test_activations, test_set.labels, label_names=_label_names(loaded)))),
        Panel("Direction spectrum", "Whether a few residual dimensions carry the property or all of them do.",
              draw("probe-weights", probe_viz.plot_weight_spectrum(probe)), wide=True),
        Panel("Cost against AUC", "Measured, not estimated. Add external methods with viz probe pareto --point.",
              draw("probe-pareto", probe_viz.plot_pareto(points))),
    ])

    best = max(sweeps["logistic"], key=lambda report: report.auc)
    path = render(
        [data_section, model_section, act_section, probe_section],
        out_dir / "index.html",
        title=f"mi-lab: {adapter.cfg.id}",
        subtitle=f"{loaded.name} - {len(loaded)} examples, {loaded.balance:.0%} positive",
        provenance=[
            f"{adapter.cfg.id} via {adapter.cfg.backend}: {adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model}, {adapter.cfg.dtype}",
            f"best probe: layer {best.layer} at depth {best.frac:.2f}, AUC {best.auc:.3f}",
            f"probe artifact {probe.n_bytes / 1024:.1f} KiB, scoring {cost.ms_per_item * 1000:.1f} us per activation",
        ],
    )
    typer.echo(f"\nWrote {path}")
    if show:
        typer.echo(f"open file://{path.resolve()}")
