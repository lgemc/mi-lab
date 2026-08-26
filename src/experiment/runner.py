from pathlib import Path
from typing import Callable, Dict, Optional

from ..core.metrics import measure
from ..data.ioi import build_ioi
from ..data.ioi import evaluate as evaluate_ioi
from ..methods.circuits import classify_heads, direct_logit_attribution, discover, patch_heads, patch_residual, verify
from ..methods.probing import difference_of_means, evaluate, sweep, train_probe
from ..model.adapter import load_adapter, require_circuits
from ..share.artifact import SUFFIX
from ..share.sharing import from_circuit
from .run import Run
from .spec import ExperimentSpec, SpecError, save_spec

"""
The runner turns a spec into a Run. It is the one place that knows the order
of operations -- resolve the model, build the data, split it, fit, evaluate,
write -- and every experiment type is a function registered against a kind,
so adding one is a registration rather than an edit here. ioi_circuit is the
proof of that: a circuit study shares none of the probing pipeline and is
still just another entry in EXPERIMENTS.

Every run gets its own directory containing the resolved spec it ran, the
run.json describing what happened, and whatever artifacts it produced. A run
that raises is still written out, with its status and error recorded, because
a failure you cannot inspect afterwards costs more than the run did.

A common pipe could be: load_spec | run_experiment | Run.load
"""

EXPERIMENTS: Dict[str, Callable[[ExperimentSpec, Run, Path], None]] = {}

def register_experiment(kind: str):
    """Register a function as the implementation of an experiment kind"""
    def decorate(function):
        EXPERIMENTS[kind] = function
        return function
    return decorate

def _prepare(spec: ExperimentSpec):
    """Resolve the model and build the train/test split this spec describes"""
    adapter = load_adapter(spec.model.resolve())
    dataset = spec.data.load(seed=spec.seed)
    train, test = dataset.split(test_frac=spec.data.test_frac, seed=spec.seed)
    return adapter, dataset, train, test

@register_experiment("probe_sweep")
def _probe_sweep(spec: ExperimentSpec, run: Run, directory: Path) -> None:
    """Fit a probe at every requested depth and keep the best one"""
    adapter, dataset, train, test = _prepare(spec)
    # difference_of_means takes no hyperparameters; passing it an lr is a TypeError, not a no-op
    hyperparameters = (
        {"seed": spec.seed, "epochs": spec.method.epochs, "lr": spec.method.lr, "l2": spec.method.l2}
        if spec.method.kind == "logistic"
        else {}
    )
    reports = sweep(adapter, train, test, fracs=spec.method.fracs, method=spec.method.kind, **hyperparameters)
    best = max(reports, key=lambda report: report.auc)

    run.record(
        best_auc=best.auc,
        best_accuracy=best.metrics["accuracy"],
        best_layer=best.layer,
        best_frac=best.frac,
        n_train=len(train),
        n_test=len(test),
        balance=dataset.balance,
    )
    for report in reports:
        run.record(**{f"auc_layer_{report.layer}": report.auc})

    if spec.output.save_probe:
        name = f"probe-layer{best.layer}.pt"
        best.probe.save(str(directory / name))
        run.produce("probe", name)

@register_experiment("probe_train")
def _probe_train(spec: ExperimentSpec, run: Run, directory: Path) -> None:
    """Fit one probe at one depth, next to the training-free baseline"""
    adapter, dataset, train, test = _prepare(spec)
    layer = adapter.cfg.layer(spec.method.fracs[0])
    provenance = {"model_id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers, "layer": layer, "dataset": dataset.name}

    train_activations = adapter.capture(train.texts, layers=[layer], position=spec.method.position)
    test_activations = adapter.capture(test.texts, layers=[layer], position=spec.method.position)

    probe = train_probe(
        train_activations, train.labels, seed=spec.seed,
        epochs=spec.method.epochs, lr=spec.method.lr, l2=spec.method.l2, **provenance,
    )
    baseline = difference_of_means(train_activations, train.labels, **provenance)
    trained = evaluate(probe, test_activations, test.labels)
    reference = evaluate(baseline, test_activations, test.labels)
    probe.metrics.update(trained)

    run.record(
        auc=trained["auc"], accuracy=trained["accuracy"],
        baseline_auc=reference["auc"], baseline_accuracy=reference["accuracy"],
        layer=layer, n_train=len(train), n_test=len(test), balance=dataset.balance,
        probe_bytes=probe.n_bytes,
    )

    if spec.output.save_probe:
        chosen = probe if spec.method.kind == "logistic" else baseline
        name = f"probe-layer{layer}.pt"
        chosen.save(str(directory / name))
        run.produce("probe", name)

@register_experiment("ioi_circuit")
def _ioi_circuit(spec: ExperimentSpec, run: Run, directory: Path) -> None:
    """Replicate the IOI circuit end to end and write down what was found

    The run records the headline numbers and nothing that is really a matrix.
    The grids -- per-head attribution, per-head causal effect, the role
    weights, the position map -- go into the artifact beside it, because a
    metrics dict with one entry per head is a file nobody reads and a chart
    nobody can redraw.

    The artifact is the shareable half of the result: run.json says what this
    machine did, and circuit.mia says what was found, in a form another lab
    can load without this repository.
    """
    adapter = require_circuits(load_adapter(spec.model.resolve()))
    dataset = build_ioi(
        adapter, size=spec.ioi.size, seed=spec.seed, frame=spec.ioi.frame, corruption=spec.ioi.corruption,
    )
    behaviour = evaluate_ioi(adapter, dataset)
    attribution = direct_logit_attribution(adapter, dataset)
    effects = patch_heads(adapter, dataset)
    roles = classify_heads(adapter, dataset)
    circuit = discover(
        adapter, dataset, threshold=spec.ioi.threshold, max_heads=spec.ioi.max_heads, effects=effects,
    )
    report = verify(adapter, dataset, circuit)
    grid = patch_residual(adapter, dataset) if spec.ioi.residual_patch else None

    run.record(
        accuracy=behaviour.accuracy,
        clean_logit_difference=behaviour.clean,
        corrupted_logit_difference=behaviour.corrupted,
        span=behaviour.span,
        attribution_remainder=attribution.residual,
        n_heads=len(circuit),
        faithfulness=report.faithfulness,
        necessity=report.necessity,
        n_spare=len(report.spare(tolerance=spec.ioi.tolerance)),
        n_prompts=len(dataset),
    )
    if grid is not None:
        best_layer, best_position, best_recovery = grid.best()
        run.record(best_patch_layer=best_layer, best_patch_position=best_position, best_patch_recovery=best_recovery)

    name = f"circuit{SUFFIX}"
    from_circuit(
        adapter.cfg, dataset, attribution, effects, report, roles=roles, grid=grid,
        tokens=dataset.token_labels(adapter), landmarks=dataset.landmarks(adapter),
        name=f"{dataset.name}-{adapter.cfg.id}",
    ).save(str(directory / name))
    run.produce("artifact", name)

def run_directory(spec: ExperimentSpec, run: Run, root: Optional[str] = None) -> Path:
    """Where a run's files go: <root>/<experiment>/<run id>

    The experiment name is a directory rather than part of the run id because a
    root accumulates runs forever, and the question asked of it is almost always
    "what did *this* experiment do" rather than "what ran on Tuesday". Grouping
    answers that with `ls` instead of a filter.

    It is computed here and nowhere else. Three callers need this path -- the
    runner that writes it, the CLI that reports it, the Hydra entry point that
    prints it -- and three copies of a layout is three chances to print a
    directory that does not exist.
    """
    return Path(root or spec.output.root) / spec.experiment / run.run_id

def run_experiment(spec: ExperimentSpec, root: Optional[str] = None) -> Run:
    """Execute a spec, writing everything it produced into its own directory

    The run is saved once before the work starts and again after it ends, so a
    process killed halfway leaves a directory that says 'running' rather than
    nothing at all.
    """
    if spec.kind not in EXPERIMENTS:
        raise SpecError(f"unknown experiment kind '{spec.kind}'; known kinds are {sorted(EXPERIMENTS)}")

    run = Run.start(experiment=spec.experiment, kind=spec.kind, spec_hash=spec.spec_hash, params=spec.as_dict())
    directory = run_directory(spec, run, root)
    directory.mkdir(parents=True, exist_ok=True)
    save_spec(spec, str(directory / "spec.yaml"))
    run.save(str(directory))

    try:
        with measure(items=1) as cost:
            EXPERIMENTS[spec.kind](spec, run, directory)
    except Exception as error:
        run.finish(error=error)
        run.save(str(directory))
        raise
    run.record(seconds=cost[0].seconds).finish()
    run.save(str(directory))
    return run
