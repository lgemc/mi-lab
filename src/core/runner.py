from pathlib import Path
from typing import Callable, Dict, Optional

from .adapter import load_adapter
from .metrics import measure
from .probing import difference_of_means, evaluate, sweep, train_probe
from .run import Run
from .spec import ExperimentSpec, SpecError, save_spec

"""
The runner turns a spec into a Run. It is the one place that knows the order
of operations -- resolve the model, build the data, split it, fit, evaluate,
write -- and every experiment type is a function registered against a kind,
so adding one is a registration rather than an edit here.

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
    provenance = {"model_id": adapter.cfg.id, "layer": layer, "dataset": dataset.name}

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

def run_experiment(spec: ExperimentSpec, root: Optional[str] = None) -> Run:
    """Execute a spec, writing everything it produced into its own directory

    The run is saved once before the work starts and again after it ends, so a
    process killed halfway leaves a directory that says 'running' rather than
    nothing at all.
    """
    if spec.kind not in EXPERIMENTS:
        raise SpecError(f"unknown experiment kind '{spec.kind}'; known kinds are {sorted(EXPERIMENTS)}")

    run = Run.start(experiment=spec.experiment, kind=spec.kind, spec_hash=spec.spec_hash, params=spec.as_dict())
    directory = Path(root or spec.output.root) / run.run_id
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
