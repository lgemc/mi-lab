from pathlib import Path

from ...experiment.run import Run
from ...experiment.runner import register_experiment
from ...experiment.spec import ExperimentSpec
from ...methods.circuits.attribution import direct_logit_attribution
from ...methods.circuits.patching import patch_heads, patch_residual
from ...methods.circuits.search import discover
from ...methods.circuits.verify import verify
from ...model.adapter import load_adapter, require_circuits
from ...share import storage
from ...share.converters.circuit import from_circuit
from .analysis.roles import classify_heads
from .data.ioi import build_ioi
from .data.ioi import evaluate as evaluate_ioi

"""
The experiment kinds that are about this domain in particular.

`experiment/runner.py` holds the order of operations -- resolve the model,
build the data, fit, evaluate, write -- and dispatches on a kind registered
against it. Two of the kinds it ships are domain-free: a probe sweep is a probe
over activations, and `circuit_comparison` runs every technique over every
registered task without naming one. `ioi_circuit` is not: it builds an IOI
dataset, classifies the four attention movements IOI is made of, and packages a
card that quotes a name and a distractor.

So it registers here, and the registry is the extension point it always was.
The runner does not import this module; `src/plugins.py` does, on the first
lookup, which is what keeps the kernel from importing a domain.

A common pipe could be: compose_spec | run_experiment | from_circuit | save
"""


def ioi_card(dataset) -> dict:
    """The fields worth quoting on an IOI artifact's task card

    `share/converters/circuit.py` writes what every CircuitTask has -- a name, a
    modality, a size, the positions -- and takes the rest as data. This is the
    rest, for this task, written where knowing that a prompt has an answer and a
    distractor in it is allowed.
    """
    return {
        "task": "indirect object identification",
        "frame": dataset.frame,
        "corruption": dataset.corruption,
        "balance": dataset.balance,
        "example": {
            "clean": dataset.examples[0].clean,
            "corrupted": dataset.examples[0].corrupted,
            "answer": dataset.examples[0].io,
            "distractor": dataset.examples[0].subject,
        } if len(dataset) else {},
    }


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

    name = f"circuit{storage.SUFFIX}"
    storage.save(
        from_circuit(
            adapter.cfg, dataset, attribution, effects, report, roles=roles, grid=grid,
            tokens=dataset.labels(adapter), landmarks=dataset.landmarks(adapter),
            name=f"{dataset.name}-{adapter.cfg.id}", description=ioi_card(dataset),
        ),
        str(directory / name),
    )
    run.produce("artifact", name)
