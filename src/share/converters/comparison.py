from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...core.config import ModelConfig
from ...data.tasks import CircuitTask
from ...methods.circuits import HeadId
from ...methods.comparison import Consistency, Specificity, TechniqueComparison
from ..definitions import describe
from ..schema.artifact import Artifact
from ..schema.control import Control
from ..schema.controls import Controls
from ..schema.metric import Metric
from ..schema.node import Node
from ..schema.payload import Payload
from ..schema.site import Site
from ..schema.span import Span
from ..schema.vocabulary import Component, NodeComponent, Position
from .common import model_ref

"""
A comparison of circuit-finding techniques, packaged so the comparison itself
survives the trip.

This is still a circuit artifact -- same kind, same required grids, same span
-- and it carries three things a single-technique circuit cannot say. Every
technique's score for every head, each in its own payload because they are not
in the same units and a tensor that mixes units is the thing Payload exists to
prevent. The two agreement matrices, which are what "the techniques agree"
means as a number. And the two control slots filled: a random circuit of the
same size, and this circuit ablated on the other tasks.

Those two slots are the point. `Controls` has always shipped them empty with a
docstring explaining that a circuit measured only on its own task is not shown
to be about that task; an artifact written from here is one where somebody
actually checked. An artifact whose cross_task list is empty and one whose
cross_task list says "the damage was the same on three other tasks" are
different claims, and they must not be byte-identical.

A common pipe could be: compare_techniques | specificity | from_comparison | save
"""

LOGIT_DIFFERENCE = "logit_difference"
REQUIRES = {"attribution": "head_attribution", "patching": "head_effects"}

def _task_card(task: CircuitTask, tokens: Sequence[str], landmarks: Dict[str, int]) -> Dict:
    """What the prompts were, read off whichever kind of task this is

    Every field is optional because a task is not required to be an
    IOIDataset: a TemplateTask states its variants where IOI states its
    balance, and a reader should expect neither and understand either.
    """
    card = {
        "name": getattr(task, "name", "unnamed"),
        "frame": getattr(task, "frame", ""),
        "corruption": getattr(task, "corruption", ""),
        "n": len(task),
        "tokens": list(tokens),
        "landmarks": dict(landmarks),
    }
    for field in ("description", "balance", "variants"):
        value = getattr(task, field, None)
        if value is not None:
            card[field] = value
    examples = getattr(task, "examples", [])
    if examples:
        first = examples[0]
        card["example"] = {
            "clean": first.clean,
            "corrupted": first.corrupted,
            "answer": getattr(first, "answer", getattr(first, "io", "")),
            "distractor": getattr(first, "distractor", getattr(first, "subject", "")),
        }
    return card

def _scored(comparison: TechniqueComparison, head: HeadId) -> Dict[str, float]:
    """Every technique's score for one head, keyed by the technique that produced it"""
    scores = {}
    for result in comparison.results:
        ranking = result.ranking
        if head[0] in ranking.layers:
            scores[result.method] = float(ranking.scores[ranking.layers.index(head[0]), head[1]])
    return scores

def from_comparison(
    cfg: ModelConfig,
    task: CircuitTask,
    comparison: TechniqueComparison,
    reference: str = "patching",
    consistency: Optional[Consistency] = None,
    specificity: Optional[Specificity] = None,
    tokens: Optional[Sequence[str]] = None,
    landmarks: Optional[Dict[str, int]] = None,
    task_key: Optional[str] = None,
    name: Optional[str] = None,
) -> Artifact:
    """Package a technique comparison: every ranking, both agreement matrices, and the controls

    `reference` is the technique whose circuit the nodes are marked as being
    in. It defaults to patching because patching is the measurement the others
    approximate, and naming it in the card is what stops a reader assuming the
    cheap technique's circuit was the one checked.

    A comparison that did not run attribution and patching cannot be written
    as a circuit artifact: those two grids are what the format requires of the
    kind, and substituting a first-order estimate for a measured effect would
    be a lie the units would not catch.

    `task_key` is how this task is named inside the specificity result, which
    is not always what the task calls itself -- an IOI dataset names itself
    after its corruption ('ioi-abc') while the cross-task sweep keys it by the
    registry name ('ioi'). Getting that wrong used to write an artifact whose
    cross-task controls were silently empty, which reads as a check nobody
    ran, so a key that does not match is refused rather than skipped.
    """
    found = comparison.by_method()
    missing = [method for method in REQUIRES if method not in found]
    if missing:
        raise ValueError(
            f"a circuit artifact carries {sorted(REQUIRES.values())}, so the comparison has to include "
            f"{sorted(REQUIRES)} and this one is missing {missing}; run them, or share the comparison as "
            "a table rather than as a circuit"
        )
    if reference not in found:
        raise ValueError(
            f"'{reference}' is not one of the techniques compared ({comparison.methods}), so its circuit "
            "cannot be the one the nodes are marked against"
        )

    key = task_key or task.name
    if specificity is not None and key not in specificity.tasks:
        raise ValueError(
            f"this task is '{key}' and the cross-task sweep covers {specificity.tasks}; pass task_key so the "
            "controls can say what this circuit cost the other tasks, rather than shipping them empty"
        )

    anchor = found[reference]
    baselines = anchor.ranking.baselines or found["patching"].ranking.baselines
    span = Span(metric=LOGIT_DIFFERENCE, clean=baselines.clean, corrupted=baselines.corrupted)
    layers = found["patching"].ranking.layers
    positions = list(tokens if tokens is not None else [])
    marks = dict(landmarks if landmarks is not None else {})

    tensors: Dict[str, Payload] = {}
    for method, result in found.items():
        payload = Payload(values=result.ranking.scores.float(), axes=["layer", "head"], units=result.ranking.units)
        tensors[REQUIRES.get(method, f"scores_{method}")] = payload
    for which, units in (("overlap", "share"), ("order", "correlation")):
        tensors[f"technique_{which}"] = Payload(
            values=torch.tensor(comparison.matrix(which), dtype=torch.float32),
            axes=["method", "against"], units=units,
            labels={"method": comparison.methods, "against": comparison.methods},
        )
    if specificity is not None:
        tensors["cross_task_damage"] = Payload(
            values=torch.tensor(specificity.matrix(), dtype=torch.float32),
            axes=["measured_on", "circuit_of"], units="share",
            labels={"measured_on": specificity.tasks, "circuit_of": specificity.tasks},
        )

    frequency = consistency.frequency if consistency is not None else {}
    selected = {result.method: set(result.heads or [head for head, _ in result.ranking.ranked(comparison.count)])
                for result in comparison.results}
    union = sorted({head for heads in selected.values() for head in heads})
    nodes = []
    for layer, head in union:
        scores = _scored(comparison, (layer, head))
        # the measured patching effect again, under the name the circuit vocabulary
        # already uses, so a reader that knows `attribution`/`causal` finds both here.
        # It is taken from patching rather than from the reference: a first-order
        # estimate written under the name of a measurement is a lie about the method
        if "patching" in scores:
            scores["causal"] = scores["patching"]
        scores["techniques"] = float(sum((layer, head) in heads for heads in selected.values()))
        if frequency:
            scores["consistency"] = float(frequency.get((layer, head), 0.0))
        if anchor.report is not None and (layer, head) in anchor.report.minimality:
            scores["minimality"] = float(anchor.report.minimality[(layer, head)])
        nodes.append(Node(
            id=f"L{layer}H{head}", component=NodeComponent.HEAD, layer=layer, head=head,
            in_circuit=(layer, head) in selected[reference], scores=scores,
        ))

    metrics: Dict[str, Metric] = {}
    for method, result in found.items():
        for metric, value in (
            ("faithfulness", result.faithfulness),
            ("necessity", result.necessity),
            ("incompleteness", result.incompleteness),
            ("passes", float(result.ranking.passes)),
        ):
            if value != value:  # a check that was not run reports nan, and is left out rather than stored as one
                continue
            definition, units = describe(metric)
            metrics[f"{metric}:{method}"] = Metric(
                value=float(value),
                definition=f"{definition}; over the {comparison.count} heads '{method}' ranked highest",
                units=units,
            )
    # not describe("n_heads"): that definition says "how many heads the greedy search
    # kept", and nothing here grew a circuit. A metric borrowing a definition from a
    # method it did not run is the exact failure the definition field exists to prevent
    metrics["n_heads"] = Metric(
        value=float(comparison.count),
        definition=(
            "how many heads each technique was allowed to select, by rank; the same for every "
            "technique, because faithfulness climbs with the head count and a comparison at "
            "different sizes is a comparison of sizes"
        ),
        units="heads",
    )
    if consistency is not None:
        definition, units = describe("reuse")
        metrics["reuse"] = Metric(
            value=consistency.reuse,
            definition=(
                f"{definition}; P={consistency.presence:g} over {len(consistency.circuits)} per-example "
                f"circuits found by '{consistency.method}', against {consistency.chance:.3f} by chance"
            ),
            units=units,
        )
    if specificity is not None:
        definition, units = describe("margin")
        metrics["margin"] = Metric(
            value=specificity.margin(key),
            definition=f"{definition}; against {len(specificity.tasks) - 1} other tasks",
            units=units,
        )

    controls = _controls(comparison, specificity, key, reference)
    return Artifact(
        kind="circuit",
        id=name or f"{task.name}-comparison-{cfg.id}",
        model=model_ref(cfg, cfg.id),
        site=Site.at(layers, cfg.n_layers or 0, component=Component.HEAD_OUT, position=Position.ALL),
        task=_task_card(task, positions, marks),
        method=f"{', '.join(comparison.methods)} compared at {comparison.count} heads, reference '{reference}'",
        metrics=metrics,
        span=span,
        controls=controls,
        nodes=nodes,
        edges=[],
        tensors=tensors,
        notes=(
            f"Every technique ranked every head and the top {comparison.count} were checked the same way, so "
            "the differences between the faithfulness numbers are differences between techniques rather than "
            "between circuit sizes. Scores are stored one payload per technique because they are not in one "
            "unit: attribution is in logits along the direct path, patching is a fraction of the span, eap is "
            "a first-order estimate of that fraction. Compare them as orders and as sets, not by subtraction. "
            "No edges were measured: this says which heads matter, not which head feeds which."
        ),
    ).validate()

def _controls(
    comparison: TechniqueComparison, specificity: Optional[Specificity], task: str, reference: str
) -> Controls:
    """Fill the two slots that say what somebody tried to make this result go away with

    Both lists stay empty when nothing was run, which is the whole contract:
    an artifact that ablated its circuit on three other tasks and one that
    never considered it must not read the same.
    """
    random_baseline: List[Control] = []
    found = comparison.by_method()
    if "random" in found and found["random"].report is not None:
        random_baseline.append(Control(
            name=f"random circuit of {comparison.count} heads",
            metric="recovery",
            value=found["random"].faithfulness,
            notes=(
                "the same number of heads, drawn without looking at the model and restored the same way; "
                "a technique that does not clear this found the model rather than the task"
            ),
        ))
    cross_task: List[Control] = []
    if specificity is not None:
        for other in specificity.tasks:
            if other == task:
                continue
            cross_task.append(Control(
                name=other,
                metric="share",
                value=specificity.damage[(other, task)].damage,
                notes=(
                    f"this task's circuit mean-ablated on '{other}', as a share of that task's clean logit "
                    f"difference; '{other}' loses {specificity.own(other):.0%} to its own circuit"
                ),
            ))
        random = specificity.control[task]
        cross_task.append(Control(
            name=f"{task} (own task, random circuit)",
            metric="share",
            value=random.damage,
            notes="a random circuit of the same size mean-ablated on this task, as the floor the rest sit above",
        ))
    return Controls(cross_task=cross_task, random_baseline=random_baseline)

def head_pairs(nodes: Sequence[Node]) -> List[Tuple[int, int]]:
    """The (layer, head) pairs a card's nodes name, for a caller rebuilding a circuit"""
    return [(node.layer, node.head) for node in nodes if node.head is not None]
