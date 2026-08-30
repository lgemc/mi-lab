import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.metrics import jaccard, spearman
from ..data.tasks import CircuitTask
from ..model.adapter import require_circuits
from .circuits import (
    Ablation,
    CircuitError,
    CircuitReport,
    Completeness,
    HeadId,
    ablate,
    behaviour,
    completeness,
    donor_bank,
    verify,
)
from .discovery import TECHNIQUES, DiscoveryError, Ranking, rank, technique_names

"""
Three questions about a circuit that finding one does not answer.

**Do the techniques agree?** Attribution, patching, ablation and their
gradient approximation are four different measurements, and a claim about a
model that changes with the technique is a claim about the technique. So they
are all run on one task, at one size, and compared as orderings (rank
correlation) and as sets (overlap) -- never by subtracting scores that are not
in the same units. Alongside each one's faithfulness, necessity, minimality
and completeness, and what it cost in forward passes: a technique that agrees
with patching at a thirtieth of the price is the whole argument for it.

**Is the circuit the same circuit twice?** Consistency. A circuit found on the
whole batch is an average, and an average can be made of components that no
single example uses. So the technique is run per example, the components
appearing in at least P of those circuits are the shared set, and reuse@P is
the mean share of a per-example circuit that lies inside it. Low reuse means
the batch-level circuit is a summary of many different circuits and should not
be described as "the" circuit for the task.

**Is it about the task?** Specificity, and this is the one that stings. Take a
circuit found on one task and ablate it on another. If the damage is about the
same, the circuit is machinery the model uses for everything and the task
label on it was never earned. This is measured in damage rather than recovery
because recovery is a fraction of one task's own corruption span, which the
other task does not have.

None of these three replace faithfulness. They are the questions a faithful
circuit can still fail, and every one of them fails quietly.

A common pipe could be: build_task | compare_techniques | consistency | specificity
"""

# ------------------------------------------------------- do the techniques agree

@dataclass
class TechniqueResult:
    """What one technique found, what it is worth, and what it cost"""
    method: str
    ranking: Ranking
    report: Optional[CircuitReport] = None
    coverage: Optional[Completeness] = None

    @property
    def heads(self) -> List[HeadId]:
        return list(self.report.circuit.heads) if self.report is not None else []

    @property
    def faithfulness(self) -> float:
        return self.report.faithfulness if self.report is not None else float("nan")

    @property
    def necessity(self) -> float:
        return self.report.necessity if self.report is not None else float("nan")

    @property
    def incompleteness(self) -> float:
        return self.coverage.incompleteness if self.coverage is not None else float("nan")

    def __str__(self) -> str:
        return (
            f"{self.method:<12} faithfulness {self.faithfulness:+.3f}  necessity {self.necessity:+.3f}  "
            f"incompleteness {self.incompleteness:.3f}  {self.ranking.passes} passes"
        )

@dataclass
class TechniqueComparison:
    """Every technique's answer to one task, side by side

    The two matrices are the comparison. `order` is rank correlation over
    every head, which is the only honest comparison between a score in logits
    and a score in recovery; `overlap` is how much the selected circuits
    actually share. They can disagree -- two techniques can order the tail
    identically and still pick different heads at the top -- and that
    disagreement is worth seeing, which is why both are kept.
    """
    task: str
    count: int
    results: List[TechniqueResult]
    order: Dict[Tuple[str, str], float] = field(default_factory=dict)
    overlap: Dict[Tuple[str, str], float] = field(default_factory=dict)

    @property
    def methods(self) -> List[str]:
        return [result.method for result in self.results]

    def by_method(self) -> Dict[str, TechniqueResult]:
        return {result.method: result for result in self.results}

    def matrix(self, which: str = "overlap") -> List[List[float]]:
        """The pairwise numbers as a square, in `methods` order, for a chart or a payload"""
        table = {"overlap": self.overlap, "order": self.order}
        if which not in table:
            raise DiscoveryError(f"unknown matrix '{which}'; this comparison holds {sorted(table)}")
        pairs = table[which]
        return [
            [1.0 if left == right else pairs.get((left, right), pairs.get((right, left), float("nan")))
             for right in self.methods]
            for left in self.methods
        ]

    def __str__(self) -> str:
        return f"{self.task}: {len(self.results)} techniques at {self.count} heads"

def compare_techniques(
    adapter,
    task: CircuitTask,
    methods: Optional[Sequence[str]] = None,
    count: int = 8,
    check: bool = True,
    samples: int = 6,
    seed: int = 0,
    **options,
) -> TechniqueComparison:
    """Run every technique on one task and score all of them the same way

    Every circuit is the same size, because faithfulness climbs with the head
    count and a comparison at different sizes is a comparison of sizes. The
    checks are then the same three questions asked of each: is this set enough
    (faithfulness), does the model need it (necessity), and does it break the
    way the model breaks (completeness).

    `check` off leaves the rankings and their agreement, which is the cheap
    half: verifying five circuits costs more forward passes than finding four
    of them.
    """
    adapter = require_circuits(adapter)
    chosen = list(methods if methods is not None else technique_names())
    unknown = [name for name in chosen if name not in TECHNIQUES]
    if unknown:
        raise DiscoveryError(f"unknown techniques {unknown}; known techniques are {technique_names()}")
    if not chosen:
        raise DiscoveryError("a comparison needs at least one technique")

    results = []
    for method in chosen:
        ranking = rank(method, adapter, task, seed=seed, **options)
        result = TechniqueResult(method=method, ranking=ranking)
        if check:
            circuit = ranking.select(count=count)
            result.report = verify(adapter, task, circuit)
            result.coverage = completeness(
                adapter, task, circuit, samples=samples, seed=seed, reference=result.report.baselines
            )
        results.append(result)

    selections = {
        result.method: result.heads or [head for head, _ in result.ranking.ranked(count)]
        for result in results
    }
    order, overlap = {}, {}
    for index, left in enumerate(chosen):
        for right in chosen[index + 1:]:
            order[(left, right)] = spearman(
                results[index].ranking.flat(), results[chosen.index(right)].ranking.flat()
            )
            overlap[(left, right)] = jaccard(selections[left], selections[right])
    return TechniqueComparison(
        task=task.name, count=count, results=results, order=order, overlap=overlap
    )

# --------------------------------------------------- is it the same circuit twice

@dataclass
class Consistency:
    """Whether one technique finds the same components example after example

    `shared` is the set of heads appearing in at least `presence` of the
    per-example circuits, and `reuse` is the mean share of one example's
    circuit that lies inside it. Reuse near 1 means the batch-level circuit is
    a real description of what happens on each example; reuse near the
    selection size over the model means the batch circuit is an average of
    circuits that have little to do with each other.
    """
    method: str
    task: str
    count: int
    presence: float
    circuits: List[List[HeadId]]
    total_heads: int

    @property
    def frequency(self) -> Dict[HeadId, float]:
        """Share of the per-example circuits each head appeared in"""
        counts: Dict[HeadId, int] = {}
        for circuit in self.circuits:
            for head in set(circuit):
                counts[head] = counts.get(head, 0) + 1
        return {head: count / len(self.circuits) for head, count in counts.items()}

    @property
    def shared(self) -> List[HeadId]:
        """The heads that turned up in at least `presence` of the examples"""
        return sorted(head for head, share in self.frequency.items() if share >= self.presence)

    @property
    def reuse(self) -> float:
        """Mean share of a per-example circuit lying inside the shared set"""
        shared = set(self.shared)
        return sum(len(shared & set(circuit)) / len(circuit) for circuit in self.circuits) / len(self.circuits)

    @property
    def chance(self) -> float:
        """What reuse two independent selections of this size would reach on their own

        A share K of the components drawn twice at random overlaps at about
        K / (2 - K), which is a few percent at the sizes circuits are reported
        at. Reuse is only a finding above this line.
        """
        share = self.count / self.total_heads if self.total_heads else 0.0
        return share / (2 - share) if share else 0.0

    def __str__(self) -> str:
        return (
            f"{self.method} on {self.task}: {len(self.circuits)} per-example circuits of {self.count}, "
            f"{len(self.shared)} shared at P={self.presence:g}, reuse {self.reuse:.2f} (chance {self.chance:.2f})"
        )

def consistency(
    adapter,
    task: CircuitTask,
    method: str = "eap",
    count: int = 8,
    presence: float = 0.5,
    examples: Optional[int] = None,
    **options,
) -> Consistency:
    """Find a circuit per example and ask how much of each one the examples share

    Per-example is the whole point and it is what makes the technique choice
    matter here: this runs the technique once per example, so a technique
    costing a forward pass per head costs that again for every example.
    'eap' is the default because it is the one whose cost does not scale with
    the model's head count, which is the same reason the recent literature
    runs consistency with it.
    """
    adapter = require_circuits(adapter)
    if not 0 < presence <= 1:
        raise CircuitError(f"presence is a share of the examples, so it must be in (0, 1]; got {presence}")
    wanted = min(len(task), examples if examples is not None else len(task))
    if wanted < 2:
        raise CircuitError(
            f"consistency compares circuits across examples and this task offers {wanted}; build a larger one"
        )

    circuits = []
    total = 0
    for index in range(wanted):
        ranking = rank(method, adapter, task.subset([index]), **options)
        circuits.append(ranking.select(count=count).heads)
        # kept from inside the loop rather than read off the last ranking: the chance
        # line divides by it, and a variable that outlived its loop is how that becomes
        # a number nobody can trace
        total = len(ranking.heads())
    return Consistency(
        method=method, task=task.name, count=count, presence=presence, circuits=circuits,
        total_heads=total,
    )

# ------------------------------------------------------------ is it about the task

@dataclass
class Specificity:
    """What each task's circuit costs every other task

    `damage[(measured, circuit)]` is the ablation of one task's circuit
    measured on another task's prompts. The diagonal is the circuit ablated on
    its own task, which is the number a paper reports; the off-diagonal is the
    one that decides whether the circuit is about the task at all.

    `control` ablates a random circuit of the same size on each task, because
    "removing eight heads costs this model half its logit difference" is a
    sentence that might be true of any eight heads.
    """
    tasks: List[str]
    circuits: Dict[str, List[HeadId]]
    damage: Dict[Tuple[str, str], Ablation]
    control: Dict[str, Ablation]
    overlap: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def own(self, task: str) -> float:
        """Damage a task takes from its own circuit"""
        return self.damage[(task, task)].damage

    def others(self, task: str) -> float:
        """Mean damage a task takes from the circuits of every other task"""
        elsewhere = [self.damage[(task, other)].damage for other in self.tasks if other != task]
        return sum(elsewhere) / len(elsewhere) if elsewhere else float("nan")

    def margin(self, task: str) -> float:
        """How much more its own circuit costs a task than another task's does

        Near zero is the finding, not a null result: it says the circuit is
        shared machinery, and that a within-task faithfulness number was never
        evidence that the circuit is specific to the task.
        """
        return self.own(task) - self.others(task)

    def matrix(self) -> List[List[float]]:
        """Damage as a square in `tasks` order, rows measured on, columns ablated with"""
        return [[self.damage[(row, column)].damage for column in self.tasks] for row in self.tasks]

    def __str__(self) -> str:
        lines = [f"{len(self.tasks)} tasks, circuits of {len(next(iter(self.circuits.values()), []))} heads"]
        for task in self.tasks:
            lines.append(
                f"  {task:<14} own {self.own(task):+.2f}  others {self.others(task):+.2f}  "
                f"margin {self.margin(task):+.2f}  random {self.control[task].damage:+.2f}"
            )
        return "\n".join(lines)

def specificity(
    adapter,
    tasks: Dict[str, CircuitTask],
    circuits: Dict[str, Sequence[HeadId]],
    donor: str = "mean",
    seed: int = 0,
) -> Specificity:
    """Ablate every task's circuit on every task, and a random circuit on each as the control

    Mean ablation, not the corrupted twin: a corruption belongs to the task it
    was written for, so writing task A's corrupted activations in while
    measuring task B would be measuring the corruption rather than the
    circuit. Replacing a head's output with its own mean over the task's clean
    prompts removes what the head knew about the example while leaving the
    model in the distribution it was measured in, and it is the same operation
    for every task.
    """
    adapter = require_circuits(adapter)
    names = list(tasks)
    if len(names) < 2:
        raise CircuitError("specificity compares a circuit against another task, so it needs at least two tasks")
    missing = [name for name in names if name not in circuits]
    if missing:
        raise CircuitError(f"no circuit was given for {missing}; every task compared needs one")

    every = [(layer, head) for layer in range(adapter.cfg.n_layers) for head in range(adapter.cfg.n_heads)]
    rng = random.Random(seed)

    damage: Dict[Tuple[str, str], Ablation] = {}
    control: Dict[str, Ablation] = {}
    for measured in names:
        task = tasks[measured]
        clean = behaviour(adapter, task)
        # one donor bank per task, reused across every circuit ablated on it: the bank is
        # the task's own activations, and sharing one across tasks is the mistake here
        bank = donor_bank(adapter, task, donor)
        for source in names:
            damage[(measured, source)] = ablate(
                adapter, task, circuits[source], donor=donor, donors=bank, clean=clean
            )
        control[measured] = ablate(
            adapter, task, rng.sample(every, len(circuits[measured])), donor=donor, donors=bank, clean=clean
        )

    overlap = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap[(left, right)] = jaccard(circuits[left], circuits[right])
    return Specificity(
        tasks=names,
        circuits={name: [(int(layer), int(head)) for layer, head in circuits[name]] for name in names},
        damage=damage, control=control, overlap=overlap,
    )

def discover_across(
    adapter, tasks: Dict[str, CircuitTask], method: str = "eap", count: int = 8, **options
) -> Dict[str, List[HeadId]]:
    """Find one circuit per task with the same technique at the same size

    Same technique and same size on purpose: an overlap between a circuit of
    eight heads and one of thirty is mostly a statement about thirty.
    """
    return {
        name: rank(method, adapter, task, **options).select(count=count).heads
        for name, task in tasks.items()
    }
