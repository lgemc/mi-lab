from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence

import torch

from .provenance import stamp

"""
What an interpretability result has to say about itself before anyone else can
use it: the model it was measured on, the place in that model, the data it was
measured over, and the baseline it is a fraction of. This module is the schema
and the checks on it, and nothing that touches a disk -- storage.py writes one
of these down and reads it back, converters/ builds one out of what this lab
measured.

Four decisions are what make an artifact loadable by a tool that did not
write it, and all four are enforced by validate():

- A tensor is never stored without its axes. `head_effects` is [layer, head]
  in recovery, `residual_patch` is [layer, position]; a bare 2-D float array
  named in a filename is one you will eventually transpose. Payload keeps the
  values, the axis names and the unit together and validate refuses a tensor
  whose axis names do not match its rank.
- The site is a depth fraction as well as an index, because layer 8 of a small
  model and layer 42 of a large one are the same place and only the fraction
  survives a model swap.
- Every claim carries the span it is a fraction of. A recovery of 0.9 against
  a corruption that moved nothing is noise scaled up to look like a finding,
  so `span` is required of anything that reports a recovery.
- A metric carries the definition that produced it. `faithfulness` names at
  least two different quantities in this field -- a logit-difference recovery
  under restoration here, a normalized KL reproduction elsewhere -- and two
  artifacts both reporting 0.9 are not comparable. Metric is to a number what
  Payload is to a tensor, for exactly the same reason.

A common pipe could be: from_circuit | validate | save | load
"""

FORMAT = "mia"
VERSION = "0.2"

KINDS = ("circuit", "probe", "steering_vector", "activation_map")

# What each kind has to carry to be that kind. An artifact missing these is not
# an incomplete artifact, it is a different thing wearing the label.
REQUIRED = {
    "circuit": ("head_attribution", "head_effects"),
    "probe": ("weight", "bias", "mean", "std"),
    "steering_vector": ("vector",),
    "activation_map": ("values",),
}

# Kinds whose numbers are a fraction of a corruption's span, and so cannot be
# read at all without it.
NEEDS_SPAN = ("circuit",)

COMPONENTS = ("residual", "head_out", "mlp_out", "attention")

# The structural assumptions that pick one direction out of the equivalence
# class of behaviourally identical ones. Steering directions are generically
# non-identifiable without at least one of them, so an artifact names which it
# imposed -- and an empty list is the honest answer, said out loud.
ASSUMPTIONS = ("independence", "sparsity", "multi_environment", "cross_layer")

class ArtifactError(ValueError):
    """Raised when an artifact is incomplete, self-contradictory, or not one at all"""


# ------------------------------------------------------------------- the card

@dataclass(frozen=True)
class ModelRef:
    """Which model the measurement was made on, in enough detail to repeat it

    hf_name and revision are what another lab resolves; the sizes are what a
    loader checks a payload against before it applies it to anything.
    """
    id: str
    hf_name: str
    revision: Optional[str] = None
    n_layers: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    dtype: str = "float32"

    @classmethod
    def from_config(cls, cfg) -> "ModelRef":
        """Read a ModelRef off a resolved ModelConfig"""
        return cls(
            id=cfg.id, hf_name=cfg.hf_name, n_layers=cfg.n_layers,
            d_model=cfg.d_model, n_heads=cfg.n_heads, dtype=cfg.dtype,
        )

@dataclass(frozen=True)
class Site:
    """Where in the model this was measured

    layers are absolute indices because that is what a hook needs, and fracs
    are the same layers as depth fractions because that is what transfers to a
    model of another size. Both are written down: deriving one from the other
    needs n_layers, and an artifact should be readable without resolving the
    checkpoint.
    """
    layers: List[int] = field(default_factory=list)
    fracs: List[float] = field(default_factory=list)
    component: str = "residual"
    position: str = "last"

    @classmethod
    def at(cls, layers: Sequence[int], n_layers: int, component: str = "residual", position: str = "last") -> "Site":
        """Build a site from absolute layers, stamping in the fractions they sit at

        The model's depth is required rather than optional. Without it the
        fraction cannot be computed, and a site carrying only indices is one
        the receiving lab cannot place -- which is the failure this whole
        field lives with today, not a corner case to paper over with a zero.
        """
        indices = [int(layer) for layer in layers]
        if not n_layers:
            raise ArtifactError(
                f"cannot place layers {indices} at a depth without knowing how many layers the model has; "
                "resolve the config against its checkpoint first, so the site says two thirds through "
                "rather than layer eight"
            )
        return cls(
            layers=indices,
            fracs=[round(layer / n_layers, 6) for layer in indices],
            component=component,
            position=position,
        )

@dataclass(frozen=True)
class Span:
    """The two baselines every fractional number here is measured between

    A recovery is 0 at `corrupted` and 1 at `clean`. Quoting one without the
    other is quoting a percentage of an unstated whole.
    """
    metric: str
    clean: float
    corrupted: float

    @property
    def span(self) -> float:
        return self.clean - self.corrupted

@dataclass(frozen=True)
class Node:
    """One component of a circuit, with what both halves of the study said about it

    scores holds every measurement made of this node, keyed by what it
    measured -- `attribution` for the direct-path logit contribution and
    `causal` for the patching recovery. Both are kept because they disagree,
    and a circuit format that stores only one of them throws away the finding.
    """
    id: str
    component: str
    layer: int
    head: Optional[int] = None
    role: Optional[str] = None
    in_circuit: bool = False
    scores: Dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class Edge:
    """A measured dependency between two nodes, as node ids and what was measured

    Nothing in this repository measures one yet, so circuits it emits carry an
    empty list. That is the honest state of the art rather than a gap in the
    schema: an artifact says it found no edges, instead of leaving a reader to
    guess whether the tool looked.
    """
    source: str
    target: str
    kind: str = "unmeasured"
    weight: float = 0.0

@dataclass(frozen=True)
class Metric:
    """A number that cannot be separated from the definition that produced it

    `faithfulness` names at least two different quantities in this field: the
    logit-difference recovery under restoration that this repository measures,
    and the normalized KL reproduction that the faithfulness critique asks
    for. Both are reported as a fraction near 0.9 and they are not comparable.

    So a metric is stored the way a tensor is -- with the thing that says what
    it means attached, not in a README beside it. `definition` is prose and
    `units` is prose, for the same reason Payload.units is: an enum is a list
    every new measurement has to be added to before it can be shared.
    """
    value: float
    definition: str
    units: str = "unspecified"

    def describe(self) -> Dict[str, Any]:
        """The card's entry for this metric: what it is, and what it is in"""
        return {"value": float(self.value), "definition": self.definition, "units": self.units}


@dataclass(frozen=True)
class Control:
    """One check run against a claim, and what it returned

    A control is not a metric: a metric describes the result, a control
    describes an attempt to make the result go away. Storing them apart is
    what lets `validate` insist the attempt was recorded even when nobody
    made one.
    """
    name: str
    metric: str
    value: float
    notes: str = ""


@dataclass(frozen=True)
class Controls:
    """The checks a claim was tested against, listed even when nothing was run

    Both slots default to empty and both are always written, for the reason
    `edges` is written empty: an artifact that ran no cross-task ablation and
    one that ran several are otherwise byte-identical, and the reader cannot
    tell "checked, and it held" from "nobody thought to check".

    `cross_task` is the one that matters most for a circuit. Ablating one
    task's circuit damages unrelated tasks about as much as its own, because
    circuits at this level are dominated by shared infrastructure -- so a
    circuit reporting only within-task numbers is not shown to be about its
    task at all.
    """
    cross_task: List[Control] = field(default_factory=list)
    random_baseline: List[Control] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether nothing at all was run, which is a thing a reader must be told"""
        return not self.cross_task and not self.random_baseline


@dataclass(frozen=True)
class Payload:
    """A tensor that cannot be separated from what its axes mean

    units is prose on purpose -- "recovery", "logits", "attention" -- because
    the alternative is an enum that every new measurement has to be added to
    before it can be shared.

    labels names the ticks along an axis: the token strings under a position
    axis, the role names under a role axis. It is what makes a heatmap
    redrawable by a tool that never saw the prompts, and it is the difference
    between shipping a figure and shipping the thing the figure was made of.
    """
    values: torch.Tensor
    axes: List[str] = field(default_factory=list)
    units: str = "unspecified"
    labels: Dict[str, List[str]] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        """The manifest's entry for this tensor: its shape, dtype, axes and unit"""
        return {
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype).removeprefix("torch."),
            "axes": list(self.axes),
            "units": self.units,
            "labels": {axis: list(names) for axis, names in self.labels.items()},
        }


@dataclass
class Artifact:
    """One shareable interpretability result: a card, and named tensors under it"""
    kind: str
    id: str
    model: ModelRef
    site: Site = field(default_factory=Site)
    task: Dict[str, Any] = field(default_factory=dict)
    method: str = "unspecified"
    metrics: Dict[str, Metric] = field(default_factory=dict)
    span: Optional[Span] = None
    identifiability: List[str] = field(default_factory=list)
    controls: Controls = field(default_factory=Controls)
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    tensors: Dict[str, Payload] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    version: str = VERSION

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if not self.provenance:
            self.provenance = stamp(VERSION)

    def tensor(self, name: str) -> torch.Tensor:
        """One payload's values by name, with a message naming what is there instead"""
        if name not in self.tensors:
            raise ArtifactError(f"artifact '{self.id}' has no tensor '{name}'; it carries {sorted(self.tensors)}")
        return self.tensors[name].values

    def score(self, name: str) -> float:
        """One metric's value by name, with a message naming what is there instead"""
        if name not in self.metrics:
            raise ArtifactError(f"artifact '{self.id}' has no metric '{name}'; it reports {sorted(self.metrics)}")
        return self.metrics[name].value

    @property
    def metric_values(self) -> Dict[str, float]:
        """Just the numbers, for a caller that already knows what they mean

        Anything reporting a metric to a human should reach for `metrics` and
        print the definition with it; this is for rebuilding an object whose
        own field is a plain mapping.
        """
        return {name: metric.value for name, metric in self.metrics.items()}

    @property
    def circuit_heads(self) -> List[Node]:
        """The nodes this artifact claims are in the circuit, in the order they were added"""
        return [node for node in self.nodes if node.in_circuit]

    @property
    def n_bytes(self) -> int:
        """What the payload weighs, which is the number that decides how it ships"""
        return sum(payload.values.numel() * payload.values.element_size() for payload in self.tensors.values())

    def validate(self) -> "Artifact":
        """Reject anything that cannot be read back and applied

        Every check here is a mistake that otherwise surfaces as a plausible
        wrong number somewhere downstream: a head grid whose rows are a subset
        of layers but are read as layer indices, a probe applied to a model of
        another width, a recovery quoted with no span behind it.
        """
        if self.kind not in KINDS:
            raise ArtifactError(f"unknown artifact kind '{self.kind}'; known kinds are {sorted(KINDS)}")
        if self.version != VERSION:
            raise ArtifactError(
                f"artifact '{self.id}' is {FORMAT} v{self.version} and this reader is v{VERSION}; "
                "read it with a matching version rather than guessing at the difference"
            )
        if self.site.component not in COMPONENTS:
            raise ArtifactError(
                f"unknown component '{self.site.component}'; known components are {sorted(COMPONENTS)}"
            )
        if len(self.site.layers) != len(self.site.fracs):
            raise ArtifactError(
                f"site names {len(self.site.layers)} layers but {len(self.site.fracs)} depth fractions; "
                "every layer has to carry the fraction it sits at, or the artifact will not transfer"
            )
        missing = [name for name in REQUIRED[self.kind] if name not in self.tensors]
        if missing:
            raise ArtifactError(
                f"a '{self.kind}' artifact must carry {sorted(REQUIRED[self.kind])}, and '{self.id}' "
                f"is missing {missing}"
            )
        if self.kind in NEEDS_SPAN and self.span is None:
            raise ArtifactError(
                f"a '{self.kind}' artifact reports recoveries, which are fractions of the span between a "
                "clean and a corrupted baseline; without the span they have no scale"
            )
        for name, metric in self.metrics.items():
            if not metric.definition.strip():
                raise ArtifactError(
                    f"metric '{name}' on '{self.id}' has a value but no definition; 'faithfulness' alone "
                    "names a logit-difference recovery here and a normalized KL reproduction elsewhere, and "
                    "the two are not comparable -- say which one this is"
                )
        unknown_assumptions = [name for name in self.identifiability if name not in ASSUMPTIONS]
        if unknown_assumptions:
            raise ArtifactError(
                f"artifact '{self.id}' claims structural assumptions {unknown_assumptions} that this format "
                f"does not know; the ones that break the equivalence class are {sorted(ASSUMPTIONS)}"
            )
        for name, payload in self.tensors.items():
            if len(payload.axes) != payload.values.dim():
                raise ArtifactError(
                    f"tensor '{name}' has rank {payload.values.dim()} but {len(payload.axes)} axis names "
                    f"{payload.axes}; a tensor stored without its axes is one the next reader transposes"
                )
            sizes = dict(zip(payload.axes, payload.values.shape, strict=True))
            for axis, names in payload.labels.items():
                if axis not in sizes:
                    raise ArtifactError(
                        f"tensor '{name}' labels an axis '{axis}' it does not have; its axes are {payload.axes}"
                    )
                if len(names) != sizes[axis]:
                    raise ArtifactError(
                        f"tensor '{name}' has {sizes[axis]} entries along '{axis}' but {len(names)} labels "
                        "for them; labels that do not line up name the wrong column"
                    )
        self._check_shapes()
        return self

    def _check_shapes(self) -> None:
        """Check every payload against the axes it claims and the model it names"""
        widths = {"layer": len(self.site.layers) or None, "head": self.model.n_heads, "d_model": self.model.d_model}
        for name, payload in self.tensors.items():
            for axis, size in zip(payload.axes, payload.values.shape, strict=True):
                expected = widths.get(axis)
                if expected is not None and size != expected:
                    raise ArtifactError(
                        f"tensor '{name}' is {size} long on its '{axis}' axis but this artifact's "
                        f"{axis} count is {expected}; a grid measured over a subset must say so in its site"
                    )

    def __str__(self) -> str:
        where = f"L{self.site.layers[0]}" if len(self.site.layers) == 1 else f"{len(self.site.layers)} layers"
        return f"{self.kind} '{self.id}' on {self.model.id} at {where}, {self.n_bytes / 1024:.1f} KiB"
