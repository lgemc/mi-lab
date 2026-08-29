from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import torch

from ..provenance import stamp
from .controls import Controls
from .edge import Edge
from .errors import ArtifactError
from .metric import Metric
from .model import ModelRef
from .node import Node
from .payload import Payload
from .site import Site
from .span import Span
from .version import FORMAT, VERSION
from .vocabulary import NEEDS_SPAN, REQUIRED, Assumption, Component, Kind, NodeComponent, one_of

"""
The card itself: everything an interpretability result has to say about itself
before anyone else can use it, and the checks that say whether it did.

The pieces live in their own modules beside this one; what is here is how they
assemble and what makes an assembly wrong. Nothing in this package touches a
disk -- storage.py writes one of these down and reads it back, converters/
builds one out of what this lab measured.

Five decisions are what make an artifact loadable by a tool that did not write
it, and all five are enforced by validate():

- A tensor is never stored without its axes (see payload.py).
- The site is a depth fraction as well as an index (see site.py).
- Every claim carries the span it is a fraction of (see span.py).
- A metric carries the definition that produced it (see metric.py).
- A check that was not run is written down as not run (see controls.py).

A common pipe could be: from_circuit | validate | save | load
"""

@dataclass
class Artifact:
    """One shareable interpretability result: a card, and named tensors under it

    Four fields here are deliberately prose rather than closed sets, because
    closing them would mean a schema change before a new measurement could be
    shared. Each says what it accepts:

    - `method` is what was run, in a sentence. This repository writes
      "direct_logit_attribution + activation_patching, greedy search" for a
      circuit, "capture" for an activation map, the probe's own fitting method
      ("logistic", "difference_of_means") for a probe, and for a steering
      vector however the direction was obtained.
    - `task` is free-form, and the keys this repository writes are `name`,
      `task`, `frame`, `corruption`, `n`, `balance`, `tokens`, `landmarks` and
      `example`. A reader should expect none of them and understand any.
    - `provenance` is stamped rather than passed, and carries `tool`,
      `format_version`, `git_commit`, `git_dirty` and `torch`.
    - `notes` is prose for the reader, and is where a converter says what the
      numbers do not: which control was not run, which vector to steer with.

    `identifiability` is a list of Assumption. Empty is the common value and
    means no structural assumption was imposed, not that the question was
    overlooked.
    """
    kind: Kind
    id: str
    model: ModelRef
    site: Site = field(default_factory=Site)
    task: Dict[str, Any] = field(default_factory=dict)
    method: str = "unspecified"
    metrics: Dict[str, Metric] = field(default_factory=dict)
    span: Optional[Span] = None
    identifiability: List[Assumption] = field(default_factory=list)
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
        # coerced rather than tested for membership, so that REQUIRED and NEEDS_SPAN
        # can be keyed by the member: a (str, Enum) hashes by name, not by value, so
        # REQUIRED["circuit"] and REQUIRED[Kind.CIRCUIT] are not the same lookup
        kind = one_of(Kind, self.kind, f"artifact '{self.id}' kind")
        if self.version != VERSION:
            raise ArtifactError(
                f"artifact '{self.id}' is {FORMAT} v{self.version} and this reader is v{VERSION}; "
                "read it with a matching version rather than guessing at the difference"
            )
        one_of(Component, self.site.component, f"artifact '{self.id}' site.component")
        if len(self.site.layers) != len(self.site.fracs):
            raise ArtifactError(
                f"site names {len(self.site.layers)} layers but {len(self.site.fracs)} depth fractions; "
                "every layer has to carry the fraction it sits at, or the artifact will not transfer"
            )
        missing = [name for name in REQUIRED[kind] if name not in self.tensors]
        if missing:
            raise ArtifactError(
                f"a '{kind.value}' artifact must carry {sorted(REQUIRED[kind])}, and '{self.id}' "
                f"is missing {missing}"
            )
        if kind in NEEDS_SPAN and self.span is None:
            raise ArtifactError(
                f"a '{kind.value}' artifact reports recoveries, which are fractions of the span between a "
                "clean and a corrupted baseline; without the span they have no scale"
            )
        for name, metric in self.metrics.items():
            if not metric.definition.strip():
                raise ArtifactError(
                    f"metric '{name}' on '{self.id}' has a value but no definition; 'faithfulness' alone "
                    "names a logit-difference recovery here and a normalized KL reproduction elsewhere, and "
                    "the two are not comparable -- say which one this is"
                )
        for claimed in self.identifiability:
            one_of(Assumption, claimed, f"artifact '{self.id}' claims a structural assumption that")
        # a node's vocabulary is not a site's: 'head' is a part of the graph and
        # 'head_out' is a place to read from, and before these were separate types
        # nothing stopped a node claiming one of the site's names
        for node in self.nodes:
            one_of(NodeComponent, node.component, f"node '{node.id}' component")
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
        return f"{Kind(self.kind).value} '{self.id}' on {self.model.id} at {where}, {self.n_bytes / 1024:.1f} KiB"
