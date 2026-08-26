import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from safetensors.torch import load_file, save_file

"""
An interpretability result is not a number, it is a number plus the model it
was measured on, the place in that model, the data it was measured over, and
the baseline it is a fraction of. Weights ship as safetensors and datasets
ship as a card plus rows; circuits, probes, steering vectors and activation
maps ship as a paper and a notebook, which is why nobody can load anyone
else's. This module is the missing envelope.

An artifact is a directory: `artifact.json` is the card and every number lives
in `tensors.safetensors` beside it. The card is stdlib-readable JSON, so
deciding whether an artifact is worth downloading never requires torch, and
the tensors are memory-mappable, so a large one need not be read whole.

Three decisions are what make it loadable by a tool that did not write it:

- A tensor is never stored without its axes. `head_effects` is [layer, head]
  in recovery, `residual_patch` is [layer, position]; a bare 2-D float array
  named in a filename is one you will eventually transpose. Payload keeps the
  values, the axis names and the unit together and `save` refuses to write a
  tensor whose axis names do not match its rank.
- The site is a depth fraction as well as an index, because layer 8 of a small
  model and layer 42 of a large one are the same place and only the fraction
  survives a model swap.
- Every claim carries the span it is a fraction of. A recovery of 0.9 against
  a corruption that moved nothing is noise scaled up to look like a finding,
  so `span` is required of anything that reports a recovery.

A common pipe could be: from_circuit | save | load | to_probe
"""

FORMAT = "mia"
VERSION = "0.1"

MANIFEST = "artifact.json"
TENSORS = "tensors.safetensors"
SUFFIX = ".mia"

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
    metrics: Dict[str, float] = field(default_factory=dict)
    span: Optional[Span] = None
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
            self.provenance = stamp()

    def tensor(self, name: str) -> torch.Tensor:
        """One payload's values by name, with a message naming what is there instead"""
        if name not in self.tensors:
            raise ArtifactError(f"artifact '{self.id}' has no tensor '{name}'; it carries {sorted(self.tensors)}")
        return self.tensors[name].values

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

    # ------------------------------------------------------------ on disk

    def to_manifest(self) -> Dict[str, Any]:
        """The card, as the plain data that gets written to artifact.json"""
        return {
            "format": FORMAT,
            "version": self.version,
            "kind": self.kind,
            "id": self.id,
            "created_at": self.created_at,
            "model": asdict(self.model),
            "site": asdict(self.site),
            "task": self.task,
            "measurement": {
                "method": self.method,
                "span": asdict(self.span) if self.span is not None else None,
                "metrics": self.metrics,
            },
            "graph": {
                "nodes": [asdict(node) for node in self.nodes],
                "edges": [asdict(edge) for edge in self.edges],
            },
            "tensors": {name: payload.describe() for name, payload in self.tensors.items()},
            "provenance": self.provenance,
            "notes": self.notes,
        }

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any], tensors: Dict[str, torch.Tensor]) -> "Artifact":
        """Rebuild an artifact from its card and the tensors that were stored beside it"""
        unknown = set(manifest) - {
            "format", "version", "kind", "id", "created_at", "model", "site",
            "task", "measurement", "graph", "tensors", "provenance", "notes",
        }
        if unknown:
            raise ArtifactError(
                f"artifact.json has unknown keys {sorted(unknown)}; a key this reader does not know is a "
                "claim it would silently drop"
            )
        if manifest.get("format") != FORMAT:
            raise ArtifactError(f"not a {FORMAT} artifact: format is {manifest.get('format')!r}")

        measurement = manifest.get("measurement") or {}
        graph = manifest.get("graph") or {}
        described = manifest.get("tensors") or {}

        stored, described_names = set(tensors), set(described)
        if stored != described_names:
            raise ArtifactError(
                f"the card and the tensor file disagree: card-only {sorted(described_names - stored)}, "
                f"file-only {sorted(stored - described_names)}"
            )

        span = measurement.get("span")
        return cls(
            kind=manifest["kind"],
            id=manifest["id"],
            model=ModelRef(**manifest["model"]),
            site=Site(**manifest["site"]),
            task=manifest.get("task") or {},
            method=measurement.get("method", "unspecified"),
            metrics=measurement.get("metrics") or {},
            span=Span(**span) if span else None,
            nodes=[Node(**node) for node in graph.get("nodes", [])],
            edges=[Edge(**edge) for edge in graph.get("edges", [])],
            tensors={
                name: Payload(
                    values=tensors[name],
                    axes=described[name]["axes"],
                    units=described[name]["units"],
                    labels=described[name].get("labels") or {},
                )
                for name in described
            },
            provenance=manifest.get("provenance") or {},
            notes=manifest.get("notes", ""),
            created_at=manifest.get("created_at", ""),
            version=manifest.get("version", VERSION),
        )

    def save(self, path: str) -> str:
        """Write the artifact as a directory: the card, and the tensors beside it

        Validation happens here rather than at read time as well, so an
        artifact that is wrong never leaves the machine that made it.
        """
        self.validate()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / MANIFEST).write_text(json.dumps(self.to_manifest(), indent=2, sort_keys=True))
        # the header is duplicated into the tensor file's own metadata so a
        # tensors.safetensors that got separated from its card still says what it is
        save_file(
            {name: payload.values.contiguous() for name, payload in self.tensors.items()},
            target / TENSORS,
            metadata={"format": FORMAT, "version": self.version, "kind": self.kind, "id": self.id},
        )
        return str(target)

    @classmethod
    def load(cls, path: str) -> "Artifact":
        """Read an artifact back from its directory, checking the card against the tensors"""
        source = Path(path)
        card, payload = source / MANIFEST, source / TENSORS
        if not card.exists():
            raise ArtifactError(f"no {MANIFEST} in {source}; a {FORMAT} artifact is a directory containing one")
        if not payload.exists():
            raise ArtifactError(f"{source} has a card but no {TENSORS}; its numbers are missing")
        try:
            manifest = json.loads(card.read_text())
        except json.JSONDecodeError as error:
            raise ArtifactError(f"{card} is not valid JSON") from error
        try:
            return cls.from_manifest(manifest, load_file(payload)).validate()
        except TypeError as error:
            raise ArtifactError(f"{card} does not describe a {FORMAT} v{VERSION} artifact: {error}") from error

    def __str__(self) -> str:
        where = f"L{self.site.layers[0]}" if len(self.site.layers) == 1 else f"{len(self.site.layers)} layers"
        return f"{self.kind} '{self.id}' on {self.model.id} at {where}, {self.n_bytes / 1024:.1f} KiB"

def find_artifacts(root: str) -> List[Artifact]:
    """Every artifact under a directory, skipping what does not read as one

    A directory that fails to load is skipped rather than fatal, the same way
    a half-written run does not make a listing unusable.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found = []
    for directory in sorted(base.rglob(f"*{SUFFIX}")):
        try:
            found.append(Artifact.load(str(directory)))
        except ArtifactError:
            continue
    return found

def stamp(**extra: Any) -> Dict[str, Any]:
    """The provenance every artifact gets: which tool, which commit, and whether it was clean

    `dirty` is the field that keeps the commit honest. A hash recorded from a
    tree with uncommitted edits names code that never existed, and an artifact
    whose provenance is confidently wrong is worse than one with none.
    """
    described = _git("describe", "--always", "--dirty")
    record: Dict[str, Any] = {
        "tool": "mi-lab",
        "format_version": VERSION,
        "git_commit": (described or "").removesuffix("-dirty") or None,
        "git_dirty": bool(described and described.endswith("-dirty")),
        "torch": torch.__version__,
    }
    record.update(extra)
    return record

def _git(*args: str) -> Optional[str]:
    """Ask git something about this checkout, or None if there is no answer to be had"""
    try:
        result = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
