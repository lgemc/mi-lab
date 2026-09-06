"""How a circuit is *run*, chosen by what its folder says it is.

A results directory is self-describing: `sheaf-<task>.json` records the config
it was pruned against, the density, the settings, and -- since the edge runs --
whether the circuit is a weight mask or a set of edges. Those two are not the
same object and cannot be run the same way. A weight mask is multiplied into
the parameters; an edge set is a hook around the forward pass that subtracts
what a source wrote from what a destination reads. The server used to know
only the first, and `discover` globbed for `sheaf-<task>-mask.pt`, so an edge
run was not rejected -- it was *invisible*, which is the worse of the two
failures because nothing anywhere says so.

So detection moves into the thing that knows how to run what it detects. A
backbone claims a directory by reading its artifact, loads whatever it needs,
and hands back a context manager that puts the circuit into the model for the
duration of a call. Adding a third kind -- a neuron mask, a subspace
projection, a LoRA delta -- is one class and one decorator, and neither the
scanner nor the HTTP layer learns its name.

Two things every backbone owes its caller:

  it restores      whatever it changed is undone on exit, including on the way
                   out of an exception, because the weights are shared state
                   and a half-applied circuit answers the *next* request too.
  it is honest     `n_open` and `n_total` are counted in the circuit's own
                   unit -- weights for a mask, edges for an edge set -- and
                   `describe` names the unit. A density is meaningless without
                   it, and 26% of edges is not 26% of weights.

A common pipe could be: scan | detect | load | applied | generate
"""

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple, Type

import torch

from ..methods.sheaves.mask import Packed, pack, unpack_one


class BackboneError(Exception):
    """Raised when a folder cannot be run: wrong model, wrong shapes, nothing recognisable"""


@dataclass
class Host:
    """The one model every circuit is applied to, and the clean weights to undo into

    Held by the server rather than by a backbone, because there is exactly one
    of it and a dozen circuits: cloning 3.4 GiB of parameters per circuit is
    the whole reason the previous implementation kept its `originals` outside
    the `Circuit` too.
    """
    adapter: object
    targets: Dict[str, torch.Tensor]
    originals: Dict[str, torch.Tensor]

    @property
    def device(self):
        return next(iter(self.originals.values())).device


@dataclass
class Spec:
    """What a folder says about itself before a single tensor is read

    This is the cheap half of discovery and it is why the server can start in
    seconds and pick up a circuit written after it started: scanning is a
    handful of small JSON reads, and nothing touches the GPU until a request
    names the circuit. The startup probe's "loading every mask takes a minute
    or two" was that cost paid for every circuit, including the ones nobody
    was going to ask for.
    """
    name: str
    directory: Path
    task: str
    kind: str
    # What `density` is a fraction *of*, carried on the spec and not only on
    # the loaded backbone: the listing shows circuits that have never been
    # read, and "26% of edges" and "26% of weights" are different claims. A
    # reader with only the number will assume the wrong one.
    unit: str = "weights"
    config: Optional[str] = None
    density: Optional[float] = None
    report: Dict[str, object] = field(default_factory=dict)
    # Set when the folder is recognisable but not runnable *here* -- almost
    # always a circuit pruned against a different model. Kept in the listing
    # with its reason rather than dropped, because a circuit that silently
    # fails to appear is the failure this module exists to end.
    problem: Optional[str] = None

    def describe(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "name": self.name, "task": self.task, "kind": self.kind,
            "unit": self.unit, "config": self.config, "directory": str(self.directory),
        }
        if self.density is not None:
            out["density"] = round(self.density, 6)
        if self.problem:
            out["problem"] = self.problem
        out.update(self.report)
        return out


class Backbone(ABC):
    """One way of putting a circuit into a model, and the detector that picks it"""

    kind: ClassVar[str]
    #: what `n_open` and `n_total` count, so a density is readable
    unit: ClassVar[str]

    def __init__(self, host: Host, spec: Spec) -> None:
        self.host = host
        self.spec = spec

    @classmethod
    @abstractmethod
    def claims(cls, directory: Path, task: str, record: Dict[str, object]) -> bool:
        """Whether this backbone can run the circuit in `directory`

        `record` is the parsed `sheaf-<task>.json`, or `{}` when there is
        none -- an old run that kept only its mask still has to be servable.
        """

    @abstractmethod
    def load(self) -> None:
        """Read the circuit onto the model's device. Called once, lazily, on first use"""

    @abstractmethod
    def applied(self) -> Iterator[None]:
        """A context in which the model runs as this circuit, restored on exit"""

    @property
    @abstractmethod
    def n_open(self) -> int:
        ...

    @property
    @abstractmethod
    def n_total(self) -> int:
        ...

    @property
    def density(self) -> float:
        return self.n_open / self.n_total if self.n_total else 0.0

    def describe(self) -> Dict[str, object]:
        return {**self.spec.describe(), "unit": self.unit, "n_open": self.n_open,
                "n_total": self.n_total, "density": round(self.density, 6), "resident": True}


#: Every registered backbone, in the order they are offered a folder.
BACKBONES: List[Type[Backbone]] = []


def backbone(cls: Type[Backbone]) -> Type[Backbone]:
    """Register a backbone. Order is registration order, and first claim wins"""
    BACKBONES.append(cls)
    return cls


@backbone
class WeightBackbone(Backbone):
    """A gate per weight, multiplied into the parameters and copied back after

    The bits stay packed on the device: one bit per weight is 170 MB for a
    1.7B circuit where the bool mask is 1.4 GiB, and unpacking a tensor on the
    GPU as it is multiplied in costs less than the multiply. With a dozen
    circuits mounted that is the difference between fitting beside the model
    and not.
    """

    kind = "weights"
    unit = "weights"

    MASK = "sheaf-{task}-mask.pt"
    GATES = "sheaf-{task}-gates.pt"

    @classmethod
    def claims(cls, directory: Path, task: str, record: Dict[str, object]) -> bool:
        return any((directory / template.format(task=task)).exists()
                   for template in (cls.MASK, cls.GATES))

    def __init__(self, host: Host, spec: Spec) -> None:
        super().__init__(host, spec)
        self.packed: Packed = {}
        self._open = self._total = 0

    def load(self) -> None:
        for template in (self.MASK, self.GATES):
            path = self.spec.directory / template.format(task=self.spec.task)
            if path.exists():
                break
        else:  # pragma: no cover - `claims` already established one exists
            raise BackboneError(f"{self.spec.name}: no weight mask to load")
        loaded = torch.load(path, weights_only=True)
        if not isinstance(loaded, dict) or any("bits" not in entry for entry in loaded.values()):
            # the logits file, or an old mask: pack it the way sheaf_prune does
            loaded = pack(loaded)
        unknown = [key for key in loaded if key not in self.host.targets]
        if unknown:
            raise BackboneError(
                f"{self.spec.name}: {len(unknown)} masked tensors are not on this model, "
                f"e.g. {unknown[0]}. A mask is only meaningful on the checkpoint it was "
                f"pruned against."
            )
        device = self.host.device
        self.packed = {key: {"shape": entry["shape"], "bits": entry["bits"].to(device)}
                       for key, entry in loaded.items()}
        opened = total = 0
        for key, entry in self.packed.items():
            mask = unpack_one(key, entry)
            opened += int(mask.sum())
            total += mask.numel()
        self._open, self._total = opened, total

    @property
    def n_open(self) -> int:
        return self._open

    @property
    def n_total(self) -> int:
        return self._total

    @contextmanager
    def applied(self) -> Iterator[None]:
        targets, originals = self.host.targets, self.host.originals
        try:
            with torch.no_grad():
                for key, parameter in targets.items():
                    if key in self.packed:
                        parameter.mul_(unpack_one(key, self.packed[key]).to(parameter.dtype))
            yield
        finally:
            with torch.no_grad():
                for key, parameter in targets.items():
                    if key in self.packed:
                        parameter.copy_(originals[key])


@backbone
class EdgeBackbone(Backbone):
    """A gate per (source, destination) edge, applied as hooks around the forward pass

    The weights are never touched, so there is nothing to undo beyond removing
    the hooks -- which `adapter.edge_gate` already does in its own `finally`.
    The circuit is the `edges_open` list in the artifact; there is no mask file
    to load and a file of 1.4e9 ones would not be one.

    Generation runs through the ordinary cached path. That is a measured claim
    rather than an assumption: `tests/edges.py` checks that greedy decoding
    with a KV cache produces the same tokens as an uncached loop under a
    non-trivial mask, because the hooks capture per-forward activations and
    incremental decoding hands them one position at a time. If that receipt
    ever fails, this backbone has to force `use_cache=False` and pay O(n^2).
    """

    kind = "edges"
    unit = "edges"

    @classmethod
    def claims(cls, directory: Path, task: str, record: Dict[str, object]) -> bool:
        return record.get("edges_open") is not None

    def __init__(self, host: Host, spec: Spec) -> None:
        super().__init__(host, spec)
        self.gates: Dict[Tuple[str, str], torch.Tensor] = {}
        self._open = self._total = 0

    def load(self) -> None:
        artifact = self.spec.directory / f"sheaf-{self.spec.task}.json"
        record = json.loads(artifact.read_text())
        kept = {tuple(edge) for edge in record["edges_open"]}
        every = list(self.host.adapter.edges())
        unknown = [edge for edge in kept if edge not in set(every)]
        if unknown:
            raise BackboneError(
                f"{self.spec.name}: {len(unknown)} edges are not in this model's residual "
                f"stream, e.g. {unknown[0]}. An edge names a layer and a head, so a circuit "
                f"from a different architecture cannot be run here."
            )
        device = self.host.device
        # One scalar per edge, drawn once: `edge_gate` multiplies the source's
        # write by it, so 1.0 leaves the edge alone and 0.0 removes it. Held as
        # tensors rather than floats because the hook multiplies them into a
        # device tensor on every forward.
        one = torch.ones((), device=device)
        zero = torch.zeros((), device=device)
        self.gates = {edge: (one if edge in kept else zero) for edge in every}
        self._open, self._total = len(kept), len(every)

    @property
    def n_open(self) -> int:
        return self._open

    @property
    def n_total(self) -> int:
        return self._total

    @contextmanager
    def applied(self) -> Iterator[None]:
        with self.host.adapter.edge_gate(self.gates):
            yield


def read_record(directory: Path, task: str) -> Dict[str, object]:
    """The folder's own account of itself, or `{}` if it kept none

    Never raises on a malformed file: a directory that happens to hold an
    unparseable JSON is a directory this cannot serve, not a reason to refuse
    to start with eleven others beside it.
    """
    artifact = directory / f"sheaf-{task}.json"
    if not artifact.exists():
        return {}
    try:
        loaded = json.loads(artifact.read_text())
    except (ValueError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _report(directory: Path, task: str, record: Dict[str, object]) -> Dict[str, object]:
    """The numbers a run recorded about itself, for the listing

    `recovered` is carried through wherever the run wrote one: `accuracy` is a
    two-way comparison whose floor is chance, so the raw number read as twice
    the result it was on the translation frame (0.720 against a 0.477 floor is
    0.47 of the range, not 0.72).
    """
    out: Dict[str, object] = {}
    for key, into in (("accuracy", "held_out_ranking"), ("recovered", "recovered"),
                      ("complement_accuracy", "complement_ranking"),
                      ("train_accuracy", "train_ranking"), ("first_token", "first_token"),
                      ("baseline_first_token", "full_model_first_token"),
                      ("settings", "settings")):
        if record.get(key) is not None:
            out[into] = record[key]
    if isinstance(record.get("units"), dict):
        out["units"] = record["units"].get("counts")
    infer = directory / f"sheaf-{task}-inference.json"
    if infer.exists():
        try:
            loaded = json.loads(infer.read_text())
        except (ValueError, OSError):
            return out
        circuit, full = loaded.get("circuit", {}), loaded.get("full_model", {})
        out["fresh_ranking"] = circuit.get("ranking")
        out["fresh_first_token"] = circuit.get("first_token")
        out.setdefault("full_model_first_token", full.get("first_token"))
    return out


def detect(directory: Path, task: str, config: Optional[str] = None) -> Optional[Spec]:
    """The `Spec` for a folder, or None if no backbone claims it

    `config` is the model the server actually holds. A folder pruned against a
    different one is returned *with* a `problem` rather than dropped, so it
    appears in the listing saying why it cannot be run -- the alternative is a
    circuit that is simply absent and a caller with no way to find out.
    """
    record = read_record(directory, task)
    for cls in BACKBONES:
        if not cls.claims(directory, task, record):
            continue
        against = record.get("config")
        problem = None
        if config and against and against != config:
            problem = f"pruned against '{against}', this server holds '{config}'"
        density = record.get("edge_density") if cls.kind == "edges" else record.get("density")
        return Spec(name=directory.name, directory=directory, task=task, kind=cls.kind,
                    unit=cls.unit, config=against if isinstance(against, str) else None,
                    density=density if isinstance(density, (int, float)) else None,
                    report=_report(directory, task, record), problem=problem)
    return None


def backbone_for(spec: Spec) -> Type[Backbone]:
    """The registered class whose `kind` the spec was detected as"""
    for cls in BACKBONES:
        if cls.kind == spec.kind:
            return cls
    raise BackboneError(f"no backbone registered for kind '{spec.kind}'")
