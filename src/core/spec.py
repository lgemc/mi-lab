import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from .config import ModelConfig, load_config
from .dataset import LabeledPrompts, load_jsonl, synthetic

"""
An ExperimentSpec is the whole experiment as data: which model, which data,
which method, at which depths. Running it is deterministic, and two people
holding the same spec get the same numbers.

The spec is composed with OmegaConf rather than parsed by hand, which buys
three things that matter once experiments start multiplying: a file may
override only the keys it cares about, the shell may override any key by
dotted path without a flag existing for it, and a mistyped key is a merge
error instead of a silently ignored line.

spec_hash covers everything that determines the result and nothing that does
not -- output paths are excluded on purpose, so writing the same experiment to
a different directory does not make it look like a different experiment.

A common pipe could be: load_spec | run_experiment | Run.save
"""

class SpecError(ValueError):
    """Raised when a spec is malformed, names something unknown, or fails to merge"""

@dataclass
class ModelSpec:
    """Which model, plus any config field this experiment wants to override

    The overrides are Optional and default to None so that "not stated" is
    distinguishable from "stated as the same value the config already had",
    and so a spec never silently reimposes a default on a config that chose
    otherwise.
    """
    config: str = "gpt2-small"
    device: Optional[str] = None
    dtype: Optional[str] = None
    batch_size: Optional[int] = None
    probe_layer_frac: Optional[float] = None

    def resolve(self) -> ModelConfig:
        """Load the named config and apply this spec's overrides to it"""
        overrides = {
            key: value
            for key, value in (
                ("device", self.device), ("dtype", self.dtype),
                ("batch_size", self.batch_size), ("probe_layer_frac", self.probe_layer_frac),
            )
            if value is not None
        }
        return replace(load_config(self.config), **overrides)

@dataclass
class DataSpec:
    """Where the labelled prompts come from, and how they are split"""
    source: str = "synthetic"
    size: int = 200
    path: Optional[str] = None
    text_field: str = "text"
    label_field: str = "label"
    limit: Optional[int] = None
    test_frac: float = 0.3

    def load(self, seed: int = 0) -> LabeledPrompts:
        """Build the dataset this spec describes"""
        if self.source == "synthetic":
            return synthetic(n=self.size, seed=seed)
        if self.source == "jsonl":
            if not self.path:
                raise SpecError("data.source is 'jsonl' but data.path is not set")
            return load_jsonl(self.path, text_field=self.text_field, label_field=self.label_field, limit=self.limit)
        raise SpecError(f"unknown data.source '{self.source}'; known sources are ['jsonl', 'synthetic']")

@dataclass
class MethodSpec:
    """Which probe is fit, at which depths, with which hyperparameters"""
    kind: str = "logistic"
    fracs: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.65, 0.85, 1.0])
    position: str = "last"
    epochs: int = 400
    lr: float = 0.05
    l2: float = 0.01

@dataclass
class OutputSpec:
    """Where the run's artifacts land

    Excluded from spec_hash: none of this changes a number.
    """
    root: str = "runs"
    save_probe: bool = True

@dataclass
class ExperimentSpec:
    """One experiment, fully described

    kind names the runner that executes it, the way backend names the adapter
    that loads a model: a new experiment type is a registration, not an edit
    to this class.
    """
    experiment: str = "unnamed"
    kind: str = "probe_sweep"
    seed: int = 0
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    method: MethodSpec = field(default_factory=MethodSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def validate(self) -> "ExperimentSpec":
        """Check everything that can be checked without loading a model"""
        if self.method.kind not in ("logistic", "difference_of_means"):
            raise SpecError(
                f"unknown method.kind '{self.method.kind}'; "
                "known kinds are ['difference_of_means', 'logistic']"
            )
        if not self.method.fracs:
            raise SpecError("method.fracs is empty, so there is nothing to probe")
        outside = [frac for frac in self.method.fracs if not 0.0 <= frac <= 1.0]
        if outside:
            raise SpecError(f"method.fracs {outside} are not depth fractions in [0, 1]")
        if not 0.0 < self.data.test_frac < 1.0:
            raise SpecError(f"data.test_frac must be strictly between 0 and 1, got {self.data.test_frac}")
        if self.data.source == "jsonl" and not self.data.path:
            raise SpecError("data.source is 'jsonl' but data.path is not set")
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Plain-data view of the whole spec"""
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)

    @property
    def spec_hash(self) -> str:
        """A reproducibility key: sha256 over everything that determines the result

        Output paths are excluded, so the same experiment written to two
        directories hashes the same, and a run can be recognized as a repeat
        of one already done.
        """
        payload = self.as_dict()
        payload.pop("output", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def load_spec(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentSpec:
    """Compose a spec from the defaults, a file, and dotted command-line overrides

    Later sources win, which is what makes `--set method.lr=0.1` mean the same
    thing whether or not the spec file mentioned lr. A key that is not in the
    schema is a merge error rather than a line that quietly did nothing.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    merged = OmegaConf.structured(ExperimentSpec)
    try:
        if path is not None:
            merged = OmegaConf.merge(merged, OmegaConf.load(path))
        if overrides:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
        spec = OmegaConf.to_object(merged)
    except OmegaConfBaseException as error:
        source = f"{path}: " if path else ""
        raise SpecError(f"{source}{error}") from error
    return spec.validate()

def save_spec(spec: ExperimentSpec, path: str) -> None:
    """Write the fully resolved spec, so a run records what it actually ran"""
    from omegaconf import OmegaConf

    with open(path, "w") as handle:
        handle.write(OmegaConf.to_yaml(OmegaConf.structured(spec)))
