import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import ModelConfig, load_config
from ..data.dataset import LabeledPrompts, load_jsonl, synthetic
from ..data.ioi import CORRUPTIONS, FRAMES
from ..data.prompts import load_prompts

"""
An ExperimentSpec is the whole experiment as data: which model, which data,
which method, at which depths. Running it is deterministic, and two people
holding the same spec get the same numbers.

There are two ways in, and they exist for different jobs.

compose_spec is the one to use: Hydra composes specs/config.yaml out of the
group directories beside it, so `model=pythia-70m method=difference_of_means`
swaps whole sections, `+preset=sentiment-sweep` pulls in a named bundle, and
`method.lr=0.1` overrides one key. Because ExperimentSpec is registered as the
schema, every value is type-checked and a mistyped key is an error rather than
a line that quietly did nothing.

load_spec reads one self-contained file with no groups and no composition. It
is what reads back the spec.yaml a run wrote, which must keep working long
after the group files beside it have been edited -- a run has to stay
reproducible from its own directory.

spec_hash covers everything that determines the result and nothing that does
not -- output paths are excluded on purpose, so writing the same experiment to
a different directory does not make it look like a different experiment.

A common pipe could be: compose_spec | run_experiment | Run.save
"""

SPEC_DIR = Path(__file__).resolve().parents[2] / "specs"
SCHEMA_NAME = "experiment_schema"

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

SOURCES = ("synthetic", "prompts", "jsonl")

@dataclass
class DataSpec:
    """Where the labelled prompts come from, and how they are split

    'prompts' is the plain-text format of core.prompts and the one to reach
    for: it is the only source that carries label names and groups, and a
    spec pointed at it splits without cutting a contrast pair in half.
    """
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
        if self.source not in SOURCES:
            raise SpecError(f"unknown data.source '{self.source}'; known sources are {sorted(SOURCES)}")
        if not self.path:
            raise SpecError(f"data.source is '{self.source}' but data.path is not set")
        if self.source == "prompts":
            return load_prompts(self.path, limit=self.limit)
        return load_jsonl(self.path, text_field=self.text_field, label_field=self.label_field, limit=self.limit)

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
class IOISpec:
    """The knobs of the IOI circuit study, which the probing kinds ignore

    A spec carries every section whatever its kind, so a probe_sweep hashes an
    IOISpec it never reads and an ioi_circuit hashes a MethodSpec it never
    reads. That is the cost of one spec type; the alternative is a spec whose
    shape depends on its kind, which no longer type-checks as one schema. What
    is not negotiable is the other direction: a knob that changes an IOI number
    lives here, inside the hash, rather than being passed at the command line.
    """
    size: int = 16
    frame: int = 0
    corruption: str = "abc"
    threshold: float = 0.8
    max_heads: int = 12
    tolerance: float = 0.05
    residual_patch: bool = True

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
    ioi: IOISpec = field(default_factory=IOISpec)
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
        if self.data.source not in SOURCES:
            raise SpecError(f"unknown data.source '{self.data.source}'; known sources are {sorted(SOURCES)}")
        if self.data.source != "synthetic" and not self.data.path:
            raise SpecError(f"data.source is '{self.data.source}' but data.path is not set")
        if self.ioi.corruption not in CORRUPTIONS:
            raise SpecError(f"unknown ioi.corruption '{self.ioi.corruption}'; known ones are {sorted(CORRUPTIONS)}")
        if not 0 <= self.ioi.frame < len(FRAMES):
            raise SpecError(f"ioi.frame must be one of the {len(FRAMES)} shipped frames, got {self.ioi.frame}")
        if self.ioi.size < 1:
            raise SpecError(f"ioi.size must be at least one prompt pair, got {self.ioi.size}")
        if self.ioi.max_heads < 1:
            raise SpecError(f"ioi.max_heads must be at least one head, got {self.ioi.max_heads}")
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

    Path(path).write_text(OmegaConf.to_yaml(OmegaConf.structured(spec)))

def _register_schema() -> None:
    """Put ExperimentSpec in Hydra's ConfigStore so composition is type-checked

    Registering twice is harmless, which matters because every compose_spec
    call has to be able to assume it has happened.
    """
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name=SCHEMA_NAME, node=ExperimentSpec)

def groups() -> Dict[str, List[str]]:
    """The config groups beside specs/config.yaml, and the options in each"""
    if not SPEC_DIR.is_dir():
        return {}
    return {
        directory.name: sorted(path.stem for path in directory.glob("*.yaml"))
        for directory in sorted(SPEC_DIR.iterdir())
        if directory.is_dir()
    }

def _reject_unknown_keys(actual: Dict[str, Any], reference: Dict[str, Any], path: str = "") -> None:
    """Fail on any key the schema does not define

    Hydra's `+` prefix means "append a key that is not there", and it honours
    that even against a structured schema: `+nonsense=1` composes cleanly, then
    vanishes when the config becomes an ExperimentSpec. The override looks like
    it did something, changes nothing, and does not even reach the spec hash.
    Struct mode does not catch it, so this does.
    """
    for key, value in actual.items():
        location = f"{path}{key}"
        if key not in reference:
            raise SpecError(f"'{location}' is not a key of ExperimentSpec; known keys here are {sorted(reference)}")
        if isinstance(value, dict) and isinstance(reference[key], dict):
            _reject_unknown_keys(value, reference[key], f"{location}.")

def compose_spec(
    preset: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    config_name: str = "config",
) -> ExperimentSpec:
    """Compose a spec from the group defaults, an optional preset, and overrides

    Overrides use Hydra's grammar, so `model=pythia-70m` swaps a whole group
    while `model.batch_size=4` overrides one key inside it.

    Hydra keeps its state in a process-global singleton, so it is cleared
    before and after composing: without that, a second call in the same process
    inherits the first one's config directory.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    if not SPEC_DIR.is_dir():
        raise SpecError(f"no spec directory at {SPEC_DIR}")

    _register_schema()
    selections = list(overrides or [])
    if preset:
        available = groups().get("preset", [])
        if preset not in available:
            raise SpecError(f"unknown preset '{preset}'; available presets are {available}")
        selections.insert(0, f"+preset={preset}")

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(SPEC_DIR), version_base="1.3"):
            merged = compose(config_name=config_name, overrides=selections)
            _reject_unknown_keys(
                OmegaConf.to_container(merged, resolve=True),
                OmegaConf.to_container(OmegaConf.structured(ExperimentSpec)),
            )
            spec = OmegaConf.to_object(merged)
    except (HydraException, OmegaConfBaseException) as error:
        raise SpecError(str(error)) from error
    finally:
        GlobalHydra.instance().clear()
    return spec.validate()
