from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

"""
Config is the single place a model fact is allowed to live. Nothing else in
the framework may name a layer index, a residual stream width, or a
checkpoint: a probe asks its ModelConfig how wide d_model is, and a hook asks
for the layer sitting at a fraction of the model's depth. That is what makes
an experiment developed against GPT-2 small rerun against a 27B by loading a
different config instead of by editing code.

Layers are addressed by *depth fraction*, never by index. Layer 8 of a
12-layer model and layer 42 of a 64-layer model are the same place -- roughly
two thirds through, where production activation monitoring hooks -- and only
the fraction 0.65 survives a model swap.

Sizes are optional here: a backend that has loaded the checkpoint already
knows n_layers and d_model, and resolve() stamps them back into the config.
Write them down explicitly only when you want the framework to assert them.

`device` is the one field here that is not a fact about the model. It is where
this machine can put it, so it defaults to "auto" and is resolved by the
backend that has to place the weights -- see resolve_device in
model/adapter.py, which is also where torch lives. Nothing in this module
imports torch, so a config stays readable on a machine with no CUDA, no GPU
and no model library at all.

A common pipe could be: load_config | resolve | layer
"""

class ConfigError(ValueError):
    """Raised when a config is unusable: unknown preset, bad file, or a size that disagrees with the checkpoint"""

class Position(str, Enum):
    """Which token position(s) a capture keeps from each prompt

    LAST is the default because the final token carries the model's decision
    state. MEAN averages over real (non-padding) tokens. ALL keeps the whole
    sequence, which aggregation experiments need and which costs seq_len times
    more memory.
    """
    LAST = "last"
    MEAN = "mean"
    ALL = "all"

@dataclass(frozen=True)
class SAEConfig:
    """Where a sparse dictionary for this model comes from, and how it behaves

    l0_varies is not decoration: a ReLU SAE's sparsity moves with the input and
    can be swept, while a TopK SAE holds L0 constant by construction, so any
    experiment that sweeps sparsity is silently a no-op on TopK dictionaries.
    """
    source: str
    release: str
    activation_fn: str = "relu"
    l0_varies: bool = True

@dataclass(frozen=True)
class TranscoderConfig:
    """Where a transcoder for this model comes from, and the two numbers that price believing it

    A transcoder is not an SAE and has its own dataclass rather than a flag on
    one. An SAE reconstructs an activation *from itself*; a transcoder maps an
    MLP's input to its output, which is what lets it *replace* the MLP --
    and replacing the MLP is the whole construction an attribution graph rests
    on (Ameisen et al. 2025). A residual-stream SAE cannot stand in here no
    matter how good it is.

    `kind` is 'cross-layer' or 'per-layer' and changes what the weights cost
    as much as what they mean: a cross-layer feature decodes into every later
    layer, so its decoders are a pyramid, and 20,480 features on a 28-layer
    1.7B is 43.6 GiB -- larger than the 8B checkpoint this repo's headline
    result runs on. `resident_gib` is recorded for the reason `cost.py` prices
    a band before the model loads: a number you find out by running out of
    disk is a number you should have read off a file.

    `variance_unexplained` is the size of what the method cannot see. Whatever
    the transcoder fails to reconstruct becomes an "error node" in the graph,
    and error nodes have no input edges -- they are where the explanation
    stops. Reading a graph without that fraction in view reads it as more
    complete than it is.
    """
    source: str
    release: str
    kind: str = "per-layer"
    activation_fn: str = "jumprelu"
    features_per_layer: Optional[int] = None
    expansion: Optional[int] = None
    l0: Optional[float] = None
    variance_unexplained: Optional[float] = None
    resident_gib: Optional[float] = None
    license: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ("cross-layer", "per-layer"):
            raise ConfigError(
                f"transcoder kind is 'cross-layer' or 'per-layer', got '{self.kind}'; the two differ in "
                "what they cost and in whether a feature's decoders reach later layers, so this is not a label"
            )
        if self.variance_unexplained is not None and not 0.0 <= self.variance_unexplained <= 1.0:
            raise ConfigError(
                f"variance_unexplained is a fraction in [0, 1], got {self.variance_unexplained}"
            )

@dataclass(frozen=True)
class ModelConfig:
    """Everything the framework is allowed to know about a model

    backend is a registry key resolved by core.adapter, not an import path, so
    swapping TransformerLens for nnsight-over-vLLM is a one-line config edit.
    """
    id: str
    backend: str
    hf_name: str
    n_layers: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    probe_layer_frac: float = 0.65
    batch_size: int = 8
    device: str = "auto"
    dtype: str = "float32"
    max_new_tokens: int = 32
    # An instruction-tuned checkpoint answers a *conversation*, not a raw
    # continuation: its post-training only ever saw text wrapped in the chat
    # template, so feeding it a bare prompt asks it something it was never
    # trained on and reads as a much worse model than it is. Set for the
    # served instruct checkpoints; the pruning work uses base models and
    # leaves it false, which keeps every existing prompt byte-identical.
    chat: bool = False
    sae: Optional[SAEConfig] = None
    transcoder: Optional[TranscoderConfig] = None

    def __post_init__(self):
        if not 0.0 <= self.probe_layer_frac <= 1.0:
            raise ConfigError(
                f"probe_layer_frac must be a fraction in [0, 1], got {self.probe_layer_frac}. "
                "Absolute layer indices are not a thing this framework accepts."
            )
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {self.batch_size}")

    @property
    def is_resolved(self) -> bool:
        """Whether the checkpoint's shape metadata has been filled in yet

        n_heads is not part of this. Everything that addresses the residual
        stream needs n_layers and d_model; only head-level circuit work needs
        the head count, and a backend that cannot expose heads should still
        count as resolved for the probing half of the framework.
        """
        return self.n_layers is not None and self.d_model is not None

    @property
    def d_head(self) -> int:
        """Width of one attention head, which is the shape a head-level patch has to be

        Derived rather than stored: a head is a slice of the residual stream
        width, and storing it separately is a second chance to disagree with
        the checkpoint.
        """
        if self.d_model is None or self.n_heads is None:
            raise ConfigError(
                f"config '{self.id}' has no d_model/n_heads yet, so a head width cannot be derived; "
                "load an adapter first"
            )
        if self.d_model % self.n_heads:
            raise ConfigError(
                f"config '{self.id}' has d_model not divisible by n_heads, so heads are not "
                "equal-width slices of the residual stream; head-level work needs a different split"
            )
        return self.d_model // self.n_heads

    def layer(self, frac: Optional[float] = None) -> int:
        """Resolve a depth fraction to an absolute layer index

        The single most important line in the framework: it is what lets one
        experiment address "two thirds of the way through" on any model. Defaults
        to the config's own probe_layer_frac.
        """
        if self.n_layers is None:
            raise ConfigError(
                f"config '{self.id}' has no n_layers yet, so a fraction cannot be resolved; "
                "load an adapter first, or write n_layers into the config"
            )
        if frac is None:
            frac = self.probe_layer_frac
        return min(round(frac * self.n_layers), self.n_layers - 1)

    def layers(self, fracs: List[float]) -> List[int]:
        """Resolve several depth fractions at once, dropping duplicates and keeping order"""
        resolved = []
        for frac in fracs:
            index = self.layer(frac)
            if index not in resolved:
                resolved.append(index)
        return resolved

    def sweep(self, count: int) -> List[int]:
        """Evenly spaced layer indices across the whole depth, for a first look at a new model"""
        if count < 1:
            raise ConfigError(f"a sweep needs at least one layer, got {count}")
        if count == 1:
            return [self.layer(0.5)]
        return self.layers([step / (count - 1) for step in range(count)])

    def with_sizes(self, n_layers: int, d_model: int, n_heads: Optional[int] = None) -> "ModelConfig":
        """Return a copy carrying the shape metadata a backend read off the checkpoint

        If the config already stated a size, disagreeing with the checkpoint is an
        error rather than a silent overwrite: it means the config describes a
        different model than the one that just loaded.
        """
        for name, stated, actual in (
            ("n_layers", self.n_layers, n_layers),
            ("d_model", self.d_model, d_model),
            ("n_heads", self.n_heads, n_heads),
        ):
            if actual is None or stated is None:
                continue
            if stated != actual:
                raise ConfigError(
                    f"config '{self.id}' declares {name}={stated} but checkpoint "
                    f"'{self.hf_name}' has {name}={actual}"
                )
        return replace(self, n_layers=n_layers, d_model=d_model, n_heads=n_heads or self.n_heads)

    def as_dict(self) -> Dict[str, Any]:
        """Plain-data view of the config, for printing, hashing or writing back out"""
        return asdict(self)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

def presets() -> List[str]:
    """Names of the configs shipped in configs/

    The directory is found relative to this file rather than to the working
    directory, so a preset name means the same thing from a notebook, a test
    run and a shell in some other folder.
    """
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))

def _read_mapping(path: Path) -> Dict[str, Any]:
    """Parse a config file, choosing the parser by suffix"""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as error:
            raise ConfigError(f"reading {path} needs PyYAML installed") from error
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        import json
        return json.loads(text)
    raise ConfigError(f"unsupported config format '{path.suffix}': use .yaml, .yml or .json")

def from_mapping(data: Dict[str, Any], default_id: Optional[str] = None) -> ModelConfig:
    """Build a ModelConfig from plain data, rejecting unknown keys instead of ignoring them

    A typo'd key that gets silently dropped is how an experiment quietly runs
    with the wrong batch size for a week.
    """
    data = dict(data)
    data.setdefault("id", default_id)
    nested = {name: data.pop(name, None) for name in ("sae", "transcoder")}
    known = set(ModelConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config keys {sorted(unknown)}; known keys are {sorted(known)}")
    for name, block in nested.items():
        if block is None:
            continue
        schema = {"sae": SAEConfig, "transcoder": TranscoderConfig}[name]
        unknown_keys = set(block) - set(schema.__dataclass_fields__)
        if unknown_keys:
            raise ConfigError(f"unknown {name} keys {sorted(unknown_keys)}")
        data[name] = schema(**block)
    return ModelConfig(**data)

def load_config(reference: str) -> ModelConfig:
    """Load a config by shipped preset name, or by path to a YAML/JSON file"""
    for path in (CONFIG_DIR / f"{reference}.yaml", Path(reference)):
        if path.exists():
            return from_mapping(_read_mapping(path), default_id=path.stem)
    raise ConfigError(
        f"'{reference}' is neither a config in {CONFIG_DIR} nor an existing file; "
        f"shipped configs are {presets()}"
    )
