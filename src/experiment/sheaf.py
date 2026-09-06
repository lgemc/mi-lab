"""A pruning run as a config file, because a run defined in argv is a run nobody can find again.

Every sheaf in `results/` was launched from a shell line. The artifact records
that line, which is enough to rerun it and not enough to *read* it: the nine
`sheaf-translation-*.yaml` pipelines under `pipelines/run/` are wrappers around
argv strings, so the parameters that decide a result -- which faithfulness
term, which density target, how the gates start -- were never anywhere a
diff could show them. Two runs differing in one flag look like two unrelated
commands.

So the parameters live here, as a typed record composed by Hydra the way
`ExperimentSpec` and `Pipeline` already are, and the shell keeps the grammar it
had: `run=qwen-ioi-kl faith=nll` overrides one key of a named experiment
without editing it. The argparse entrypoint stays and builds the same record,
so both doors open into one code path and one schema -- a flag that does not
exist here cannot be passed there.

Unknown keys are errors (invariant 3). A misspelled `warmpup:` in a YAML is a
line that silently did nothing, and on a two-hour run that is discovered at the
end or never.

`results` is a field rather than the `MI_LAB_RESULTS` environment variable it
used to be. Where a run writes is part of what the run *is*; keeping it in the
environment is how two experiments end up sharing a directory and the second
one reads the first one's cached scores as its own.

A common pipe could be: compose | SheafSpec.from_mapping | overrides | prune
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SHEAF_DIR = Path(__file__).resolve().parents[2] / "sheaves"
SCHEMA_NAME = "sheaf_schema"


class SheafError(ValueError):
    """Raised when a sheaf experiment is not runnable as written"""


@dataclass
class SheafSpec:
    """Every parameter of one pruning run, and nothing that is not one

    The defaults are the *function's* defaults, not a recommended experiment:
    a YAML that states only what it changes is readable, and one that restates
    the whole surface hides which two keys are the point of it.
    """

    # what is pruned, and on what
    config: str = "qwen3-1.7b"
    task: str = "translation"
    layers: str = "all"
    results: Optional[str] = None

    # the data
    size: int = 1024
    seed: int = 0
    holdout: float = 0.25

    # the schedule
    steps: int = 2000
    batch: int = 8
    rate: float = 0.1
    probe_every: int = 25
    # held-out rows the in-run probe scores. `batch` was the accidental
    # denominator and 8 rows is +-0.18 -- a collapse detector, not a curve.
    probe_size: int = 128

    # what faithfulness means. "nll" is the paper's and is only equivalent to
    # the task where the full model's argmax *is* the answer -- 98.4% of GPT-2
    # small's IOI prompts and 52.4% of Qwen3-1.7B's, which is why a protocol
    # copied between them is not the same protocol.
    faith: str = "kl"
    completeness: float = 0.3

    # the gates
    init: float = 5.0
    init_low: float = 2.0
    temperature: float = 1.0
    anneal: bool = False
    protect: float = 0.0
    granular: Optional[str] = None
    attribute: int = 0

    # the sparsity price: a target with a learned price, or a hand-ramped one
    target: Optional[float] = None
    warmup: Optional[int] = None
    dual_rate: Optional[float] = None
    dual_restart: bool = False
    sparsity: float = 1.0
    max_times: float = 1000.0

    # edges
    edges_only: bool = False
    edge_sparsity: float = 0.0

    # guards and bookkeeping
    reserve: float = 4.0
    needs: float = 0.05
    force: bool = False
    save_gates: bool = False
    tracking: str = "mlflow"

    def __post_init__(self) -> None:
        if self.faith not in ("pair", "kl", "nll", "gold"):
            raise SheafError(
                f"faith '{self.faith}' is not one of pair, kl, nll, gold"
            )
        if self.target is not None and not 0.0 < self.target <= 1.0:
            raise SheafError(f"target is a fraction in (0, 1], got {self.target}")
        if not 0.0 < self.holdout < 1.0:
            raise SheafError(f"holdout is a fraction in (0, 1), got {self.holdout}")
        if self.steps < 1:
            raise SheafError(f"steps must be at least 1, got {self.steps}")
        if self.edges_only and self.edge_sparsity <= 0.0 and self.target is None:
            raise SheafError(
                "edges_only prunes edges alone, so something has to price them: set "
                "edge_sparsity, or target for a learned price on the edge density"
            )
        if self.edges_only and (self.protect > 0 or self.granular or self.attribute):
            raise SheafError(
                "protect, granular and attribute all shape the weight gates, and "
                "edges_only has none"
            )

    @property
    def faith_kind(self) -> str:
        """`prune`'s name for `faith`, so a spec is what the runner already reads

        The runner takes an object and reads attributes off it; matching every
        name means the argparse door and the YAML door open into one code path
        instead of two that drift.
        """
        return self.faith

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "SheafSpec":
        """A composed config as a spec, with an unknown key as an error rather than a shrug"""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise SheafError(
                f"unknown sheaf keys {unknown}; a key that is not a parameter is a line "
                f"that would have done nothing. Known keys are {sorted(known)}"
            )
        return cls(**cfg)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def command(self) -> str:
        """The equivalent shell line, recorded in the artifact so a YAML run is rerunnable"""
        parts = [f"run={self.task}"]
        for field in fields(self):
            value = getattr(self, field.name)
            if value != field.default:
                parts.append(f"{field.name}={value}")
        return "uv run python -m scripts.sheaf " + " ".join(parts)


def register_schema() -> None:
    """Put SheafSpec in Hydra's ConfigStore as the base of the `run` group

    The same trick `spec.register_schema` uses, and it buys the same two
    things. Composition is type-checked, so `steps: two thousand` is an error
    at compose time rather than a TypeError after the model loads. And every
    field *exists* with its default, so a run file can state only what it
    changes and the shell can still override any key -- without the schema,
    `run.edges_only=true` against a file that does not mention edges is
    "Could not override", which pushes every run file back to restating the
    whole surface.
    """
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(group="run", name=SCHEMA_NAME, node=SheafSpec)


def compose(overrides: Sequence[str], directory: Optional[Path] = None) -> Dict[str, Any]:
    """The sheaf config, composed with Hydra's grammar so a run can be swapped or edited inline

    Hydra keeps its state in a process-global singleton, cleared before and
    after for the same reason `spec.compose_spec` and `pipeline.compose` do.
    """
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    directory = directory or SHEAF_DIR
    if not directory.is_dir():
        raise SheafError(f"no sheaf config directory at {directory}")
    register_schema()
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            cfg = hydra_compose(config_name="config", overrides=list(overrides))
        loaded = OmegaConf.to_container(cfg, resolve=True)
    except (HydraException, OmegaConfBaseException) as error:
        raise SheafError(str(error)) from error
    finally:
        GlobalHydra.instance().clear()
    # `run` is the group; its keys are the spec, and the composition root holds
    # nothing else worth carrying into it.
    run = loaded.get("run") or {}
    if not isinstance(run, dict):
        raise SheafError("the composed config has no `run` mapping")
    return run
