"""A phase as a sequence of steps named in a file, rather than in a paragraph.

Phase 1b is nine invocations in an order that matters: the FLOPs table before
anything that divides by it, the sweep before the ranking, the ranking before
the significance gate that decides whether the ranking may be used. That
order lived in commit messages and in whoever ran it last, which is the same
place a protocol goes to be misremembered -- and every one of those steps
takes a config, an environment and a budget that have to agree across all
nine.

So the sequence is a Hydra config under `pipelines/`, composed the way
`specs/config.yaml` is composed, and this runs it. Beside `specs/` rather
than inside it: `specs/` is the ExperimentSpec composition tree and every
directory in it is a group Hydra can swap into one, which a pipeline is not.
The value is not saving keystrokes; it is that the environment reaches every
step identically. The results root and the corpus size are set once at the
top and exported to each subprocess, so a run cannot half-happen at 497
sentences and half at 200.

Steps run as subprocesses rather than imports. Each phase script reads its
constants at import time from the environment, so running two steps with
different roots in one interpreter would give the second one the first one's
paths; a subprocess boundary is what makes the environment per-step at all.

Resumable like everything else here: each completed step is flushed to a
state file beside the results and skipped on re-entry. `repeat` re-invokes a
step while it reports stopping on its budget -- the marker `observe.Budget`
prints -- so a sweep that needs four invocations gets them without four
commands.

A common pipe could be: compose | select | run_step | record | report
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..telemetry.observe import BUDGET_MARKER, banner, duration, log, set_log_file
from ..telemetry.results import ENV_ROOT, load_state, save_state

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "pipelines"
STATE_FILE = "pipeline-state.json"
LOG_FILE = "pipeline.log"

# How much of a step's output is worth repeating in the pipeline log: the
# last lines say what it did and, on failure, the traceback's end says why.
STDOUT_TAIL = 6
STDERR_TAIL = 15

class PipelineError(ValueError):
    """A step that failed, or a pipeline that cannot be composed"""

@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: List[str] = field(default_factory=list)
    repeat: bool = False

    @property
    def marker(self) -> str:
        """The key a completed step is recorded under"""
        return f"{self.module}:{self.name}"

    def command(self, config: str) -> List[str]:
        return [sys.executable, "-m", self.module, config, *self.args]

@dataclass(frozen=True)
class Pipeline:
    name: str
    config: str
    env: Dict[str, str]
    steps: List[Step]
    resume: bool = True
    dry_run: bool = False
    only: List[str] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return Path(self.env.get(ENV_ROOT, "results"))

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def selected(self) -> List[Step]:
        if not self.only:
            return list(self.steps)
        unknown = [name for name in self.only if name not in {step.name for step in self.steps}]
        if unknown:
            raise PipelineError(f"no step named {unknown}; steps are {[step.name for step in self.steps]}")
        return [step for step in self.steps if step.name in self.only]

    @classmethod
    def from_mapping(cls, cfg: Dict[str, Any]) -> "Pipeline":
        try:
            run = cfg["run"]
            steps = [Step(name=s["name"], module=s["module"],
                          args=[str(a) for a in (s.get("args") or [])], repeat=bool(s.get("repeat", False)))
                     for s in run["steps"]]
            return cls(name=run["name"], config=run["config"],
                       env={key: str(value) for key, value in (run.get("env") or {}).items()},
                       steps=steps, resume=bool(cfg.get("resume", True)), dry_run=bool(cfg.get("dry_run", False)),
                       only=list(cfg.get("only") or []))
        except KeyError as error:
            raise PipelineError(f"pipeline config is missing {error}") from None

def compose(overrides: Sequence[str], directory: Optional[Path] = None) -> Dict[str, Any]:
    """The pipeline config, composed with Hydra's grammar so a run can be swapped or edited inline

    Hydra keeps its state in a process-global singleton, cleared before and
    after for the same reason `spec.compose_spec` does.
    """
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.errors import HydraException
    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    directory = directory or PIPELINE_DIR
    if not directory.is_dir():
        raise PipelineError(f"no pipeline directory at {directory}")
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            cfg = hydra_compose(config_name="config", overrides=list(overrides))
        return OmegaConf.to_container(cfg, resolve=True)
    except (HydraException, OmegaConfBaseException) as error:
        raise PipelineError(str(error)) from error
    finally:
        GlobalHydra.instance().clear()

def run_step(step: Step, config: str, env: Dict[str, str]) -> Dict[str, Any]:
    """One step as a subprocess, repeated while it says it stopped on budget"""
    command = step.command(config)
    environment = {**os.environ, **env}
    invocations, started, budget_exits = 0, time.time(), 0
    while True:
        invocations += 1
        log(f"$ {' '.join(command)}", indent=1)
        finished = subprocess.run(command, env=environment, capture_output=True, text=True)
        tail = (finished.stdout or "").strip().splitlines()
        for line in tail[-STDOUT_TAIL:]:
            log(line, indent=2)
        if finished.returncode != 0:
            log(f"!! {step.name} failed with exit {finished.returncode}", indent=1)
            for line in (finished.stderr or "").strip().splitlines()[-STDERR_TAIL:]:
                log(line, indent=2)
            raise PipelineError(
                f"step '{step.name}' failed. The pipeline stops here rather than running the steps "
                "after it, because every one of them reads what this was supposed to write."
            )
        hit_budget = any(BUDGET_MARKER in line for line in tail)
        if hit_budget:
            budget_exits += 1
        if not (step.repeat and hit_budget):
            break
        log(f"{step.name} stopped on its budget; re-invoking ({invocations} so far)", indent=1)
    return {"invocations": invocations, "budget_exits": budget_exits,
            "seconds": round(time.time() - started, 1)}

def run(pipeline: Pipeline) -> Dict[str, Any]:
    """Every selected step in order, recording each as it completes; returns the state"""
    steps = pipeline.selected
    pipeline.root.mkdir(parents=True, exist_ok=True)
    set_log_file(pipeline.root / LOG_FILE)
    state = load_state(pipeline.state_path, {"done": {}})
    banner(f"pipeline: {pipeline.name}", {
        "config": pipeline.config,
        "steps": f"{len(steps)} selected of {len(pipeline.steps)}, {len(state['done'])} already done",
        "environment": ", ".join(f"{k}={v}" for k, v in pipeline.env.items()) or "(inherited)",
        "resume": pipeline.resume,
        "state": str(pipeline.state_path),
    })
    for index, step in enumerate(steps, start=1):
        if pipeline.resume and step.marker in state["done"]:
            log(f"[{index}/{len(steps)}] {step.name} already done "
                f"({duration(state['done'][step.marker]['seconds'])}) -- skipping")
            continue
        log(f"[{index}/{len(steps)}] {step.name}")
        if pipeline.dry_run:
            log(f"$ {' '.join(step.command(pipeline.config))}", indent=1)
            continue
        record = run_step(step, pipeline.config, pipeline.env)
        state["done"][step.marker] = record
        save_state(pipeline.state_path, state)
        log(f"{step.name} done in {duration(record['seconds'])} ({record['invocations']} invocation(s))", indent=1)
    log("dry run: nothing executed" if pipeline.dry_run else f"pipeline '{pipeline.name}' complete")
    return state
