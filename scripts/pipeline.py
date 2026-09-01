"""Run a phase as a sequence of steps named in a file, rather than in a paragraph.

Phase 1b is nine invocations in an order that matters: the FLOPs table before
anything that divides by it, the sweep before the ranking, the ranking before
the significance gate that decides whether the ranking may be used. That order
lived in commit messages and in whoever ran it last, which is the same place a
protocol goes to be misremembered -- and every one of those steps takes a
config, an environment and a budget that have to agree across all nine.

So the sequence is a Hydra config under specs/pipeline/, composed the way
specs/config.yaml is composed, and this runs it. The value is not saving
keystrokes; it is that the environment reaches every step identically. The
results root and the corpus size are set once at the top and exported to each
subprocess, so a run cannot half-happen at 497 sentences and half at 200 -- the
failure that the baseline guard catches after the fact and this prevents.

Steps run as subprocesses rather than imports. Each phase1b script reads its
constants at import time from the environment, so running two steps with
different roots in one interpreter would give the second one the first one's
paths; a subprocess boundary is what makes the environment per-step at all.

Resumable like everything else here: each completed step is flushed to a state
file beside the results and skipped on re-entry. `repeat` re-invokes a step
while it reports stopping on its budget, which is the marker Budget prints, so
a sweep that needs four invocations gets them without four commands.

A common pipe could be: compose | plan | run_step | record | report

Run: uv run python -m scripts.pipeline run=phase1b-1.7b
     uv run python -m scripts.pipeline run=phase1b-8b dry_run=true
     uv run python -m scripts.pipeline run=phase1b-1.7b only=[significance]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.observe import banner, duration, log, set_log_file

SPEC_DIR = Path(__file__).resolve().parent.parent / "specs" / "pipeline"

# The line Budget prints when it exits on the allowance rather than on finishing.
# `repeat` re-invokes on it, so this is a contract between the two and not a guess.
BUDGET_MARKER = "stopping cleanly:"

def compose(overrides):
    """The pipeline config, composed with Hydra's grammar so a run can be swapped or edited inline"""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(SPEC_DIR), version_base=None):
        cfg = hydra_compose(config_name="config", overrides=list(overrides))
    GlobalHydra.instance().clear()
    return OmegaConf.to_container(cfg, resolve=True)

def state_path(env: dict) -> Path:
    return Path(env.get("MI_LAB_RESULTS", "results")) / "pipeline-state.json"

def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"done": {}}

def run_step(step: dict, config: str, env: dict) -> dict:
    """One step as a subprocess, repeated while it says it stopped on budget"""
    command = [sys.executable, "-m", step["module"], config, *[str(a) for a in step.get("args", [])]]
    environment = {**os.environ, **{key: str(value) for key, value in env.items()}}
    invocations, started, budget_exits = 0, time.time(), 0
    while True:
        invocations += 1
        log(f"$ {' '.join(command)}", indent=1)
        finished = subprocess.run(command, env=environment, capture_output=True, text=True)
        tail = (finished.stdout or "").strip().splitlines()
        for line in tail[-6:]:
            log(line, indent=2)
        if finished.returncode != 0:
            log(f"!! {step['name']} failed with exit {finished.returncode}", indent=1)
            for line in (finished.stderr or "").strip().splitlines()[-15:]:
                log(line, indent=2)
            raise SystemExit(
                f"step '{step['name']}' failed. The pipeline stops here rather than running the steps "
                "after it, because every one of them reads what this was supposed to write."
            )
        hit_budget = any(BUDGET_MARKER in line for line in tail)
        if hit_budget:
            budget_exits += 1
        if not (step.get("repeat") and hit_budget):
            break
        log(f"{step['name']} stopped on its budget; re-invoking ({invocations} so far)", indent=1)
    return {"invocations": invocations, "budget_exits": budget_exits,
            "seconds": round(time.time() - started, 1)}

def main() -> None:
    cfg = compose(sys.argv[1:])
    run = cfg["run"]
    env = dict(run.get("env") or {})
    steps = [s for s in run["steps"] if not cfg["only"] or s["name"] in cfg["only"]]

    Path(env.get("MI_LAB_RESULTS", "results")).mkdir(parents=True, exist_ok=True)
    set_log_file(Path(env.get("MI_LAB_RESULTS", "results")) / "pipeline.log")
    state = load_state(state_path(env))
    banner(f"pipeline: {run['name']}", {
        "config": run["config"],
        "steps": f"{len(steps)} selected of {len(run['steps'])}, {len(state['done'])} already done",
        "environment": ", ".join(f"{k}={v}" for k, v in env.items()) or "(inherited)",
        "resume": cfg["resume"],
        "state": str(state_path(env)),
    })
    for index, step in enumerate(steps, start=1):
        marker = f"{step['module']}:{step['name']}"
        if cfg["resume"] and marker in state["done"]:
            log(f"[{index}/{len(steps)}] {step['name']} already done "
                f"({duration(state['done'][marker]['seconds'])}) -- skipping")
            continue
        log(f"[{index}/{len(steps)}] {step['name']}")
        if cfg["dry_run"]:
            log(f"$ {sys.executable} -m {step['module']} {run['config']} "
                f"{' '.join(str(a) for a in step.get('args', []))}", indent=1)
            continue
        record = run_step(step, run["config"], env)
        state["done"][marker] = record
        state_path(env).write_text(json.dumps(state, indent=2) + "\n")
        log(f"{step['name']} done in {duration(record['seconds'])} "
            f"({record['invocations']} invocation(s))", indent=1)
    log("dry run: nothing executed" if cfg["dry_run"] else f"pipeline '{run['name']}' complete")

if __name__ == "__main__":
    main()
