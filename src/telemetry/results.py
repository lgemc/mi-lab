"""Where a phase's artifacts land, and the refusal that keeps two models apart.

Every phase script writes to a fixed set of filenames under one root, and
every one of them is resumable: a stage that finds its progress file skips the
work it records. That combination is safe exactly as long as the file and the
model agree. Point a sweep at a second checkpoint and it reads 306 cached
component scores belonging to a different network, reports them as its own and
skips the run -- and a means cache is worse than that, because it is reused
whenever its layers cover the ones asked for, so a 28-layer model would ablate
toward a 36-layer model's activations and produce a number rather than an error.

So the root is settable and the model is stamped into it. `MI_LAB_RESULTS`
moves every artifact somewhere else in one place, and `guard` records which
config first wrote a directory and refuses a second one. The refusal is the
point: the failure it replaces is silent, produces plausible numbers, and would
be found -- if at all -- long after the numbers were quoted.

Beside the root sit the two shapes a resumable script keeps on disk: a
progress state that is one JSON document rewritten whole after every unit of
work, and a report that several scripts each own one section of. Both are
read-if-present-else-default, which is the whole of what "resumable" means at
the file level, and both write the file in one call so a killed process leaves
the previous version rather than half of the next.

Stdlib only, like the rest of `telemetry`: a results directory has to be
readable on a machine that cannot load the model that wrote it.

A common pipe could be: MI_LAB_RESULTS=... | guard | result | load_state | save_state
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

ENV_ROOT = "MI_LAB_RESULTS"
DEFAULT_ROOT = "results"
STAMP = ".model"

class ResultsError(ValueError):
    """A results directory that would mix two models, said with the way out"""

def root() -> Path:
    """The active results root, read from the environment on every call

    Re-read rather than cached at import so a pipeline that sets the variable
    per step, and a test that points it at a temporary directory, both get the
    answer they set rather than the one the first import saw.
    """
    return Path(os.environ.get(ENV_ROOT, DEFAULT_ROOT))

def result(name: str) -> Path:
    """One artifact under the active root"""
    return root() / name

def owner(directory: Optional[Path] = None) -> Optional[str]:
    """Which config stamped a results directory, or None if nobody has yet"""
    stamp = (directory or root()) / STAMP
    if not stamp.exists():
        return None
    return json.loads(stamp.read_text())["config"]

def guard(config: str, directory: Optional[Path] = None) -> Path:
    """Bind a results directory to one config, or refuse

    First writer stamps it. Everyone after has to match. The message names the
    way out rather than only the problem, because the way out is a one-line
    environment variable and nobody should have to find that out by reading
    this file.
    """
    directory = directory or root()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = directory / STAMP
    holder = owner(directory)
    if holder is None:
        stamp.write_text(json.dumps({"config": config}) + "\n")
        return directory
    if holder != config:
        raise ResultsError(
            f"{directory}/ holds results for '{holder}' and this is '{config}'. Every artifact in there is "
            f"named for its phase and not for its model, and the resumable stages would read "
            f"'{holder}' numbers as their own cache and skip the work.\n\n"
            f"  {ENV_ROOT}={DEFAULT_ROOT}/{config} uv run python -m scripts.<script> {config} ...\n\n"
            f"gives '{config}' its own directory. Or unset the stamp at {stamp} if you are certain "
            f"the contents really do belong to '{config}'."
        )
    return directory

def load_state(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A resumable script's progress, or `default` (a fresh copy) if it has not started"""
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(json.dumps(default or {}))

def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Rewrite the progress file whole, creating its directory on the first write"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

def merge_section(path: Path, section: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace one top-level section of a shared JSON report, keeping the others

    Several scripts each contribute one deliverable to a feasibility report;
    each rewrites only its own key, so running them in any order, or one of
    them again, leaves the rest of the report as it was.
    """
    data = load_state(path)
    data[section] = payload
    save_state(path, data)
    return data
