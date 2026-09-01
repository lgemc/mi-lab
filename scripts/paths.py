"""Where a phase's artifacts land, and the refusal that keeps two models apart.

Every phase1b script writes to a fixed set of filenames under one root, and
every one of them is resumable: a stage that finds its progress file skips the
work it records. That combination is safe exactly as long as the file and the
model agree. Point the sweep at a second checkpoint and it reads 306 cached
component scores belonging to a different network, reports them as its own and
skips the run -- and `capture_means` is worse than that, because it reuses any
cache whose layers cover the ones asked for, so a 28-layer model would ablate
toward a 36-layer model's activations and produce a number rather than an error.

So the root is settable and the model is stamped into it. `MI_LAB_RESULTS`
moves every artifact somewhere else in one place, and `guard` records which
config first wrote a directory and refuses a second one. The refusal is the
point: the failure it replaces is silent, produces plausible numbers, and would
be found -- if at all -- long after the numbers were quoted.

A common pipe could be: MI_LAB_RESULTS=... | guard | result | run
"""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("MI_LAB_RESULTS", "results"))

STAMP = ".model"

def result(name: str) -> Path:
    """One artifact under the active root"""
    return ROOT / name

def guard(config: str) -> None:
    """Bind this results directory to one config, or refuse

    First writer stamps it. Everyone after has to match. The message names the
    way out rather than only the problem, because the way out is a one-line
    environment variable and nobody should have to find that out by reading
    this file.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    stamp = ROOT / STAMP
    if not stamp.exists():
        stamp.write_text(json.dumps({"config": config}) + "\n")
        return
    owner = json.loads(stamp.read_text())["config"]
    if owner != config:
        raise SystemExit(
            f"{ROOT}/ holds results for '{owner}' and this is '{config}'. Every artifact in there is "
            f"named for its phase and not for its model, and the resumable stages would read "
            f"'{owner}' numbers as their own cache and skip the work.\n\n"
            f"  MI_LAB_RESULTS=results/{config} uv run python -m scripts.<script> {config} ...\n\n"
            f"gives '{config}' its own directory. Or unset the stamp at {stamp} if you are certain "
            f"the contents really do belong to '{config}'."
        )
