"""Run a sheaf experiment defined in a file, which is where an experiment belongs.

`scripts/sheaf_prune.py` takes the same parameters as flags and still works;
this is the door to prefer. A run under `sheaves/run/` is a record that can be
read, diffed and reviewed before it costs two hours of GPU, and the shell keeps
Hydra's grammar for the one key you want to change today:

    uv run python -m scripts.sheaf run=qwen-ioi-kl
    uv run python -m scripts.sheaf run=qwen-ioi-kl run.steps=500
    uv run python -m scripts.sheaf run=qwen-ioi-kl run.results=results/scratch

Both doors build the same `SheafSpec` and call the same `run`, so a parameter
that does not exist in the schema cannot be passed to either -- and a key
misspelled in a YAML fails at composition rather than after the model loads.

`results` comes from the spec and is exported before anything reads it, because
`telemetry.results` takes it from the environment: where a run writes is part
of what the run is, and leaving it in the shell is how two experiments end up
sharing a directory.

A common pipe could be: compose | SheafSpec.from_mapping | export results | run

Run: uv run python -m scripts.sheaf --help-runs   (list what is defined)
"""

import os
import sys
from pathlib import Path

from src.experiment.sheaf import SHEAF_DIR, SheafError, SheafSpec, compose


def main() -> None:
    if "--help-runs" in sys.argv[1:] or "--list" in sys.argv[1:]:
        runs = sorted(p.stem for p in (SHEAF_DIR / "run").glob("*.yaml"))
        print(f"sheaf runs defined in {SHEAF_DIR / 'run'}:")
        for name in runs:
            print(f"  {name}")
        print("\nuv run python -m scripts.sheaf run=<name> [run.key=value ...]")
        return
    try:
        spec = SheafSpec.from_mapping(compose(sys.argv[1:]))
    except SheafError as error:
        raise SystemExit(str(error)) from None
    if spec.results:
        # Exported before the import-time readers in telemetry.results see it.
        os.environ["MI_LAB_RESULTS"] = str(spec.results)
        Path(spec.results).mkdir(parents=True, exist_ok=True)

    from scripts.sheaf_prune import run
    from src.telemetry.results import guard

    guard(spec.config)
    run(spec)


if __name__ == "__main__":
    main()
