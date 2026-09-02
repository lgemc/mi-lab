"""Run a phase as a sequence of steps named in a file, rather than in a paragraph.

Phase 1b is nine invocations in an order that matters: the FLOPs table before
anything that divides by it, the sweep before the ranking, the ranking before
the significance gate that decides whether the ranking may be used. That order
lived in commit messages and in whoever ran it last, which is the same place a
protocol goes to be misremembered. So the sequence is a Hydra config under
pipelines/, and `experiment.pipeline` composes and runs it; this is the
command line over that module and nothing else.

Steps run as subprocesses rather than imports, because each phase1b script
reads its results root from the environment at import time. Resumable: each
completed step is flushed to a state file beside the results and skipped on
re-entry, and a `repeat` step is re-invoked while it stops on its budget.

A common pipe could be: compose | Pipeline.from_mapping | run

Run: uv run python -m scripts.pipeline run=phase1b-1.7b
     uv run python -m scripts.pipeline run=phase1b-8b dry_run=true
     uv run python -m scripts.pipeline run=phase1b-1.7b only=[significance]
"""

import sys

from src.experiment.pipeline import Pipeline, PipelineError, compose, run


def main() -> None:
    try:
        run(Pipeline.from_mapping(compose(sys.argv[1:])))
    except PipelineError as error:
        raise SystemExit(str(error)) from None

if __name__ == "__main__":
    main()
