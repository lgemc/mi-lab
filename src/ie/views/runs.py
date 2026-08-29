from typing import List

from ...experiment.run import find_runs
from ..view import Detail, Hint, Row, View

"""
What this machine has actually run, newest first.

A run directory is the record of a thing that happened here, and unlike an
artifact it is not meant to leave the machine -- so this view is the one that
answers "what did I do last night" without touching a model. Failed runs are
listed and marked rather than hidden: a failed run is still written out on
purpose, and a listing that dropped them would make the promise unverifiable.

A common pipe could be: :runs | enter | y
"""

class Runs(View):
    """Every run under the session root, newest first"""

    title = "runs"
    columns = ("status", "run id", "experiment", "kind", "created", "metrics")
    hints = (Hint("enter", "run detail"), Hint("y", "record"))

    def rows(self) -> List[Row]:
        found = []
        for run in find_runs(self.session.root):
            summary = "  ".join(f"{name}={value:.3g}" for name, value in list(run.metrics.items())[:3])
            found.append(Row(
                key=run.run_id,
                cells=(run.status, run.run_id, run.experiment, run.kind, run.created_at[:19], summary),
                payload=run,
            ))
        return found

    def _empty_note(self) -> str:
        if self.filter:
            return super()._empty_note()
        return (
            f"no runs under '{self.session.root}' -- run one with "
            "`python -m src.cli run exec -e ioi-circuit`, or point elsewhere with `:root <dir>`"
        )

    def on_enter_row(self, row: Row) -> None:
        run = row.payload
        record = {
            "run_id": run.run_id, "experiment": run.experiment, "kind": run.kind,
            "status": run.status, "spec_hash": run.spec_hash,
            "created_at": run.created_at, "finished_at": run.finished_at,
            "error": run.error or "-",
            "produced": [f"{ref.kind}:{ref.id}" for ref in run.produced],
        }
        record.update({f"param.{name}": value for name, value in run.params.items()})
        record.update({f"metric.{name}": value for name, value in run.metrics.items()})
        self.explorer.push(Detail(self.explorer, self.session, f"run {run.run_id}", record))

    def detail(self, row: Row):
        run = row.payload
        return None if run is None else {
            "run_id": run.run_id, "experiment": run.experiment, "kind": run.kind,
            "status": run.status, "spec_hash": run.spec_hash, "params": run.params,
            "metrics": run.metrics, "error": run.error or "-",
        }
