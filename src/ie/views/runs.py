from pathlib import Path
from typing import ClassVar, Dict, List

from ...experiment.run import Run, RunError
from ...share import storage
from ..view import Detail, Hint, Row, View

"""
What this machine has actually run, newest first.

A run directory is the record of a thing that happened here, and unlike an
artifact it is not meant to leave the machine -- so this view is the one that
answers "what did I do last night" without touching a model. Failed runs are
listed and marked rather than hidden: a failed run is still written out on
purpose, and a listing that dropped them would make the promise unverifiable.

The walk here is `find_runs` done again rather than called, because a Run does
not carry the directory it was read from and this view needs it: an ioi_circuit
run writes a circuit.mia beside its run.json, and a listing that could not
reach it leaves the result of the run one directory away and unreachable.

A common pipe could be: :runs | a | :nodes
"""

def _walk(root: str):
    """Every run under a root as (record, directory), newest first

    `find_runs` returns the records alone, which is right for a listing and not
    enough for an explorer: what a run produced sits beside its run.json.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found = []
    for marker in base.rglob("run.json"):
        try:
            found.append((Run.load(str(marker.parent)), str(marker.parent)))
        except RunError:
            continue
    return sorted(found, key=lambda pair: pair[0].run_id, reverse=True)

def _artifacts_in(directory: str):
    """The .mia directories a run wrote, loaded so the listing can name them"""
    found = []
    for path in sorted(Path(directory).glob(f"*{storage.SUFFIX}")):
        try:
            found.append(storage.load(str(path)))
        except Exception:
            continue
    return found

class Runs(View):
    """Every run under the session root, newest first"""

    title = "runs"
    columns = ("status", "run id", "experiment", "kind", "created", "metrics")
    hints = (Hint("enter", "run detail"), Hint("a", "artifacts it produced"), Hint("y", "record"))
    keys: ClassVar[Dict[str, str]] = {"a": "open_artifact"}

    columns = ("status", "run id", "experiment", "kind", "created", "shipped", "metrics")

    def rows(self) -> List[Row]:
        found = []
        for run, directory in _walk(self.session.root):
            summary = "  ".join(f"{name}={value:.3g}" for name, value in list(run.metrics.items())[:3])
            shipped = _artifacts_in(directory)
            found.append(Row(
                key=run.run_id,
                cells=(
                    run.status, run.run_id, run.experiment, run.kind, run.created_at[:19],
                    f"{len(shipped)} .mia" if shipped else "-", summary,
                ),
                payload=(run, directory, shipped),
            ))
        return found

    def open_artifact(self) -> None:
        """`a`: open what this run shipped, which is the point of having run it"""
        row = self.selected()
        if row is None:
            self.explorer.flash("nothing selected")
            return
        run, _, shipped = row.payload
        if not shipped:
            self.explorer.flash(
                f"{run.run_id} ({run.kind}) wrote no {storage.SUFFIX} -- only ioi_circuit does so far"
            )
            return
        self.session.artifact = shipped[0]
        if len(shipped) > 1:
            self.explorer.flash(f"{len(shipped)} artifacts here; opened {shipped[0].id}")
        self.explorer.run_command("nodes" if shipped[0].nodes else "tensors")

    def _empty_note(self) -> str:
        if self.filter:
            return super()._empty_note()
        return (
            f"no runs under '{self.session.root}' -- run one with "
            "`python -m src.cli run exec -e ioi-circuit`, or point elsewhere with `:root <dir>`"
        )

    def on_enter_row(self, row: Row) -> None:
        run, directory, shipped = row.payload
        record = {
            "run_id": run.run_id, "experiment": run.experiment, "kind": run.kind,
            "status": run.status, "spec_hash": run.spec_hash,
            "created_at": run.created_at, "finished_at": run.finished_at,
            "error": run.error or "-",
            "produced": [f"{ref.kind}:{ref.id}" for ref in run.produced],
            "directory": directory,
            "artifacts": [artifact.id for artifact in shipped] or "none -- press <a> for why",
        }
        record.update({f"param.{name}": value for name, value in run.params.items()})
        record.update({f"metric.{name}": value for name, value in run.metrics.items()})
        self.explorer.push(Detail(self.explorer, self.session, f"run {run.run_id}", record))

    def detail(self, row: Row):
        run = row.payload[0] if row.payload else None
        return None if run is None else {
            "run_id": run.run_id, "experiment": run.experiment, "kind": run.kind,
            "status": run.status, "spec_hash": run.spec_hash, "params": run.params,
            "metrics": run.metrics, "error": run.error or "-",
        }
