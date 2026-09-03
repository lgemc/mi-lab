"""What a long run is doing, on disk while it is still doing it.

Every number this repo produces arrives at the end. `prune` accumulates a
`history` list in memory and writes it out when it returns, so a two-hour run
is a blank terminal for two hours and a process killed at minute 118 leaves
nothing at all -- which has happened here, twice. The only way anyone has read
a live figure out of this repo is `py-spy dump --locals` against a running PID,
which is a debugger, not instrumentation.

This is the smallest thing that fixes that, and it is deliberately not a
tracking server. A server wants to be up, and what a run here needs first is
for the number it already computed to reach the disk before the process dies.
So: one append-only JSONL of metrics, flushed every write, plus a `run.json` of
the parameters beside it. `tail -f` is a live view, `jq` is a query, and
nothing has to be running for either to work.

A tracking server answers the *other* question -- how does this run compare to
the four before it -- so `sink` takes one, and `telemetry/tracking.py` mirrors
to MLflow. The ordering is the design: the row is on disk before it is sent,
and a sink that fails costs a comparison rather than a result.

Stdlib only, and no torch -- the same rule `experiment/run.py` keeps and for
the same reason. A journal has to be readable on a machine that cannot load the
model that wrote it, including while the run is still going on another host.

Three things it does not do, each on purpose. It does not buffer, because the
reason it exists is that the process may not survive to flush. It does not
aggregate, because a mean over a run whose price ramps a thousandfold is a
number about nothing -- the reader aggregates, having seen the curve. And it
does not own the metrics: `log` takes whatever it is handed, so adding a term
to a loss is not also a change here.

A common pipe could be: params | journal | log per step | tail | run.json
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

RUN = "run.json"
METRICS = "metrics.jsonl"

class TelemetryError(ValueError):
    """A journal that cannot be written or read, said with the way out"""

def _now() -> float:
    return time.time()

def run_id(name: str = "") -> str:
    """A timestamp-led id, so the lexical order of a directory is its time order

    Led by the stamp rather than the name for the reason `find_runs` orders by
    id: a root accumulates runs and the question asked of it is nearly always
    "what ran last".
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"{stamp}-{name}" if name else stamp

class Journal:
    """One run's parameters and its metric stream, written as they happen

    Opened on construction so that a run which dies before its first step still
    leaves a directory saying what it was trying to do -- the same promise
    `experiment/runner.py` makes by writing a failed run out rather than
    cleaning it up.
    """

    def __init__(self, directory, params: Optional[Dict[str, Any]] = None,
                 name: str = "") -> None:
        self.directory = Path(directory)
        # One run per directory, refused rather than merged: `run_id` is
        # second-resolution, and three sweeps launched from one shell line
        # landed in the same journal, their curves interleaved row by row in
        # one metrics file that each artifact then named as its own.
        if self.directory.exists() and any(self.directory.iterdir()):
            raise TelemetryError(
                f"{self.directory} already holds a run; a journal is one run's record. "
                f"Wait a second between launches, or point this one at another directory."
            )
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TelemetryError(
                f"cannot open a journal at {self.directory}: {error}. Point it somewhere "
                f"writable, or pass journal=None to run without one."
            ) from error
        self.name = name
        self.started = _now()
        self.steps = 0
        self._params = dict(params or {})
        # An optional mirror -- anything with `.log(step, metrics)`. Set after
        # construction so the journal exists before the sink does, which is the
        # order that matters: the file is the record and the sink is a copy.
        self.sink = None
        self._handle = (self.directory / METRICS).open("a", encoding="utf-8")
        self._write_run("running")

    @property
    def metrics_path(self) -> Path:
        return self.directory / METRICS

    @property
    def params(self) -> Dict[str, Any]:
        """A copy of what this run was started with, for a sink that wants it too"""
        return dict(self._params)

    def _write_run(self, status: str, **extra: Any) -> None:
        record = {
            "name": self.name,
            "status": status,
            "started": self.started,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started)),
            "steps_logged": self.steps,
            "params": self._params,
        }
        record.update(extra)
        (self.directory / RUN).write_text(json.dumps(record, indent=2, default=str) + "\n")

    def log(self, step: int, **metrics: Any) -> None:
        """Append one step's metrics, on disk before this returns

        Flushed rather than fsynced: a flush is what makes the line visible to
        another process reading the file, which is the whole point, and an
        fsync per step would charge a disk round trip against a loop whose
        steps are measured in seconds.
        """
        row = {"step": int(step), "elapsed": round(_now() - self.started, 3)}
        for key, value in metrics.items():
            row[key] = round(value, 6) if isinstance(value, float) else value
        self._handle.write(json.dumps(row, default=str) + "\n")
        self._handle.flush()
        self.steps += 1
        # After the flush, always. A sink that raises must not cost the journal
        # the row it already has on disk, and a sink that is slow must not make
        # the on-disk record late.
        if self.sink is not None:
            self.sink.log(int(step), row)

    def finish(self, status: str = "completed", **summary: Any) -> None:
        """Close the stream and stamp the run, once

        Safe to call twice: the second call is the one that happens in a
        `finally` after the first already ran in the happy path.
        """
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()
        elapsed = _now() - self.started
        self._write_run(status, finished=_now(), elapsed=round(elapsed, 3), **summary)

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        self.finish("failed" if kind else "completed",
                    **({"error": f"{kind.__name__}: {value}"} if kind else {}))
        return False

def read_run(directory) -> Dict[str, Any]:
    """The parameters and status of one run"""
    path = Path(directory) / RUN
    if not path.exists():
        raise TelemetryError(f"{path} does not exist; {directory} is not a journal directory")
    return json.loads(path.read_text())

def read_metrics(directory) -> List[Dict[str, Any]]:
    """Every metric row, skipping a torn last line

    A run still in flight can be read at any moment, including the moment
    between a write and its newline, so the final line is allowed to be
    incomplete. Any *earlier* malformed line is a real problem and raises,
    because silently dropping rows out of the middle of a curve is how a
    reader ends up explaining a gap that is not in the data.
    """
    path = Path(directory) / METRICS
    if not path.exists():
        raise TelemetryError(f"{path} does not exist; {directory} holds no metrics")
    rows, lines = [], path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise TelemetryError(
                f"{path}:{index + 1} is not valid JSON, and it is not the last line, so it is "
                f"a damaged journal rather than a write in progress"
            ) from None
    return rows

def journals(root) -> List[Path]:
    """Every journal directory under a root, newest first by id"""
    base = Path(root)
    if not base.exists():
        return []
    found = [path.parent for path in base.rglob(RUN)]
    return sorted(set(found), key=lambda path: path.name, reverse=True)

def latest(root) -> Optional[Path]:
    """The most recent journal under a root, or None where there is none"""
    found = journals(root)
    return found[0] if found else None

def progress(directory) -> Dict[str, Any]:
    """Where a run has got to, and what that projects -- computed, not stored

    Derived on read rather than written each step, because a rate written into
    the file is a rate that is wrong the moment the run slows down, and because
    the numbers it is derived from are already there.
    """
    run, rows = read_run(directory), read_metrics(directory)
    total = run.get("params", {}).get("steps")
    last = rows[-1] if rows else {}
    elapsed = last.get("elapsed", run.get("elapsed", 0.0))
    done = last.get("step", -1) + 1
    rate = elapsed / done if done else 0.0
    remaining = rate * (total - done) if total and done and total > done else 0.0
    return {
        "status": run.get("status", "unknown"),
        "step": last.get("step"),
        "total": total,
        "fraction": round(done / total, 4) if total else None,
        "elapsed": round(elapsed, 1),
        "seconds_per_step": round(rate, 3),
        "eta_seconds": round(remaining, 1),
        "last": last,
    }

def to_columns(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """The rows as one list per metric, for a plot or a table

    Missing keys become None rather than being dropped, so every column is the
    same length as every other and a metric that started being logged halfway
    through does not silently shift against the step axis.
    """
    keys: List[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    return {key: [row.get(key) for row in rows] for key in keys}

def tail(directory, count: int = 10) -> Iterator[Dict[str, Any]]:
    """The last `count` metric rows"""
    rows = read_metrics(directory)
    return iter(rows[-max(1, count):])

def env_root(default: str = "journals") -> Path:
    """Where journals land: MI_LAB_JOURNALS, or `journals/` beside the run

    One environment variable, the way `paths.ROOT` takes `MI_LAB_RESULTS`, so
    moving every journal somewhere else is one place and not an argument
    threaded through every script.
    """
    return Path(os.environ.get("MI_LAB_JOURNALS", default))
