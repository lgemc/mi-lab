import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

"""
A Run is what an experiment leaves behind: which spec produced it, whether it
finished, what it measured, and what files it wrote. It is deliberately plain
data -- this module imports nothing but the standard library, so a run can be
read back by anything, including a machine that has no torch installed.

A failed run is still a run. The status and the error are recorded rather than
the directory being cleaned up, because the runs that fail are the ones worth
being able to look at later.

A common pipe could be: Run.start | record | finish | save
"""

STATUSES = ("running", "completed", "failed")

class RunError(ValueError):
    """Raised when a run on disk is missing or malformed"""

@dataclass(frozen=True)
class Ref:
    """A pointer to something a run produced, as (kind, path relative to the run)"""
    kind: str
    id: str

@dataclass
class Run:
    """One execution of one spec"""
    run_id: str
    experiment: str
    kind: str
    spec_hash: str
    status: str = "running"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    produced: List[Ref] = field(default_factory=list)
    created_at: str = ""
    finished_at: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def start(cls, experiment: str, kind: str, spec_hash: str, params: Optional[Dict[str, Any]] = None) -> "Run":
        """Open a run, stamped with a sortable id that carries its own spec hash

        The id leads with the timestamp so a directory listing is chronological,
        and ends with the spec hash so two runs of the same experiment are
        distinguishable at a glance from two runs of different ones.
        """
        started = datetime.now(timezone.utc)
        return cls(
            run_id=f"{started.strftime('%Y%m%d-%H%M%S')}-{spec_hash}",
            experiment=experiment,
            kind=kind,
            spec_hash=spec_hash,
            params=dict(params or {}),
            created_at=started.isoformat(),
        )

    def record(self, **metrics: float) -> "Run":
        """Add measured numbers to the run"""
        self.metrics.update({key: float(value) for key, value in metrics.items()})
        return self

    def produce(self, kind: str, name: str) -> "Run":
        """Note that the run wrote a file, by kind and name relative to its directory"""
        self.produced.append(Ref(kind=kind, id=name))
        return self

    def finish(self, error: Optional[BaseException] = None) -> "Run":
        """Close the run as completed, or as failed with the reason why"""
        self.status = "failed" if error is not None else "completed"
        self.error = f"{type(error).__name__}: {error}" if error is not None else None
        self.finished_at = datetime.now(timezone.utc).isoformat()
        return self

    @property
    def duration_seconds(self) -> Optional[float]:
        """Wall-clock time from start to finish, or None while still running"""
        if not self.finished_at:
            return None
        return (datetime.fromisoformat(self.finished_at) - datetime.fromisoformat(self.created_at)).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Run":
        payload = dict(payload)
        payload["produced"] = [Ref(**ref) for ref in payload.get("produced", [])]
        missing = {"run_id", "experiment", "kind", "spec_hash"} - set(payload)
        if missing:
            raise RunError(f"run is missing fields {sorted(missing)}")
        return cls(**payload)

    def save(self, directory: str) -> str:
        """Write run.json into the given directory, creating it if needed"""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "run.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return str(path)

    @classmethod
    def load(cls, directory: str) -> "Run":
        """Read a run back from its directory"""
        path = Path(directory) / "run.json"
        if not path.exists():
            raise RunError(f"no run.json in {directory}")
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except json.JSONDecodeError as error:
            raise RunError(f"{path} is not valid JSON") from error

def find_runs(root: str) -> List["Run"]:
    """Every run under a root directory, newest first

    A directory without a readable run.json is skipped rather than fatal: a
    run killed mid-write should not make the whole listing unusable.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    runs = []
    for directory in sorted(base.iterdir(), reverse=True):
        try:
            runs.append(Run.load(str(directory)))
        except RunError:
            continue
    return runs
