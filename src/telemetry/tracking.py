"""Mirror a run's metrics to an MLflow server, without depending on MLflow.

The journal beside this module is the record: a file on this disk, flushed
every step, readable while the run is going and still there when it is not.
That is the thing a two-hour run cannot do without. What it does not give you
is the other half of the question -- how does this run compare to the four
before it -- and that is what a tracking server is for.

So this is a *sink*, not a replacement. Every metric goes to the journal first
and reaches MLflow second, and the ordering is the point: if the server is
down, moved, or slow, the run still has its curve.

Talks to MLflow's REST API over urllib rather than importing `mlflow`. Two
reasons, and neither is dependency squeamishness for its own sake. The package
this lives in is stdlib-only so that a result can be read on a machine that
cannot load the model that produced it, and pulling a tracking client with its
own transitive tree into it would end that. And what is actually needed here
is four calls -- create an experiment, create a run, log a batch, set a status
-- against an API that has been stable at 2.0 for years. `mlflow` would be the
right answer the moment artifacts, autologging or the model registry are
wanted; it is not needed to put numbers on a chart.

**Nothing here may raise into a training loop.** A tracker that kills a
two-hour run because a pod restarted is worse than no tracker, so every network
failure disables the sink, says so once, and returns. `failed` records the
reason, and the journal is unaffected either way.

Metrics are buffered and flushed in batches. At 3.3 seconds a step a per-step
POST would be fine, and at 0.5 seconds -- which is what gpt2-small runs at --
it is two thousand round trips against a server that accepts them a hundred at
a time.

Credentials, if the server ever enforces them, come from the environment
(`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, the names the real
client uses) and never from the config file, which is committed.

Every run in this repo goes through `track`, which is the whole of the policy:
one MLflow run per invocation, tagged with the commit and whether the tree was
dirty, closed FINISHED or FAILED with the error, and carrying as artifacts
every file the run wrote under the paths it names. A result on disk that
MLflow does not know about is a result nobody can trace back to the code that
made it. On by default; `MI_LAB_TRACKING=none` turns it off, and so does
running under `python -m unittest`, so the suite never posts.

A common pipe could be: track | tracker.log per step | finish | mlflow

Run: uv run python -m scripts.sheaf_prune gpt2-small --tracking mlflow
"""

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "tracking"

# Cloudflare sits in front of the tunnel and refuses `Python-urllib/3.x` with a
# 403 as suspected-bot traffic, which is not a thing the MLflow API ever sees
# and so reads as an auth problem. Any identifying agent passes; this one says
# who it is. The in-cluster address is unaffected, which is exactly what made
# the first diagnosis wrong -- curl worked, the client did not, same URL.
USER_AGENT = "mi-lab-telemetry/1.0"

# The largest file `track` uploads as an artifact. Results are JSON, logs and
# plots; what is bigger than this is weights, masks and caches, which have a
# home on disk and would turn every run into a multi-gigabyte upload.
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

# The run id a tracked process hands to the processes it starts. The pipeline
# runs each step as a subprocess, and a step that finds this in its
# environment files itself under that run -- MLflow's own parent tag -- so a
# phase reads as one run with its steps inside it instead of nine strangers.
PARENT = "MI_LAB_MLFLOW_PARENT"

class TrackingError(ValueError):
    """A tracking config that cannot be read, said with the way out"""

@dataclass
class TrackingConfig:
    """Where runs are mirrored, and whether they are at all

    `enabled: false` is a first-class setting rather than an absent file, so a
    run that deliberately does not track says so in its own config instead of
    being indistinguishable from one that forgot.
    """
    enabled: bool = False
    uri: str = ""
    experiment: str = "mi-lab"
    timeout: float = 10.0
    flush_every: int = 25
    tags: Dict[str, str] = field(default_factory=dict)

def from_mapping(data: Dict[str, Any]) -> TrackingConfig:
    """Build a TrackingConfig, rejecting unknown keys rather than ignoring them

    The same rule `core.config.from_mapping` keeps, for the same reason: a
    typo'd key that is silently dropped is a run that quietly tracked nothing.
    """
    data = dict(data)
    known = set(TrackingConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise TrackingError(
            f"unknown tracking keys {sorted(unknown)}; known keys are {sorted(known)}"
        )
    return TrackingConfig(**data)

def load_tracking(reference: str) -> TrackingConfig:
    """Load a tracking config by name from configs/tracking/, or by path"""
    if reference in ("", "none", "off"):
        return TrackingConfig(enabled=False)
    for path in (CONFIG_DIR / f"{reference}.yaml", Path(reference)):
        if path.exists():
            import yaml

            return from_mapping(yaml.safe_load(path.read_text()) or {})
    shipped = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.is_dir() else []
    raise TrackingError(
        f"'{reference}' is neither a tracking config in {CONFIG_DIR} nor an existing file; "
        f"shipped configs are {shipped}, and 'none' disables tracking"
    )

def _now_ms() -> int:
    return int(time.time() * 1000)

class Tracker:
    """One MLflow run, mirrored from a journal, and disabled the moment it misbehaves

    `active` is the only thing a caller needs to read. It starts false when
    tracking is off and goes false on the first failure, so a partially
    recorded run is visible as one rather than being mistaken for a complete
    one that happens to stop early.
    """

    def __init__(self, config: TrackingConfig, name: str = "",
                 params: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.active = False
        self.run_id: Optional[str] = None
        self.artifact_root = ""
        self.failure: Optional[str] = None
        self._buffer: List[dict] = []
        if not config.enabled or not config.uri:
            return
        try:
            experiment_id = self._experiment(config.experiment)
            body = self._post("runs/create", {
                "experiment_id": experiment_id,
                "run_name": name,
                "start_time": _now_ms(),
                "tags": [{"key": k, "value": str(v)} for k, v in config.tags.items()],
            })
            self.run_id = body["run"]["info"]["run_id"]
            self.artifact_root = body["run"]["info"].get("artifact_uri", "")
            self.active = True
            if params:
                self.log_params(params)
        # Deliberately broad: a tracker may never raise into a training loop.
        except Exception as error:
            self._disable(error)

    @property
    def url(self) -> str:
        if not self.run_id:
            return ""
        return f"{self.config.uri.rstrip('/')}/#/experiments/_/runs/{self.run_id}"

    def _disable(self, error: object) -> None:
        self.failure = f"{type(error).__name__}: {error}"
        self.active = False

    def _request(self, path: str, payload: Optional[dict], method: str) -> dict:
        url = f"{self.config.uri.rstrip('/')}/api/2.0/mlflow/{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        request.add_header("User-Agent", USER_AGENT)
        user = os.environ.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if user and password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read() or b"{}")

    def _post(self, path: str, payload: dict) -> dict:
        return self._request(path, payload, "POST")

    def _experiment(self, name: str) -> str:
        """The experiment id, creating it the first time and reusing it after

        `create` answers 400 RESOURCE_ALREADY_EXISTS rather than returning the
        existing id, so the get-by-name is the fallback and not the other way
        round -- doing it in the other order costs a round trip on every run
        after the first.
        """
        try:
            return self._post("experiments/create", {"name": name})["experiment_id"]
        except urllib.error.HTTPError as error:
            if error.code not in (400, 409):
                raise
            found = self._request(
                f"experiments/get-by-name?experiment_name={urllib.parse.quote(name)}", None, "GET")
            return found["experiment"]["experiment_id"]

    def log_params(self, params: Dict[str, Any]) -> None:
        """Params are strings to MLflow, and a None is written as one

        Dropping a null would make "gated every layer" (`layers: None`) and
        "this run did not record a band" the same row.
        """
        if not self.active:
            return
        try:
            self._post("runs/log-batch", {
                "run_id": self.run_id,
                "params": [{"key": str(k), "value": str(v)[:6000]} for k, v in params.items()],
            })
        except Exception as error:
            self._disable(error)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Tags are how a run is found again: the commit, the script, the config"""
        if not self.active:
            return
        try:
            self._post("runs/log-batch", {
                "run_id": self.run_id,
                "tags": [{"key": str(k), "value": str(v)[:5000]} for k, v in tags.items()],
            })
        except Exception as error:
            self._disable(error)

    def log_artifact(self, path: Path, name: Optional[str] = None) -> bool:
        """Upload one file through the server's artifact proxy; False if it was not sent

        The run's artifact root is `s3://<bucket>/<experiment>/<run>/artifacts`
        and the proxy is addressed by the part after the bucket, which is what
        the stock client does too. A failed upload disables the sink like any
        other failure, and a file over MAX_ARTIFACT_BYTES is skipped, not sent.
        """
        if not self.active or not self.artifact_root:
            return False
        path = Path(path)
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            return False
        location = urllib.parse.urlparse(self.artifact_root)
        prefix = location.path.lstrip("/") if location.scheme in ("mlflow-artifacts", "") else \
            f"{location.netloc}{location.path}".split("/", 1)[1]
        target = urllib.parse.quote(f"{prefix}/{name or path.name}")
        url = f"{self.config.uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/{target}"
        request = urllib.request.Request(url, data=path.read_bytes(), method="PUT")
        request.add_header("User-Agent", USER_AGENT)
        try:
            with urllib.request.urlopen(request, timeout=max(self.config.timeout, 60.0)):
                return True
        except Exception as error:
            self._disable(error)
            return False

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        """Buffer one step; flush when the buffer is full"""
        if not self.active:
            return
        stamp = _now_ms()
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self._buffer.append({"key": str(key), "value": float(value),
                                 "timestamp": stamp, "step": int(step)})
        if len(self._buffer) >= self.config.flush_every * max(1, len(metrics)):
            self.flush()

    def flush(self) -> None:
        """Send what is buffered, in chunks the server will accept

        MLflow caps a log-batch at a thousand metrics, and a run that buffered
        more than that between flushes would otherwise fail the whole batch.
        """
        if not self.active or not self._buffer:
            return
        pending, self._buffer = self._buffer, []
        try:
            for start in range(0, len(pending), 1000):
                self._post("runs/log-batch",
                           {"run_id": self.run_id, "metrics": pending[start : start + 1000]})
        except Exception as error:
            self._disable(error)

    def finish(self, status: str = "FINISHED", summary: Optional[Dict[str, Any]] = None) -> None:
        """Flush, record the final numbers as metrics, and close the run"""
        if not self.active:
            return
        self.flush()
        if summary:
            self.log(0, {f"final_{k}": v for k, v in summary.items()})
            self.flush()
        try:
            self._post("runs/update",
                       {"run_id": self.run_id, "status": status, "end_time": _now_ms()})
        except Exception as error:
            self._disable(error)
        self.active = False

def default_tracking() -> str:
    """'mlflow' unless the environment says otherwise, and 'none' under the test runner

    Tracking is the default because a run that was not recorded cannot be
    traced, and "I forgot the flag" is how that happens. The suite is the one
    caller that must never post, and it is recognized by its entry point so no
    test has to remember to switch it off.
    """
    if "MI_LAB_TRACKING" in os.environ:
        return os.environ["MI_LAB_TRACKING"]
    entry = Path(sys.argv[0]) if sys.argv and sys.argv[0] else Path()
    if entry.parent.name == "unittest" or "unittest" in entry.name:
        return "none"
    return "mlflow"

def provenance() -> Dict[str, str]:
    """What a result has to carry to be traced back: the commit, whether it was the whole story, the command

    `dirty` matters as much as the commit. This repo is routinely run from an
    uncommitted tree, and a run tagged with a commit it did not actually use
    is a trace that points at the wrong code with complete confidence.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                                  check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {
        "git_commit": git("rev-parse", "HEAD") or "unknown",
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no"))).lower(),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "mlflow.source.name": sys.argv[0] if sys.argv else "",
        # The MI_LAB_* variables move results roots and corpus sizes, so they are part of what ran
        **{f"env.{key}": value for key, value in sorted(os.environ.items())
           if key.startswith("MI_LAB_") and key != PARENT},
    }

def _written_since(paths: Sequence[Path], since: float) -> List[Path]:
    """Every file under `paths` modified at or after `since`, which is what this run wrote"""
    found = []
    for base in paths:
        base = Path(base)
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        found.extend(path for path in candidates
                     if path.is_file() and path.stat().st_mtime >= since and "__pycache__" not in path.parts)
    return found

@contextmanager
def track(name: str, experiment: str, params: Optional[Dict[str, Any]] = None,
          outputs: Sequence[Path] = (), tracking: Optional[str] = None,
          tags: Optional[Dict[str, Any]] = None) -> Iterator[Tracker]:
    """One MLflow run around a block: provenance in, status and written files out

    `experiment` groups runs of one question -- a script, a study -- and
    replaces the config's own, which names the one experiment `sheaf_prune`
    reports to. `outputs` are the directories or files this run writes into;
    whatever in them changed while the block ran is uploaded when it ends,
    whether it ended well or not, because the log of a failed run is the part
    most worth having.

    Yields the Tracker, inactive when tracking is off or the server is
    unreachable, so the body can `log` unconditionally. The block's own
    exception is re-raised untouched after the run is marked FAILED.
    """
    config = replace(load_tracking(tracking if tracking is not None else default_tracking()),
                     experiment=experiment)
    started = time.time() - 1.0
    tracker = Tracker(config, name=name, params=params)
    parent = os.environ.get(PARENT)
    tracker.set_tags({**provenance(), **({"mlflow.parentRunId": parent} if parent else {}), **(tags or {})})
    if tracker.active:
        os.environ[PARENT] = tracker.run_id
    status = "FINISHED"
    try:
        yield tracker
    except SystemExit as error:
        # `--help`, and a main returning 0 through sys.exit, are clean exits and not failures
        if error.code not in (None, 0):
            status = "FAILED"
            tracker.set_tags({"error": f"SystemExit: {error.code}"})
        raise
    except BaseException as error:
        status = "KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        tracker.set_tags({"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if tracker.active:
            if parent:
                os.environ[PARENT] = parent
            else:
                os.environ.pop(PARENT, None)
        for path in _written_since([Path(p) for p in outputs], started):
            tracker.log_artifact(path, name=str(path).lstrip("/"))
        tracker.finish(status)

def tracked_main(main: Callable[[], Any], experiment: str, outputs: Sequence[Path] = ()) -> Any:
    """Run a script's `main` inside `track`, named after the script and its arguments

    The one line every `scripts/*.py` ends with. The script's own metrics are
    in the files it writes, so those are what go up: everything under
    `outputs` (the results root, for the phase scripts) that changed while it
    ran. A script with numbers worth charting also calls `track` itself --
    `wsbench_baseline` and `sheaf_prune` do -- and does not use this.

    A main that returns an exit code gets it back, and a nonzero one closes
    the run FAILED on its way out, the same as an exception would.
    """
    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "script"
    arguments = sys.argv[1:]
    # a directory named on the command line is where scripts like sheaf_infer write
    named = [Path(argument) for argument in arguments if Path(argument).is_dir()]
    with track(" ".join([script, *arguments]), experiment, outputs=[*outputs, *named],
               params={"script": script, "argv": " ".join(arguments)}, tags={"script": script}):
        code = main()
        if code not in (None, 0):
            raise SystemExit(code)
        return code
