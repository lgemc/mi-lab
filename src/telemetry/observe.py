"""What a long GPU run has to say about itself while it is still running.

Every ablation script is a chain of multi-minute passes over a large model: a
means capture, a baseline translation, one ablated translation per component.
Between them the terminal shows nothing, and a silent process is
indistinguishable from a hung one -- which is exactly what happened on the
first run of the random-control arm, where the checkpoint loaded and the next
line was three minutes away.

Silence is also expensive in a second way. These runs are budgeted in
wall-clock seconds and resumed across invocations, so "how far in am I, and
will the next item fit in what is left" is a question the operator has to
answer *during* the run to use the budget at all. A progress line that carries
an ETA and the remaining budget answers it; a spinner does not.

So: every long step announces itself before it blocks, reports a rate and an
ETA while it runs, and states its elapsed time and peak GPU memory when it
finishes. Everything is timestamped and mirrored to a log file, because a
terminal scrollback is not a record and the interesting line is always the one
that scrolled away.

`Budget` owns one more contract. A run that exits on its allowance prints
`BUDGET_MARKER`, and `experiment/pipeline.py` re-invokes a step on seeing it,
so the marker is a fact both sides read from here rather than a string each
guessed.

GPU memory is read through torch rather than nvidia-smi: on the GB10 the smi
query returns N/A for used and total, and unified memory makes the number
worth watching anyway. torch is imported lazily and inside a guard, which is
how this stays usable from a process that has no torch at all -- the journal
rule, kept for the same reason.

A common pipe could be: banner | step | Progress.tick | Budget.fits | log
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_LOG_FILE: Optional[Path] = None

# The line Budget prints when it exits on the allowance rather than on finishing.
BUDGET_MARKER = "stopping cleanly:"

def set_log_file(path) -> None:
    """Mirror everything logged from here on into `path`, appending; None goes back to stdout only

    Appending rather than truncating because these scripts are resumed: a run
    is a chain of invocations against one budget, and a log that kept only the
    last of them would lose the part where the interesting thing happened.
    """
    global _LOG_FILE
    if path is None:
        _LOG_FILE = None
        return
    _LOG_FILE = Path(path)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(sys.argv)}\n")

def log_file() -> Optional[Path]:
    """Where `log` is mirroring to, or None while it only prints"""
    return _LOG_FILE

def log(message: str = "", indent: int = 0) -> None:
    """One timestamped line, flushed, and mirrored to the log file if one is set"""
    line = f"[{time.strftime('%H:%M:%S')}] {'  ' * indent}{message}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a") as handle:
            handle.write(line + "\n")

def gpu() -> str:
    """Allocated / reserved / peak on the accelerator, or a note that there is none

    Read through torch because nvidia-smi reports N/A for both used and total
    on the GB10; unified memory is the reason to watch it in the first place.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "no cuda"
        giga = 1024 ** 3
        return (f"{torch.cuda.memory_allocated() / giga:.1f}/"
                f"{torch.cuda.memory_reserved() / giga:.1f} GiB, peak "
                f"{torch.cuda.max_memory_allocated() / giga:.1f}")
    except Exception as error:  # observability must never be the thing that fails a run
        return f"gpu unreadable ({type(error).__name__})"

def host_memory_gib(field: str = "MemFree") -> float:
    """A /proc/meminfo figure in GiB, or 0 where it cannot be read

    Unified memory on the GB10 means the accelerator and the host draw on one
    pool, so the host figure is the binding one and `torch.cuda.mem_get_info`
    would report the same pool twice.

    The default is MemFree rather than MemAvailable, which is the more
    pessimistic of the two on purpose. MemAvailable adds the page cache the
    kernel would reclaim under pressure, and a CUDA allocation does not get to
    wait for that: a run once read 27 GiB available here and took
    NV_ERR_NO_MEMORY from the driver two seconds later. The number that
    matters before a big allocation is what is free right now; a feasibility
    report that wants the kernel's own estimate asks for MemAvailable.
    """
    try:
        with Path("/proc/meminfo").open() as handle:
            for line in handle:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0

def duration(seconds: float) -> str:
    """Seconds as the unit a human compares against a budget"""
    total = round(max(0.0, float(seconds)))
    # rounded to whole seconds *before* the carry, or 1799.6s formats as 29m60s
    if total < 90:
        return f"{total}s"
    minutes, rest = divmod(total, 60)
    if minutes < 90:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"

def banner(title: str, fields: dict) -> None:
    """The run's own header: what it is, on what, writing where

    Printed before anything blocks, so a run that dies in the first pass still
    said what it was trying to do.
    """
    log("=" * 78)
    log(title)
    for key, value in fields.items():
        log(f"{key:>18}: {value}")
    log(f"{'gpu':>18}: {gpu()}")
    log("=" * 78)

@contextmanager
def step(label: str, indent: int = 0) -> Iterator[dict]:
    """Announce a blocking step before it blocks, and report what it cost when it ends

    Yields a dict the body can drop numbers into; they are printed on the
    closing line. A step that raises still logs, with the exception named --
    the failure mode this exists for is a pass that dies twenty minutes in and
    leaves a traceback with no indication of which pass it was.
    """
    log(f"-> {label}", indent=indent)
    started = time.time()
    facts: dict = {}
    try:
        yield facts
    except BaseException as error:
        log(f"!! {label} FAILED after {duration(time.time() - started)}: "
            f"{type(error).__name__}: {error}", indent=indent)
        raise
    extra = "".join(f", {key} {value}" for key, value in facts.items())
    log(f"<- {label} in {duration(time.time() - started)} ({gpu()}{extra})", indent=indent)

class Progress:
    """A counter that reports a rate and an ETA rather than only a position

    `every` throttles the printing: a 500-item loop whose items take a second
    does not need 500 lines, and one whose items take a minute needs every one.
    The last tick always prints, so a finished loop never looks truncated.
    """

    def __init__(self, total: int, label: str, every: int = 1, indent: int = 1):
        self.total = max(1, int(total))
        self.label = label
        self.every = max(1, int(every))
        self.indent = indent
        self.started = time.time()
        self.done = 0

    def tick(self, note: str = "") -> None:
        self.done += 1
        if self.done % self.every and self.done != self.total:
            return
        elapsed = time.time() - self.started
        rate = elapsed / self.done
        remaining = rate * (self.total - self.done)
        log(f"{self.label} {self.done}/{self.total} · {duration(elapsed)} elapsed · "
            f"eta {duration(remaining)} · {rate:.1f}s/item{(' · ' + note) if note else ''}",
            indent=self.indent)

    def finish(self) -> float:
        elapsed = time.time() - self.started
        log(f"{self.label} complete: {self.total} in {duration(elapsed)} "
            f"({elapsed / self.total:.1f}s/item)", indent=self.indent)
        return elapsed

class Budget:
    """The wall-clock allowance, and whether the next item fits in what is left

    These scripts exit cleanly on a budget rather than being killed, so the
    budget is a thing the loop consults; making it an object means the amount
    left gets *reported* at every step instead of only being tested at the top
    of the loop.
    """

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.started = time.time()

    @property
    def spent(self) -> float:
        return time.time() - self.started

    @property
    def left(self) -> float:
        return self.seconds - self.spent

    def fits(self, estimate: float) -> bool:
        """Whether an item expected to take `estimate` seconds still fits"""
        return self.left > estimate

    def state(self) -> str:
        return f"budget {duration(self.spent)}/{duration(self.seconds)} used, {duration(self.left)} left"

    def stop_line(self, estimate: float, item: str = "next item") -> str:
        """The line to log when the next item does not fit -- the one the pipeline re-invokes on"""
        return f"{BUDGET_MARKER} {self.state()}, {item} needs ~{duration(estimate)}"
