"""What a long GPU run has to say about itself while it is still running.

Every script in this phase is a chain of multi-minute passes over an 8B model:
a means capture, a baseline translation, one ablated translation per component.
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

`preview` is the observability that matters most here and is the least like
logging: it prints actual generations. A corpus BLEU cannot distinguish "this
translates badly" from "this has stopped emitting language", those are
entirely different findings, and the retracted run of 2026-08-31 reported a
32x threshold pass on a model whose output was `a a a a a`. Anything that
ablates prints samples of what came out, at every arm, with no flag to turn it
off.

GPU memory is read through torch rather than nvidia-smi: on the GB10 the smi
query returns N/A for used and total, and unified memory makes the number
worth watching anyway.

A common pipe could be: banner | step | Progress.tick | preview | log
"""

import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

_LOG_FILE: Optional[Path] = None

PREVIEW_SAMPLES = 3
PREVIEW_WIDTH = 96

DEGENERACY_TYPES = 0.30   # unique tokens / total tokens below this is repetition, not translation
DEGENERACY_SHARE = 0.50   # or one token taking this much of the output on its own
DEGENERACY_DUPLICATE = 0.02  # or the identical output on this share of the corpus

def set_log_file(path) -> None:
    """Mirror everything logged from here on into `path`, appending

    Appending rather than truncating because these scripts are resumed: a run
    is a chain of invocations against one budget, and a log that kept only the
    last of them would lose the part where the interesting thing happened.
    """
    global _LOG_FILE
    _LOG_FILE = Path(path)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(sys.argv)}\n")

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

def degeneracy(hypotheses: Sequence[str]) -> float:
    """The fraction of outputs that have stopped being language

    Deliberately crude and deliberately automatic. The checklist item it
    implements is "read ten generations at the largest ablation and stop if
    they are not language"; a human still should, but a sweep that runs
    unattended needs the machine to raise its hand. It lives here rather than
    beside any one experiment because every arm of every ablation wants it,
    and a second copy would be a second threshold to disagree with the first.
    """
    if not hypotheses:
        return 0.0
    # Collapse is not always visible inside one hypothesis. A model ablated past
    # the frontier answers 'the' to all 200 sources: each one is three tokens or
    # fewer, so the repetition rules below never examine it, and the corpus
    # scored 8.5% broken while BLEU said 0.01. The signal is only there across
    # hypotheses -- 200 distinct news sentences do not share a translation --
    # so the same text landing on a share of the corpus is counted as collapse
    # whatever its length. The floor of 3 keeps a short list from making any
    # coincidence fatal, and the healthy arms of this project peak at 2.
    repeats = Counter(normalized for text in hypotheses if (normalized := " ".join(text.split()).casefold()))
    duplicate_limit = max(3, round(DEGENERACY_DUPLICATE * len(hypotheses)))
    broken = 0
    for text in hypotheses:
        tokens = text.split()
        # an empty generation is the most degenerate outcome there is, and an
        # early version scored it 0.0 by falling through the too-short guard
        # below -- a matched random ablation that silenced the model entirely
        # was reported as perfectly healthy
        if not tokens:
            broken += 1
            continue
        # punctuation with no letters in it is not a translation at any length:
        # '...', '(', '(   (   (' all scored clean under a rule that only looked
        # for repetition among four or more tokens
        if not any(character.isalpha() for character in text):
            broken += 1
            continue
        if repeats[" ".join(tokens).casefold()] >= duplicate_limit:
            broken += 1
            continue
        if len(tokens) < 4:
            continue
        counts = Counter(tokens)
        if (len(counts) / len(tokens) < DEGENERACY_TYPES
                or counts.most_common(1)[0][1] / len(tokens) > DEGENERACY_SHARE):
            broken += 1
    return round(broken / len(hypotheses), 3)

def preview(hypotheses: Sequence[str], label: str = "sample", count: int = PREVIEW_SAMPLES,
            indent: int = 1) -> None:
    """Print a few actual generations, because a metric cannot show you a broken model

    Not optional and not behind a flag. The checklist item is "read ten
    generations at the largest ablation and stop if they are not language",
    and the run that skipped it reported a threshold pass by a factor of 32 on
    a model emitting a single repeated token.
    """
    if not hypotheses:
        log(f"{label}: no generations at all -- the pass produced nothing", indent=indent)
        return
    for index, text in enumerate(hypotheses[:count]):
        shown = text if len(text) <= PREVIEW_WIDTH else text[:PREVIEW_WIDTH - 1] + "…"
        log(f"{label}[{index}]: {shown!r}", indent=indent)
    empty = sum(1 for text in hypotheses if not text.strip())
    if empty:
        log(f"{label}: {empty}/{len(hypotheses)} generations are empty", indent=indent)
    broken = degeneracy(hypotheses)
    if broken:
        log(f"{label}: DEGENERACY {broken:.1%} of generations are repeated tokens, not language -- "
            "this is a broken model, not an ablated one", indent=indent)
