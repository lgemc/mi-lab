from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ..core.readout import Score
from ..plugins import load_domains

"""
A circuit is found *on a task*, and the moment there is more than one task the
task itself has to become data.

Everything downstream of here -- attribution, patching, the gradient
approximation of patching, every metric read off them -- needs exactly five
things from a task: a batch of clean inputs, a corrupted twin for each, the
rule that turns what the model produced into one number, the positions those
inputs share, and a way to take a subset. That list is CircuitTask, and it is a
Protocol rather than a base class because IOIDataset already satisfied it
before this module existed.

The third of those was a pair of token ids until the measurement contract, and
is a `Score` now. The change is what lets the modules that measure stop being
about language: they ask the task how a number comes out and never index a
vocabulary themselves.

What is here is the *shape* of a task and the registry of them. The tasks
themselves are domain work -- a frame is written in a language, a pool is
filtered by a tokenizer, an answer is a token id -- and live in
`domains/*/tasks.py`, which registers into the tables below. So this module
knows that tasks exist and how one is named, and knows nothing about any of
them; `build_task` loads the domains before it looks, which is the one door
described in `src/plugins.py`.

Why more than one task at all: a circuit measured only on its own task is not
shown to be *about* that task. Ablating one task's circuit damages unrelated
tasks about as much as its own, because circuits at this granularity are
dominated by shared machinery, and the only way to see that is to have a
second task to ablate against.

A common pipe could be: build_task | patch_heads | discover | specificity
"""

class TaskError(ValueError):
    """Raised when a task cannot be built honestly: prompts that do not line up, pools the tokenizer splits"""

@runtime_checkable
class CircuitTask(Protocol):
    """What every circuit measurement needs a task to be, and nothing more

    `clean` and `corrupted` are parallel sequences of the same length, one twin
    per example, and their element type is whatever this task's inputs are --
    strings for a decoder, and the measurement layer never looks inside one.

    `readout` is the third thing, and it is the one that used to be missing.
    Attribution, patching, the gradient approximations of patching and every
    metric read off them need exactly a batch of inputs, a set of addressable
    sites, and one scalar per example; the scalar was a logit difference over
    two token ids, hardcoded in `methods/common/span.py`. It is the task's
    opinion about this model for the same reason the answer ids were --
    "the ids are the tokenizer's opinion and a task carrying them would be
    silently wrong on the next model" -- and the generalization is one step
    further: the whole scoring rule is that opinion, not just the ids in it.
    """

    name: str
    #: What kind of thing `clean` holds, for the `.mia` card. "text" here.
    modality: str

    @property
    def clean(self) -> Sequence[Any]:
        """One input per example, all of one length at the sites patching addresses"""

    @property
    def corrupted(self) -> Sequence[Any]:
        """The twin of each clean input, differing in the one thing the task is about"""

    def readout(self, adapter) -> Score:
        """How a number comes out of this model on this task, bound to these examples

        Built against the adapter in hand for the reason `answers` was: the
        answer ids, the class embedding, the success predicate are all this
        model's opinion, and a task carrying one is silently wrong on the next.
        """

    def labels(self, adapter) -> List[str]:
        """The first example's positions as strings, for labelling a position axis"""

    def landmarks(self, adapter) -> Dict[str, int]:
        """The positions worth reading a patching grid at, by name"""

    def subset(self, indices: Sequence[int]) -> "CircuitTask":
        """The same task over a chosen few of its examples

        Consistency is measured by finding a circuit per example and asking
        which components recur, so a task that cannot be cut down to one
        example cannot answer that question at all.
        """

    def __len__(self) -> int:
        """How many clean/corrupted pairs there are"""

# ----------------------------------------------------------------- the registry

@dataclass(frozen=True)
class Recipe:
    """How to build one task, and one line saying what the model is being asked"""
    build: Callable[..., CircuitTask]
    description: str

TASKS: Dict[str, Recipe] = {}

def register_task(name: str, description: str) -> Callable:
    """Register a builder under a task name, so a task can be named as data

    Adding a task is a registration and a pool, never an edit to the code that
    measures one. That is the same reason backends and experiment kinds are
    registries: the comparison in methods/circuits/comparison.py sweeps what is in
    here, so a new task joins every cross-task number by existing.
    """
    def decorate(build: Callable[..., CircuitTask]) -> Callable[..., CircuitTask]:
        TASKS[name] = Recipe(build=build, description=description)
        return build
    return decorate

#: How a caller's own input becomes a prompt for a task, where that is possible.
#:
#: A task's `frame` is a template, and the templates do not agree on how many
#: holes they have: translation's takes one word, `greater_than` takes a noun
#: *and* a year, `agreement` a subject *and* an attractor, and IOI's prompts are
#: whole sentences with no single free slot at all. So this maps only the tasks
#: with exactly one thing a caller could reasonably supply, and everything else
#: is served raw prompts rather than given a guessed second argument. A missing
#: entry is not an omission; it is the honest answer that there is no one-input
#: form of that task.
FRAMES: Dict[str, Callable[[str], str]] = {}

def register_frame(name: str) -> Callable:
    """Register the one-input prompt form of a task, for callers that have an input not a prompt"""
    def wrap(builder: Callable[[str], str]) -> Callable[[str], str]:
        FRAMES[name] = builder
        return builder
    return wrap

def frame_for(name: str) -> Optional[Callable[[str], str]]:
    """The task's one-input prompt form, or None where it has none"""
    load_domains()
    return FRAMES.get(name)

def framed_tasks() -> List[str]:
    """Every task a bare input can be turned into a prompt for"""
    load_domains()
    return sorted(FRAMES)

def task_names() -> List[str]:
    """Every task this repository knows how to build, sorted"""
    load_domains()
    return sorted(TASKS)

#: task name -> {option: the values its builder accepts}.
#:
#: A spec names a task and the options it wants built -- IOI's corruption and
#: frame are the two that exist -- and a spec naming an impossible one should
#: fail before the model loads rather than three frames into the build. What is
#: allowed is a fact about the task, so it is registered with the task, and the
#: kernel checks against the table without knowing what a corruption is.
OPTIONS: Dict[str, Dict[str, List[Any]]] = {}

def register_options(name: str, **allowed: Sequence[Any]) -> None:
    """Declare the values a task's options may take, so a spec can be checked against them"""
    OPTIONS.setdefault(name, {}).update({option: list(values) for option, values in allowed.items()})

def check_options(name: str, **given: Any) -> None:
    """Refuse an option value the named task cannot build, listing the ones it can

    Silent about an option nothing registered: a builder is free to take
    keywords whose range it does not enumerate, and refusing those here would
    make this table a second, incomplete copy of every builder's signature.
    """
    load_domains()
    allowed = OPTIONS.get(name, {})
    for option, value in given.items():
        if option in allowed and value not in allowed[option]:
            raise TaskError(
                f"unknown {name}.{option} '{value}'; '{name}' accepts {sorted(allowed[option], key=str)}"
            )

def build_task(name: str, adapter, size: int = 16, seed: int = 0, **options) -> CircuitTask:
    """Build a named task for the model in hand"""
    load_domains()
    if name not in TASKS:
        raise TaskError(f"unknown task '{name}'; known tasks are {task_names()}")
    if size < 1:
        raise TaskError(f"a task needs at least one example, got size={size}")
    return TASKS[name].build(adapter, size=size, seed=seed, **options)

