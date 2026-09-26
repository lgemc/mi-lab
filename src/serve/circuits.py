"""A model held once, and whatever circuits the mounted folder happens to contain.

The model is the deployment; the circuits are data. That split is the whole
point of this module and it was not true before: every mask under the results
root was read onto the GPU in `__init__`, so a circuit written after the pod
started did not exist until someone restarted it, and the startup probe waited
out a minute or two of unpacking masks nobody had asked for. Rebuilding an
image to publish a circuit is rebuilding an image to publish a *file*.

So discovery is split in two. Scanning reads only each folder's own
`sheaf-<task>.json` -- small, cheap, and enough to say what the circuit is,
what model it was pruned against and how dense it is. Loading happens the
first time a request names the circuit, through the backbone that claimed the
folder (`backbones.py`), and the root is rescanned whenever a name is not
recognised, so a folder that appears while the process is running is served on
the next request that asks for it.

Three consequences worth stating because each was a failure mode:

  a new folder needs no restart      it is scanned at request time
  a foreign circuit says so          a mask pruned against another config is
                                     listed with the reason it cannot run,
                                     rather than silently missing
  an unclaimed folder is visible     `/circuits` reports what it skipped

Residency is capped. A dozen 1.7B weight masks at 170 MB of packed bits each
is 2 GB of circuits beside a 7 GB model on a time-sliced GPU, so the least
recently used is dropped once the cap is passed. Evicting is free: the folder
is still there and the next request that wants it reads it again.

A common pipe could be: scan | detect | load on demand | applied | generate
"""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from ..data.tasks import frame_for, framed_tasks, task_names
from ..methods.sheaves.gateable import gateable
from .backbones import Backbone, BackboneError, Host, Spec, backbone_for, detect

FULL = "full"

#: Circuits kept on the device at once, past which the least recently used is dropped.
MAX_RESIDENT = 8


class ServeError(Exception):
    pass


class Circuits:
    """The model, its clean weights, and every circuit the root holds *right now*"""

    def __init__(self, adapter, root: Optional[Path] = None, tasks: Optional[Sequence[str]] = None,
                 config: Optional[str] = None, max_resident: int = MAX_RESIDENT) -> None:
        self.adapter = adapter
        # None means "whatever the folder holds". One process serves one model,
        # and a model does more than one task -- pinning the server to a single
        # one meant an IOI circuit and a translation circuit on the same
        # checkpoint needed two deployments of the same 7 GiB of weights.
        unknown = sorted(set(tasks or ()) - set(task_names()))
        if unknown:
            raise ServeError(
                f"unknown task(s) {unknown}; this build knows {task_names()}"
            )
        self.wanted: Optional[List[str]] = sorted(tasks) if tasks else None
        self.tasks: List[str] = list(self.wanted or ())
        # `root=None` is the plain causal-generation server: a checkpoint served
        # for its own answers, with no circuits over it. It is a mode rather than
        # an empty directory because the difference is not cosmetic -- see the
        # clean-weight copy below.
        self.root = Path(root) if root is not None else None
        self.config = config or getattr(adapter.cfg, "id", None)
        self.max_resident = max_resident
        # Re-entrant: `generate` holds it across a whole request and calls
        # `get`, which takes it again to load and evict without a second
        # request pulling a circuit out from under an applied one.
        self.lock = threading.RLock()
        # Applying a mask overwrites the weights, so the clean ones are kept
        # beside the model to copy back after each request -- doubling resident
        # memory. With no circuits nothing ever overwrites anything, so that copy
        # is pure cost, and on a GPU shared with four other things it is the
        # difference between fitting and not.
        if self.root is None:
            targets, originals = {}, {}
        else:
            targets = gateable(adapter)
            with torch.no_grad():
                originals = {name: p.detach().clone() for name, p in targets.items()}
        self.host = Host(adapter=adapter, targets=targets, originals=originals)
        self.specs: Dict[str, Spec] = {}
        self.resident: Dict[str, Backbone] = {}
        # Monotonic use counter per name, which is the whole LRU: a clock, not
        # a timestamp, because two requests inside one millisecond otherwise
        # tie and the eviction order stops being defined.
        self._clock = 0
        self._used: Dict[str, int] = {}
        self.skipped: List[str] = []
        self.scan()

    # -- discovery ---------------------------------------------------------

    def candidates(self) -> List[Tuple[Path, str]]:
        """Every (directory, task) under the root that might hold a circuit

        Three patterns, because the two kinds of run leave different files: a
        weight run always writes its mask and may predate the artifact, and an
        edge run writes only the artifact -- a file of 1.4e9 ones is not a
        circuit and is not written.

        The task comes out of the filename and is then checked against the task
        registry, which is what keeps `sheaf-translation-inference.json` from
        being read as a task called `translation-inference`. A file naming a
        task this build does not know is skipped rather than guessed at.
        """
        if self.root is None:
            return []
        known = set(task_names())
        found = set()
        for pattern, trim in (("sheaf-*.json", len(".json")),
                              ("sheaf-*-mask.pt", len("-mask.pt")),
                              ("sheaf-*-gates.pt", len("-gates.pt"))):
            for path in self.root.rglob(pattern):
                task = path.name[len("sheaf-"):-trim]
                if task not in known:
                    continue
                if self.wanted is not None and task not in self.wanted:
                    continue
                found.add((path.parent, task))
        return sorted(found)

    def scan(self) -> Dict[str, Spec]:
        """Re-read the root. Cheap: small JSON reads, no tensors, no device traffic"""
        with self.lock:
            specs: Dict[str, Spec] = {}
            skipped: List[str] = []
            claimed: Dict[str, List[Spec]] = {}
            for directory, task in self.candidates():
                spec = detect(directory, task, config=self.config)
                name = (str(directory.relative_to(self.root))
                        if directory != self.root else directory.name)
                if spec is None:
                    skipped.append(f"{name} [{task}]")
                    continue
                spec.name = name
                claimed.setdefault(name, []).append(spec)
            for name, found in claimed.items():
                # One directory usually holds one task, and then the directory
                # name is the circuit's name. Where it holds two, both get the
                # task appended rather than one silently winning -- and only
                # those two, so an unrelated circuit's name never moves because
                # a second task appeared somewhere else under the root.
                for spec in found:
                    spec.name = name if len(found) == 1 else f"{name}:{spec.task}"
                    specs[spec.name] = spec
            self.specs = specs
            self.skipped = skipped
            self.tasks = sorted({spec.task for spec in specs.values()}) or list(self.wanted or ())
            # A circuit whose folder went away should not stay on the GPU.
            for gone in [name for name in self.resident if name not in specs]:
                self.resident.pop(gone, None)
                self._used.pop(gone, None)
            return specs

    def names(self) -> List[str]:
        """`full` plus every circuit that can actually be run here"""
        return [FULL, *sorted(name for name, spec in self.specs.items() if not spec.problem)]

    def describe(self) -> List[Dict[str, object]]:
        """The listing: every folder found, resident or not, runnable or not"""
        out = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            loaded = self.resident.get(name)
            out.append(loaded.describe() if loaded else {**spec.describe(), "resident": False})
        return out

    # -- loading -----------------------------------------------------------

    def _evict(self) -> None:
        while len(self.resident) > self.max_resident:
            oldest = min(self.resident, key=lambda name: self._used.get(name, 0))
            self.resident.pop(oldest, None)
            self._used.pop(oldest, None)

    def get(self, name: str) -> Optional[Backbone]:
        """The named circuit, loaded if it is not already. `full` is None, not an error"""
        if name == FULL:
            return None
        with self.lock:
            if name not in self.specs:
                # The folder may have been written after the last scan; that is
                # the case this whole design exists for, so look before failing.
                self.scan()
            if name not in self.specs:
                raise ServeError(
                    f"no circuit '{name}' under {self.root}; available: {self.names()}"
                    + (f" (skipped, nothing claims them: {self.skipped})" if self.skipped else "")
                )
            spec = self.specs[name]
            if spec.problem:
                raise ServeError(f"circuit '{name}' cannot be run here: {spec.problem}")
            self._clock += 1
            self._used[name] = self._clock
            if name in self.resident:
                return self.resident[name]
            loaded = backbone_for(spec)(self.host, spec)
            try:
                loaded.load()
            except BackboneError as error:
                raise ServeError(str(error)) from None
            self.resident[name] = loaded
            self._evict()
            return loaded

    @contextmanager
    def loaded(self, name: str) -> Iterator[None]:
        """The named circuit put into the model for the duration; `full` is a no-op"""
        circuit = self.get(name)
        if circuit is None:
            yield
            return
        with circuit.applied():
            yield

    # -- inference ---------------------------------------------------------

    def generate(self, prompts: Sequence[str], circuits: Sequence[str],
                 max_new_tokens: int) -> Dict[str, List[str]]:
        """Each prompt continued under each named circuit, {circuit: [completion per prompt]}"""
        if not prompts:
            raise ServeError("no prompts")
        out: Dict[str, List[str]] = {}
        with self.lock:
            # Resolved before any of them runs, so a request naming one good
            # circuit and one bad one fails without having generated half of it.
            for name in circuits:
                self.get(name)
            for name in circuits:
                with self.loaded(name):
                    out[name] = self.adapter.generate(list(prompts), max_new_tokens=max_new_tokens)
        return out

    def infer(self, inputs: Sequence[str], circuits: Sequence[str], max_new_tokens: int,
              task: Optional[str] = None, trim: Optional[bool] = None) -> Dict[str, object]:
        """One door for every task: the task names the frame, the inputs fill it

        There was an endpoint per task, which meant every new task was a new
        route re-implementing the same three lines around a different template.
        The task is a parameter now, and what a task *is* stays in
        `data/tasks.py` where the rest of it lives -- so serving a new one is a
        `@register_frame` there and nothing here.

        `task` omitted means the inputs are already prompts, which is what the
        tasks with no single-slot frame need: an IOI prompt is a whole
        sentence, and `greater_than` wants a noun and a year, so neither has a
        one-input form to guess at.

        The prompts are returned alongside the outputs. The prompt is half of
        what a circuit answers and a caller who cannot see it cannot tell a bad
        circuit from a badly framed question.
        """
        if not inputs:
            raise ServeError("no inputs")
        if task is not None and task not in task_names():
            raise ServeError(f"unknown task '{task}'; this build knows {task_names()}")
        frame = frame_for(task) if task else None
        if task and frame is None:
            raise ServeError(
                f"task '{task}' has no one-input frame -- its prompts are not a template with a "
                f"single hole. Send whole prompts with no `task`, or use one of {framed_tasks()}"
            )
        # A circuit answers the frame it was pruned under. Framing a
        # translation circuit's word as an IOI sentence, or the reverse, asks it
        # something it never saw and reads as a broken circuit.
        wrong = [name for name in circuits
                 if name != FULL and name in self.specs and self.specs[name].task != task]
        if task is not None and wrong:
            raise ServeError(
                f"{wrong} were pruned for {sorted({self.specs[n].task for n in wrong})}, not "
                f"'{task}'; drop `task` to send whole prompts instead"
            )
        prompts = [frame(text) for text in inputs] if frame else list(inputs)
        raw = self.generate(prompts, circuits, max_new_tokens)
        # A framed task continues into the next example -- the translation
        # frame ends in a newline and the model starts another pair -- so the
        # answer is the first line. Raw prompts are returned whole, because
        # nothing here knows where the caller's answer ends.
        cut = (frame is not None) if trim is None else trim
        outputs = ({name: [text.split("\n", 1)[0].strip() for text in texts]
                    for name, texts in raw.items()} if cut else raw)
        return {"task": task, "inputs": list(inputs), "prompts": prompts, "outputs": outputs}
