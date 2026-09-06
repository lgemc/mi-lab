"""Several models in one pod, resident only while somebody is using them.

The pod is the deployment, the circuits are data -- and the *weights* are
neither. A server that loads its checkpoint at startup and holds it forever
occupies the GPU for the 23 hours a day nobody asks it anything, and on a
time-sliced GPU that is not politeness, it is the whole constraint: this node
advertises four slices and all four were claimed by processes that were idle,
so the fifth model had nowhere to go.

So a model is loaded the first time a request names it and dropped again once
it has been idle for `idle_timeout`. What that buys is not memory efficiency in
the abstract; it is that two checkpoints can share one slice, because they are
almost never busy at the same moment.

Three things this has to get right, and each is a way it would be worse than
holding the model:

  never evict a live one   a sweep that frees weights out from under a running
                           generation is a crash, so `use` marks a model busy
                           for the length of a request and the sweeper skips
                           anything busy no matter how old its clock is.
  free it for real         dropping the last Python reference is not returning
                           the memory: the caching allocator keeps the blocks.
                           `unload` collects and then empties the cache, and
                           `resident_bytes` is how you check it worked rather
                           than assume. It can only drop what nothing else
                           holds -- `with pool.use(name) as circuits:` leaves
                           `circuits` bound *after* the block, which is enough
                           to pin a checkpoint on the device forever.
  say what it is doing     a first request that pays a model load is slow for a
                           reason, and a server that hides that reads as a slow
                           server. `/health` reports what is resident and what
                           is merely configured.

`Circuits` is unchanged and still owns exactly one model: the pool holds one
per name and builds it on demand. A pool of one behaves like the old server
with `preload=True`, which is what the two circuit stacks still pass.

A common pipe could be: specs | use | load on demand | sweep | unload
"""

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import torch

from ..model.adapter import load_adapter, require_circuits
from .circuits import Circuits

#: Seconds a model may sit unused before the sweeper drops it. Three minutes is
#: long enough that a person reading one answer and typing the next never pays a
#: reload, and short enough that a browser tab left open overnight does not hold
#: a GPU slice until morning.
IDLE_TIMEOUT = 180.0

#: How often the sweeper looks. A model is dropped between `IDLE_TIMEOUT` and
#: `IDLE_TIMEOUT + SWEEP_EVERY` after its last use; the slack is not worth a
#: tighter loop.
SWEEP_EVERY = 20.0


class PoolError(ValueError):
    """A pool that cannot serve what was asked of it, said with the way out"""


@dataclass(frozen=True)
class ModelSpec:
    """One served model: what to call it, what to load, and what to load over it"""

    name: str
    config: str
    circuits: Optional[Path] = None
    tasks: Optional[Sequence[str]] = None

    @staticmethod
    def parse(text: str) -> "ModelSpec":
        """`name=config@circuits` -- the deployable form, since a pod has no argv

        `config` alone is the common case and names the model after it.
        `@none`, or no `@` at all, is the plain generation server.
        """
        # The separator, not the right-hand side, is what says whether a name was
        # given: `name=` is a spec with its config left off, and reading it as a
        # bare config called `name` would load something nobody asked for.
        name, separator, rest = text.partition("=")
        if not separator:
            name, rest = "", name
        config, _, root = rest.partition("@")
        config = config.strip()
        if not config:
            raise PoolError(f"'{text}' names no config; write `config`, or `name=config@circuits`")
        root = root.strip()
        return ModelSpec(
            name=(name.strip() or config),
            config=config,
            circuits=None if root.lower() in ("", "none", "off") else Path(root),
        )


@dataclass
class _Held:
    """One loaded model and the two facts the sweeper needs about it"""

    circuits: Circuits
    used_at: float = field(default_factory=time.monotonic)
    busy: int = 0


class ModelPool:
    """Every model this server can produce, and the few that are loaded right now"""

    def __init__(self, specs: Sequence[ModelSpec], idle_timeout: float = IDLE_TIMEOUT,
                 max_resident_circuits: int = 8,
                 build: Optional[Callable[[ModelSpec], Circuits]] = None) -> None:
        if not specs:
            raise PoolError("a pool needs at least one model spec")
        duplicates = sorted({s.name for s in specs if [x.name for x in specs].count(s.name) > 1})
        if duplicates:
            raise PoolError(f"two models are both called {duplicates}; names are how a request "
                            f"picks one, so they have to differ -- use `name=config`")
        self.specs: Dict[str, ModelSpec] = {spec.name: spec for spec in specs}
        self.idle_timeout = idle_timeout
        self.max_resident_circuits = max_resident_circuits
        # Injected for the tests, which must not load a checkpoint to check that
        # eviction works.
        self._build = build or self._load
        self._held: Dict[str, _Held] = {}
        # Re-entrant and held only around bookkeeping -- never across a load or a
        # generation, or one slow request would serialize the whole server.
        self._lock = threading.RLock()
        self._loading: Dict[str, threading.Lock] = {name: threading.Lock() for name in self.specs}
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None

    # -- what is configured, and what is loaded -----------------------------

    def names(self) -> List[str]:
        return list(self.specs)

    def resident(self) -> List[str]:
        with self._lock:
            return sorted(self._held)

    def describe(self) -> List[dict]:
        """Every configured model, said the same way whether or not it is loaded"""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "name": spec.name,
                    "config": spec.config,
                    "circuits": str(spec.circuits) if spec.circuits else None,
                    "resident": spec.name in self._held,
                    "busy": self._held[spec.name].busy if spec.name in self._held else 0,
                    "idle_seconds": (round(now - self._held[spec.name].used_at, 1)
                                     if spec.name in self._held else None),
                }
                for spec in self.specs.values()
            ]

    def resident_bytes(self) -> Optional[int]:
        """Device memory actually held, so `unload` can be checked rather than trusted"""
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else None

    # -- loading ------------------------------------------------------------

    def _load(self, spec: ModelSpec) -> Circuits:
        adapter = require_circuits(load_adapter(spec.config))
        return Circuits(adapter, spec.circuits, tasks=spec.tasks, config=spec.config,
                        max_resident=self.max_resident_circuits)

    def peek(self, name: str) -> Optional[Circuits]:
        """The named model if it is already loaded, **without touching its clock**

        `/health` is the liveness probe and runs every ten seconds forever. If
        reading it counted as using the model, the default model's idle timer
        would be reset before it could ever expire and that model would sit on
        the device for the life of the pod -- which is exactly the state this
        module exists to end. So looking is not using.
        """
        with self._lock:
            held = self._held.get(name)
            return held.circuits if held is not None else None

    def get(self, name: str) -> Circuits:
        """The named model, loading it if it is not resident. Does not mark it busy.

        Prefer `use`, which does. This is here for a caller that only wants to
        read what a model *is* -- listing its circuits, say -- and is willing to
        race a sweep it did not hold off.
        """
        if name not in self.specs:
            raise PoolError(f"unknown model '{name}'; this server serves {self.names()}")
        with self._lock:
            held = self._held.get(name)
            if held is not None:
                held.used_at = time.monotonic()
                return held.circuits
        # Outside the pool lock: a load is tens of seconds and holding the lock
        # across it would block `/health` and every other model's requests. The
        # per-name lock is what keeps two simultaneous first requests from
        # loading the same checkpoint twice.
        with self._loading[name]:
            with self._lock:
                held = self._held.get(name)
                if held is not None:
                    held.used_at = time.monotonic()
                    return held.circuits
            circuits = self._build(self.specs[name])
            with self._lock:
                self._held[name] = _Held(circuits)
            return circuits

    @contextmanager
    def use(self, name: str) -> Iterator[Circuits]:
        """The named model, held against eviction for the length of the block"""
        circuits = self.get(name)
        with self._lock:
            held = self._held.get(name)
            if held is None:  # swept between the load and here; take it again
                held = self._held.setdefault(name, _Held(circuits))
            held.busy += 1
        try:
            yield held.circuits
        finally:
            with self._lock:
                held.busy -= 1
                held.used_at = time.monotonic()

    # -- unloading ----------------------------------------------------------

    def unload(self, name: str, force: bool = False) -> bool:
        """Drop a model's weights. Refuses a busy one unless forced; returns whether it went"""
        with self._lock:
            held = self._held.get(name)
            if held is None:
                return False
            if held.busy and not force:
                return False
            del self._held[name]
        # Outside the lock, and in this order. Dropping the reference is not
        # returning the memory -- torch's caching allocator holds the blocks
        # until it is told to let go, so a pool that only deleted would free
        # nothing a second model could use, which is the entire point here.
        del held
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def sweep(self) -> List[str]:
        """Drop every model idle past the timeout. Returns what went"""
        now = time.monotonic()
        with self._lock:
            stale = [name for name, held in self._held.items()
                     if not held.busy and now - held.used_at >= self.idle_timeout]
        return [name for name in stale if self.unload(name)]

    # -- the sweeper thread -------------------------------------------------

    def start(self, on_unload: Optional[Callable[[List[str]], None]] = None) -> None:
        """Begin sweeping in the background. Idempotent"""
        if self._sweeper is not None or self.idle_timeout <= 0:
            return

        def loop():
            while not self._stop.wait(SWEEP_EVERY):
                dropped = self.sweep()
                if dropped and on_unload:
                    on_unload(dropped)

        self._sweeper = threading.Thread(target=loop, name="model-pool-sweeper", daemon=True)
        self._sweeper.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sweeper is not None:
            self._sweeper.join(timeout=SWEEP_EVERY + 5)
            self._sweeper = None
