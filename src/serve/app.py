"""The HTTP face of `Circuits`: a health check, the circuit list, and one inference route.

Three routes and a page. `GET /health` says what is served; `GET /circuits`
**rescans the mounted folder** and lists everything under it with what its run
recorded; `POST /infer` runs prompts under one or more circuits, framing them
for a task when asked. `GET /` is a single page that calls all three, so a
browser is enough to put a circuit and the full model side by side.

**One inference route, not one per task.** `/translate` and `/generate` were
the same three lines around a different template, and every new task meant
another route. `task` is a parameter now and the frame lives in
`data/tasks.py` with the rest of what a task is, so serving a new one is a
`@register_frame` there and nothing here. Omitting `task` sends whole prompts,
which is what the tasks with no single-slot frame need.

The rescan is the point of the route, not a detail. The circuits are a
read-only mount and the model is the deployment, so publishing a circuit is
writing a folder -- and a listing that answered from a snapshot taken at
process start would report that the folder is not there. `POST /generate`
rescans too, by way of `Circuits.get`, so a caller who already knows the name
never has to list first.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..core.config import load_config
from ..data.tasks import frame_for, framed_tasks
from .circuits import FULL, Circuits, ServeError
from .examples import Examples
from .models import ModelPool, ModelSpec, PoolError

MAX_PROMPTS = 64
#: Ceiling on a request's token budget, not a default. 64 was enough while every
#: served model answered a frame in one word; a served instruct model puts its
#: answer after a <think> block whose median on ToolAlpaca is a few hundred
#: words, so that ceiling truncated every answer inside its own reasoning.
MAX_NEW_TOKENS = 1024

PAGE = (Path(__file__).parent / "page.html").read_text()


class InferRequest(BaseModel):
    """Inputs, and optionally the task whose frame turns them into prompts"""

    inputs: List[str] = Field(..., min_length=1, max_length=MAX_PROMPTS)
    circuits: List[str] = Field(default_factory=lambda: [FULL])
    max_new_tokens: int = Field(16, ge=1, le=MAX_NEW_TOKENS)
    # None means the inputs are already prompts. Named rather than inferred
    # from the circuits: two circuits for different tasks in one request is a
    # mistake worth an error, not a frame to pick by majority.
    task: Optional[str] = None
    # None keeps the first line for a framed task (the frame runs on into the
    # next example) and the whole continuation for raw prompts.
    trim: Optional[bool] = None
    # None means the server's default model. Named per request rather than per
    # deployment because the pool exists precisely so one pod can answer for
    # more than one checkpoint.
    model: Optional[str] = None


def create_app(circuits: Optional[Circuits] = None, config: Optional[str] = None,
               examples: Optional[Examples] = None,
               pool: Optional[ModelPool] = None) -> FastAPI:
    """The HTTP app over either one resident model or a pool that swaps them.

    `circuits=` is the single-model form the circuit stacks use and is kept
    exactly as it was: one checkpoint, loaded before the app exists, never
    dropped. Passing `pool=` instead makes the model a request parameter and
    lets it be unloaded when idle -- which is the only reason two checkpoints
    fit on one time-sliced GPU slice.
    """
    if (circuits is None) == (pool is None):
        raise ValueError("create_app takes exactly one of `circuits` or `pool`")
    app = FastAPI(title="mi-lab circuits", version="0.1")
    examples = examples or Examples()

    if pool is None:
        # A single resident model, expressed as a pool of one so that everything
        # below has one shape. `idle_timeout=0` is what makes it never unload.
        pool = ModelPool([ModelSpec(name=config or circuits.adapter.cfg.id,
                                    config=config or circuits.adapter.cfg.id,
                                    circuits=circuits.root)],
                         idle_timeout=0, build=lambda spec: circuits)
        pool.get(pool.names()[0])
    default_model = pool.names()[0]

    def pick(name: Optional[str]) -> str:
        """Resolve the requested model, refusing an unknown one as a bad request

        Checked here rather than left to `pool.get`, so that a typo'd model name
        is a 400 naming the models that exist instead of a 500 from three frames
        down -- and, more to the point, so `/health` cannot be made to fail its
        own probe by a query string.
        """
        chosen = name or default_model
        if chosen not in pool.specs:
            raise HTTPException(
                status_code=400,
                detail=f"unknown model '{chosen}'; this server serves {pool.names()}")
        return chosen


    @app.get("/health")
    def health(model: Optional[str] = None):
        """What is served, answered **without loading anything**.

        This is the liveness and startup probe: it runs every ten seconds
        forever, so it must not walk a mounted filesystem and -- now that a
        model can be absent -- it must not pull one onto the device either. A
        probe that loads a checkpoint would restart the pod every time the
        sweeper had just dropped one.

        So the facts that come from the *config* (the token budget, whether the
        checkpoint wants a chat template) are read from `configs/`, which costs
        no torch; the facts that only a loaded model has (its device, its
        circuits) are reported for the resident ones and left null otherwise.
        """
        name = pick(model)
        spec = pool.specs[name]
        settings = load_config(spec.config)
        # `peek`, not `get`: a probe that counted as a use would keep the
        # default model's idle timer permanently reset, and a probe that loaded
        # would pull a swept model straight back onto the device.
        loaded = pool.peek(name)
        return {"status": "ok",
                "model": name,
                "config": spec.config,
                # Every model this pod can answer for, and which are in memory
                # right now. A first request that pays a load is slow for a
                # reason, and this is where a caller can see the reason.
                "models": pool.describe(),
                "resident_models": pool.resident(),
                "idle_timeout": pool.idle_timeout,
                # From the config, so this is right whether or not it is loaded.
                "max_new_tokens": min(settings.max_new_tokens, MAX_NEW_TOKENS),
                "chat": bool(getattr(settings, "chat", False)),
                "tasks": loaded.tasks if loaded else [],
                "device": str(loaded.adapter.model.device) if loaded else None,
                "root": (str(spec.circuits) if spec.circuits else None),
                "resident": sorted(loaded.resident) if loaded else [],
                # the frame each task was pruned under, so the page can show a
                # caller exactly what the circuit is given -- the prompt is
                # half of what a circuit answers, and hiding it hides that.
                # Keyed by task and built by running the registered builder on
                # a marker, because one hard-coded template served the
                # translation stack a frame and the IOI stack a lie: the page
                # then framed `Then, John and Mary...` as a Spanish word and
                # every request 400'd. A frame is the prompt up to where the
                # answer goes, so the answer slot is appended to it.
                "frames": {task: frame_for(task)("{input}") + "{answer}"
                           for task in framed_tasks()},
                # which tasks a bare input can be framed for; everything else
                # takes whole prompts with no `task`
                "framed_tasks": framed_tasks(),
                "circuits": loaded.names() if loaded else [FULL]}

    @app.get("/circuits")
    def list_circuits(model: Optional[str] = None):
        """Rescan the mounted folder. **This one loads the model** if it is not resident.

        Unlike `/health`, which a probe calls: a caller asking what circuits a
        model has is about to run one, so paying the load here is paying it a
        moment early rather than gratuitously.
        """
        name = pick(model)
        circuits = pool.get(name)
        circuits.scan()
        return {"model": name,
                "full": {"name": FULL, "density": 1.0, "kind": "full", "resident": True},
                "root": str(circuits.root) if circuits.root else None,
                "circuits": circuits.describe(),
                # folders under the root that no backbone claimed. Reported
                # rather than dropped: a circuit that is silently absent is
                # indistinguishable from one that was never written.
                "skipped": circuits.skipped}

    @app.get("/models")
    def list_models():
        """Every checkpoint this pod can answer for, and which are in memory"""
        return {"default": default_model, "idle_timeout": pool.idle_timeout,
                "models": pool.describe()}

    @app.get("/examples")
    def list_examples():
        """Held-out prompts, ready to POST to /infer

        Re-read per request for the same reason `/circuits` rescans: the files
        are data under a mount, and a listing answered from a snapshot taken at
        process start reports that a file written since is not there.
        """
        return examples.load()

    @app.post("/infer")
    def infer(request: InferRequest):
        name = pick(request.model)
        try:
            # `use` rather than `get`: it marks the model busy for the length of
            # the generation, so the sweeper cannot free the weights out from
            # under a request that is already running.
            with pool.use(name) as circuits:
                result = circuits.infer(request.inputs, request.circuits, request.max_new_tokens,
                                        task=request.task, trim=request.trim)
            return {"model": name, **result}
        except ServeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        except PoolError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    return app
