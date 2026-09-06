"""Serve a model and whatever circuits the mounted folder holds, asked from anywhere.

Loads one config once and scans a results root for self-describing circuit
folders, then answers `/translate` and `/generate` under the full model or any
of them (see `src/serve`). This is the deployable form of
`scripts.sheaf_infer`: the same load, kept resident.

The model is the deployment and the circuits are data. Scanning reads each
folder's own `sheaf-<task>.json` and nothing else, the circuit itself is read
the first time a request names it, and the root is rescanned on `/circuits`
and on any name the server does not recognise -- so publishing a circuit is
writing a folder under the mount, not rebuilding an image. Which *kind* of
circuit a folder holds is the folder's business too: a weight mask is
multiplied into the parameters, an edge set is applied as hooks around the
forward pass, and `src/serve/backbones.py` picks between them by reading the
artifact. A third kind is one class and one decorator there.

Run: uv run python -m scripts.serve qwen3-1.7b --circuits results/qwen3-1.7b-sweep
     curl -s localhost:8000/circuits | python -m json.tool
     curl -s localhost:8000/translate -H 'content-type: application/json' \\
          -d '{"words": ["perro", "casa"], "circuits": ["full", "gold-t02"]}'

**More than one model in one pod.** `--models` takes a comma-separated list of
`name=config@circuits`, and then a checkpoint is loaded the first time a request
names it and dropped again once it has been idle for `--idle-timeout` seconds.
That is not a memory nicety: this node time-slices one GPU into four claims, all
four were held by servers that were idle, and a fifth model had nowhere to go.
Two checkpoints that are never busy at the same moment can share one claim.

`--config`/`--circuits` remain the single-model form and behave exactly as they
did -- loaded before the server answers anything, never dropped.

Environment (for the container, where there is no command line to speak of):
MI_LAB_CONFIG, MI_LAB_CIRCUITS, MI_LAB_MODELS, MI_LAB_IDLE_TIMEOUT,
MI_LAB_TASKS, MI_LAB_MAX_RESIDENT, MI_LAB_EXAMPLES, PORT.
"""

import argparse
import os
from pathlib import Path

import uvicorn

from src.model.adapter import load_adapter, require_circuits
from src.serve.app import create_app
from src.serve.circuits import Circuits
from src.serve.examples import Examples
from src.serve.models import IDLE_TIMEOUT, ModelPool, ModelSpec
from src.telemetry.observe import banner, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default=os.environ.get("MI_LAB_CONFIG", "qwen3-1.7b"))
    parser.add_argument("--circuits", type=str,
                        default=os.environ.get("MI_LAB_CIRCUITS", "results/qwen3-1.7b-sweep"),
                        help="results root; every circuit folder below it is served, rescanned "
                             "at request time. Empty or 'none' serves the model alone -- plain "
                             "causal generation, no circuits, no clean-weight copy.")
    parser.add_argument("--max-resident", type=int, dest="max_resident",
                        default=int(os.environ.get("MI_LAB_MAX_RESIDENT", "8")),
                        help="circuits kept on the device at once; past this the least recently "
                             "used is dropped and reread when it is next asked for")
    parser.add_argument("--tasks", default=os.environ.get("MI_LAB_TASKS", os.environ.get("MI_LAB_TASK", "")),
                        help="comma-separated tasks to serve; empty (the default) serves every "
                             "task the folder holds, which is the point -- one model, one pod, "
                             "and its IOI and translation circuits side by side")
    parser.add_argument("--models", type=str, default=os.environ.get("MI_LAB_MODELS", ""),
                        help="comma-separated `name=config@circuits` specs served from one "
                             "process, each loaded on demand and dropped when idle. Overrides "
                             "--config/--circuits, which are the single always-resident model.")
    parser.add_argument("--idle-timeout", type=float, dest="idle_timeout",
                        default=float(os.environ.get("MI_LAB_IDLE_TIMEOUT", IDLE_TIMEOUT)),
                        help="seconds a pooled model may sit unused before its weights are "
                             "dropped; 0 never drops one")
    parser.add_argument("--examples", type=str, default=os.environ.get("MI_LAB_EXAMPLES", ""),
                        help="root of exported held-out example files served at /examples; "
                             "empty uses the shipped data/sdft_examples")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    pool = None
    circuits = None

    if args.models.strip():
        # The pooled form: nothing is loaded here, on purpose. The startup probe
        # answers from `configs/` and the first request pays for its own model.
        specs = [ModelSpec(**{**vars(ModelSpec.parse(text)), "tasks": wanted})
                 for text in args.models.split(",") if text.strip()]
        for spec in specs:
            if spec.circuits is not None and not spec.circuits.is_dir():
                raise SystemExit(f"{spec.circuits} (for '{spec.name}') is not a directory; "
                                 f"write `{spec.name}={spec.config}@none` to serve it alone")
        pool = ModelPool(specs, idle_timeout=args.idle_timeout,
                         max_resident_circuits=args.max_resident)
        banner("model pool", {
            "models": ", ".join(
                f"{s.name} [{s.config}" + (f" @ {s.circuits}]" if s.circuits else " alone]")
                for s in specs),
            "loaded": "none yet -- each is loaded by the first request that names it",
            "idle timeout": (f"{args.idle_timeout:g}s, then the weights are dropped"
                             if args.idle_timeout > 0 else "never unloaded"),
            "listen": f"{args.host}:{args.port}",
        })
        pool.start(on_unload=lambda names: log(f"unloaded after {args.idle_timeout:g}s idle: "
                                               f"{', '.join(names)}"))
    else:
        # A served checkpoint with no circuits over it is a mode, not a missing
        # mount: the SDFT server (configs/qwen3-0.6b-sdft.yaml) is a model being
        # read for its own answers. Said explicitly so that a *mistyped* circuits
        # path still fails loudly instead of quietly serving the bare model.
        root = None if args.circuits.strip().lower() in ("", "none", "off") else Path(args.circuits)
        if root is not None and not root.is_dir():
            raise SystemExit(f"{root} is not a directory; pass --circuits none to serve the "
                             f"model alone")
        adapter = require_circuits(load_adapter(args.config))
        circuits = Circuits(adapter, root, tasks=wanted, config=args.config,
                            max_resident=args.max_resident)
        found = [
            f"{spec.name} [{spec.kind}"
            + (f" {spec.density:.2%}]" if spec.density is not None else "]")
            + (f" UNRUNNABLE: {spec.problem}" if spec.problem else "")
            for spec in sorted(circuits.specs.values(), key=lambda s: s.name)
        ]
        banner("circuit server" if root is not None else "generation server", {
            "config": args.config,
            "tasks": ", ".join(circuits.tasks) or "none found",
            "root": str(root) if root is not None else "none (model only)",
            # Nothing is on the device yet -- this is what the scan found, and
            # the first request for one of them is what reads it.
            "found": ", ".join(found) or "none",
            "resident": f"0 of {args.max_resident} max, loaded on first use",
            "listen": f"{args.host}:{args.port}",
        })
        if circuits.skipped:
            log(f"skipped (no backbone claims them): {', '.join(circuits.skipped)}")
        for spec in circuits.specs.values():
            if spec.problem:
                log(f"warning: '{spec.name}' is listed but cannot be run -- {spec.problem}")
        if root is not None and not circuits.names()[1:]:
            log(f"warning: no runnable circuit under {root}; only the full model is served")

    examples = Examples(Path(args.examples) if args.examples else None)
    found_examples = examples.load()
    log("examples: " + (", ".join(
        f"{d['dataset']} ({d['count']} held out of {d['split_size']})"
        for d in found_examples["datasets"]) or f"none under {found_examples['root']}"))
    for problem in found_examples["problems"]:
        log(f"warning: example file unreadable -- {problem}")
    uvicorn.run(create_app(circuits, config=args.config if circuits else None,
                           examples=examples, pool=pool),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
