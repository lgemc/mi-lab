"""Phase 1b deliverable 3: what the candidate components cost in MACs.

The pre-registered ceiling says the circuit must be <= 25% of the model's
FLOPs, so before any ablation the bookkeeping has to say what a head and an
MLP are worth. `methods.cost` does the counting; this script names the model,
states the context length (a measurement of the eval prompts, not an
assumption) and writes the table every later flops_share divides by.

A common pipe could be: read_dimensions | CostModel.from_dimensions | report | write

Run: uv run python -m scripts.phase1b_flops qwen3-8b 160
"""

import json
import sys
from dataclasses import replace

from src.core.config import load_config
from src.experiment import translation_study as study
from src.methods import components as comp
from src.methods.cost import CostModel, read_dimensions, report
from src.telemetry.results import guard

DEFAULT_CONTEXT = 160    # tokens per few-shot eval prompt, measured on the WMT shortlist

RESULTS = study.artifact("cost")

def main() -> None:
    # a config id like every sibling, not a bare hf_name. Taking the checkpoint
    # name meant this was the one script with no config to guard the results
    # directory with, so running it for a second model overwrote the first
    # model's MAC bookkeeping in place -- and every flops_share downstream would
    # then have been dividing by the wrong denominator without saying so.
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    context = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CONTEXT
    guard(config)

    model = load_config(config)
    dims = read_dimensions(model.hf_name)
    cost = CostModel.from_dimensions(dims, context)
    # the band is a depth fraction; it needs the layer count the config may not state
    band = comp.band(replace(model, n_layers=dims.n_layers))

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        **report(dims, cost, band, comp.CANDIDATE_BAND),
        "command": f"uv run python -m scripts.phase1b_flops {config} {context}",
    }, indent=2) + "\n")
    print(json.dumps(json.loads(RESULTS.read_text())["candidate_set"], indent=2))
    print("head MACs", cost.head_macs, "| mlp MACs", cost.mlp_macs, "| total/token", cost.total_macs)

if __name__ == "__main__":
    main()
