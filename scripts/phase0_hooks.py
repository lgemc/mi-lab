"""Phase 0 deliverable 4: is every site the circuit work needs actually hookable?

Four checks, all through the repo's own adapter so nothing is verified that an
experiment would not use:

- residual stream: capture() at every layer,
- attention heads: head_outputs() at every layer,
- MLPs: decompose(), whose `remainder` is the receipt that the recorded head,
  MLP, bias and embedding writes really sum to the residual stream the model
  produced (the check tests/circuits.py runs on GPT-2, here on the new model),
- round trip: capture a run, patch the same values back in, and demand the
  logits do not move -- a patch that writes back what was read is exactly a
  no-op, and every causal number later is a difference against that no-op.

Shape facts (layer count, heads, MLP width) are read off the loaded model,
never stated, and land in results/phase0-feasibility.json.

A common pipe could be: capture | head_outputs | decompose | patch | merge_section

Run: uv run python -m scripts.phase0_hooks qwen3-8b
"""

import sys

import torch

from scripts.phase0_smoke import merge
from src.core.config import Position
from src.experiment import translation_study as study
from src.model.adapter import load_adapter, require_circuits
from src.telemetry.results import guard

PROMPTS = [
    "El gato duerme al sol. The cat sleeps in the sun.",
    "La ciudad despierta temprano. The city wakes early.",
]

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    guard(config)
    adapter = require_circuits(load_adapter(config))
    cfg = adapter.cfg
    layers = list(range(cfg.n_layers))

    residual = adapter.capture(PROMPTS, layers=layers, position=Position.ALL)
    heads = adapter.head_outputs(PROMPTS, layers=layers)

    decomposition = adapter.decompose(PROMPTS)
    remainder = float(decomposition.remainder.abs().max())
    residual_norm = float(decomposition.residual.norm(dim=-1).mean())
    # bf16 keeps ~8 mantissa bits, and Qwen3's residual stream carries
    # massive-activation coordinates in the hundreds-to-thousands, where one
    # bf16 ulp is already 4-8 -- so the honest zero here is relative, not
    # absolute the way GPT-2's float32 ~1e-6 is in tests/circuits.py.
    relative_remainder = float(
        (decomposition.remainder.norm(dim=-1) / decomposition.residual.norm(dim=-1)).max()
    )

    baseline = adapter.logits(PROMPTS)
    middle = cfg.layer(0.5)
    with adapter.patch(residual={middle: residual[:, middle]}):
        residual_replay = adapter.logits(PROMPTS)
    with adapter.patch(heads={middle: {0: heads[:, middle, 0]}}):
        head_replay = adapter.logits(PROMPTS)

    mlp_widths = {type(mlp).__name__ for mlp in adapter.mlps} if hasattr(adapter, "mlps") else set()
    intermediate = getattr(adapter.model.config, "intermediate_size", None)

    merge("hooks", {
        "config": config,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "n_kv_heads": getattr(adapter.model.config, "num_key_value_heads", None),
        "mlp_width": intermediate,
        "mlp_module": sorted(mlp_widths),
        "residual_capture_shape": list(residual.shape),
        "head_outputs_shape": list(heads.shape),
        "decompose_remainder_max_abs": remainder,
        "decompose_remainder_relative": relative_remainder,
        "decompose_residual_mean_norm": round(residual_norm, 2),
        "residual_patch_noop_max_logit_shift": float((residual_replay - baseline).abs().max()),
        "head_patch_noop_max_logit_shift": float((head_replay - baseline).abs().max()),
        "verdict": {
            "residual_hookable": True,
            "heads_hookable": True,
            "mlps_hookable": intermediate is not None and bool(mlp_widths),
            "round_trip_exact": bool(torch.equal(residual_replay, baseline) and torch.equal(head_replay, baseline)),
        },
        "command": f"uv run python -m scripts.phase0_hooks {config}",
    })
    print("remainder", remainder, "| residual noop", float((residual_replay - baseline).abs().max()),
          "| head noop", float((head_replay - baseline).abs().max()))

if __name__ == "__main__":
    main()
