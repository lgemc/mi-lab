"""Phase 1b deliverable 3: what the candidate components cost in MACs.

The pre-registered ceiling says the circuit must be <= 25% of the model's
FLOPs, so before any ablation the bookkeeping has to say what a head and an
MLP are worth. Every dimension is read off the checkpoint's config (no
weights loaded); attention-score MACs depend on context length, which is a
measurement of the eval prompts, stated rather than assumed. Embeddings and
the LM head are excluded (they are not maskable components).

Run: uv run python -m scripts.phase1b_flops Qwen/Qwen3-8B 160
"""

import json
import sys
from pathlib import Path

RESULTS = Path("results/phase1b-flops-model.json")

CANDIDATE_LAYERS = list(range(27, 36))

def main() -> None:
    hf_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    context = int(sys.argv[2]) if len(sys.argv) > 2 else 160

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_name)
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_head = getattr(cfg, "head_dim", None) or d_model // n_heads
    d_ff = cfg.intermediate_size

    qo_per_head = 2 * d_model * d_head                      # this head's slice of q_proj and o_proj
    kv_per_head = 2 * d_model * d_head * n_kv // n_heads    # its share of the grouped k/v projections
    scores_per_head = 2 * context * d_head                  # QK^T and AV, per query token at this context
    head_macs = qo_per_head + kv_per_head + scores_per_head
    attention_macs = n_heads * head_macs
    mlp_macs = 3 * d_model * d_ff                           # gate, up, down
    layer_macs = attention_macs + mlp_macs
    total_macs = n_layers * layer_macs

    candidate_heads = len(CANDIDATE_LAYERS) * n_heads
    candidate = {
        "heads": candidate_heads,
        "mlps": len(CANDIDATE_LAYERS),
        "heads_macs_share": candidate_heads * head_macs / total_macs,
        "mlps_macs_share": len(CANDIDATE_LAYERS) * mlp_macs / total_macs,
    }
    candidate["total_share"] = candidate["heads_macs_share"] + candidate["mlps_macs_share"]

    heads_for_quarter = 0.25 * total_macs / head_macs
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": {"hf_name": hf_name, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                  "n_kv_heads": n_kv, "d_head": d_head, "d_ff": d_ff},
        "assumptions": {
            "context_tokens": context,
            "counted": "decoder-stack MACs per token: qkv/o projections, attention scores at the stated "
                       "context, and the three MLP matmuls; embeddings and LM head excluded",
        },
        "per_component_macs": {
            "head": head_macs,
            "mlp": mlp_macs,
            "mlp_over_head": round(mlp_macs / head_macs, 2),
            "attention_share_of_layer": round(attention_macs / layer_macs, 4),
        },
        "totals": {
            "per_layer_macs": layer_macs,
            "model_macs_per_token": total_macs,
            "all_heads_share": round(n_layers * attention_macs / total_macs, 4),
            "all_mlps_share": round(n_layers * mlp_macs / total_macs, 4),
        },
        "quarter_flops_line": {
            "heads_equal_to_25pct": round(heads_for_quarter, 1),
            "heads_in_model": n_layers * n_heads,
            "note": "more heads than the model has would be needed to reach 25% of FLOPs, so a head-only "
                    "circuit can never touch the pre-registered ceiling; the ceiling binds through MLPs, "
                    "each of which costs "
                    f"{round(100 * mlp_macs / total_macs, 2)}% of the model on its own"
                    if heads_for_quarter > n_layers * n_heads else
                    f"{round(heads_for_quarter)} heads reach 25% of FLOPs",
        },
        "candidate_set_layers_27_35": {key: round(value, 4) if isinstance(value, float) else value
                                       for key, value in candidate.items()},
        "masked_fraction_examples": {
            f"p={p}": round(p * candidate["total_share"], 4)
            for p in (0.25, 0.5, 0.75, 1.0)
        },
        "command": f"uv run python -m scripts.phase1b_flops {hf_name} {context}",
    }, indent=2) + "\n")
    print(json.dumps(json.loads(RESULTS.read_text())["candidate_set_layers_27_35"], indent=2))
    print("head MACs", head_macs, "| mlp MACs", mlp_macs, "| total/token", total_macs)

if __name__ == "__main__":
    main()
