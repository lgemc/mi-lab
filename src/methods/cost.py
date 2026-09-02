"""What a component costs, in the MACs a forward pass spends on it, and sets matched on that.

The pre-registered ceiling of the translation study says the circuit must be
at most a quarter of the model's FLOPs, so before any ablation the bookkeeping
has to say what a head and an MLP are worth. Every dimension is read off the
checkpoint's config with no weights loaded; attention-score MACs depend on
context length, which is a measurement of the eval prompts and is stated
rather than assumed. Embeddings and the LM head are excluded, because they
are not maskable components.

The second thing here is the reason the first matters: a random control has
to be matched to the discovered set *by cost*, not by count. One MLP costs as
many MACs as dozens of heads, so a set matched on component count is not
matched on anything the ceiling is about, and the comparison it licenses is
not the one being claimed. `matched_draw` samples to a share of model MACs.

A common pipe could be: read_dimensions | CostModel.from_dimensions | share | matched_draw
A common pipe could be: read_dimensions | CostModel.from_dimensions | report | json
"""

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

from . import components
from .circuits import CircuitError

# A set matched to within this fraction of the target's MACs counts as
# matched. Components are lumpy -- one MLP costs many heads -- so an exact
# match is generally not on the lattice, and the honest move is to state the
# tolerance rather than to pretend the draw landed on the number.
MATCH_TOLERANCE = 0.05

class CostError(CircuitError):
    """A cost that cannot be computed for the model or the set it was asked about"""

@dataclass(frozen=True)
class Dimensions:
    """The shape facts a MAC count needs, read off a checkpoint config"""
    hf_name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int

def read_dimensions(hf_name: str) -> Dimensions:
    """The dimensions of a checkpoint from its config alone -- no weights are loaded"""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(hf_name)
    # `hidden_size` and friends are the attribute map every config exposes;
    # the MLP width is the one fact without a common name (GPT-2's `n_inner`
    # is None for "four times the width"), so it is read by candidates
    d_model = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    d_ff = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or 4 * d_model
    return Dimensions(
        hf_name=hf_name,
        d_model=d_model,
        n_layers=cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=getattr(cfg, "num_key_value_heads", None) or n_heads,
        d_head=getattr(cfg, "head_dim", None) or d_model // n_heads,
        d_ff=d_ff,
    )

@dataclass(frozen=True)
class CostModel:
    """MACs per token for one head, one MLP and the whole decoder stack"""
    head_macs: int
    mlp_macs: int
    n_heads: int
    n_layers: int
    context: int

    @classmethod
    def from_dimensions(cls, dims: Dimensions, context: int) -> "CostModel":
        qo_per_head = 2 * dims.d_model * dims.d_head                          # this head's slice of q and o
        kv_per_head = 2 * dims.d_model * dims.d_head * dims.n_kv_heads // dims.n_heads  # its share of grouped k/v
        scores_per_head = 2 * context * dims.d_head                           # QK^T and AV at this context
        # gate, up, down -- three matmuls, which is also right for a two-matmul
        # MLP within the tolerance any matched draw is held to
        mlp_macs = 3 * dims.d_model * dims.d_ff
        return cls(head_macs=qo_per_head + kv_per_head + scores_per_head, mlp_macs=mlp_macs,
                   n_heads=dims.n_heads, n_layers=dims.n_layers, context=context)

    @property
    def attention_macs(self) -> int:
        return self.n_heads * self.head_macs

    @property
    def layer_macs(self) -> int:
        return self.attention_macs + self.mlp_macs

    @property
    def total_macs(self) -> int:
        return self.n_layers * self.layer_macs

    def macs(self, chosen: Sequence[str]) -> int:
        """The MACs the named components spend per token"""
        total = 0
        for cid in chosen:
            kind, _, _ = components.parse(cid)
            total += self.mlp_macs if kind == "mlp" else self.head_macs * (self.n_heads if kind == "heads" else 1)
        return total

    def share(self, chosen: Sequence[str]) -> float:
        """The named components as a fraction of the model's MACs per token"""
        return self.macs(chosen) / self.total_macs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostModel":
        """Read back either this shape or the phase1b-flops-model.json layout that preceded it"""
        if "head_macs" in data:
            return cls(**{key: data[key] for key in ("head_macs", "mlp_macs", "n_heads", "n_layers", "context")})
        try:
            return cls(
                head_macs=data["per_component_macs"]["head"],
                mlp_macs=data["per_component_macs"]["mlp"],
                n_heads=data["model"]["n_heads"],
                n_layers=data["model"]["n_layers"],
                context=data["assumptions"]["context_tokens"],
            )
        except KeyError as error:
            raise CostError(f"not a cost model: missing {error}") from None

def matched_draw(pool: Sequence[str], cost: CostModel, target_share: float, seed: int) -> Tuple[List[str], float]:
    """A random component set costing as close to `target_share` of model MACs as the lattice allows

    Returns the set and its relative match error. Sampled without replacement
    in a shuffled order, taking every component that still fits under the
    target rather than stopping at the first one that does not. Stopping at
    the first overshoot ends the draw on an MLP while hundreds of heads that
    would have fitted go unused, which leaves the arm systematically *under*
    the budget it claims to match; scanning on fills the remainder with the
    cheap components and tightens the match to roughly one head's worth. The
    single cheapest skipped component is then added if overshooting by it
    lands nearer the target than stopping short.

    The error is returned rather than judged: a lattice too coarse to match is
    not fatal, but the arm is then not matched, and a mismatched control is a
    different claim -- the caller compares against `MATCH_TOLERANCE` and says so.
    """
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    chosen: List[str] = []
    skipped: List[str] = []
    for cid in order:
        if cost.share([*chosen, cid]) <= target_share:
            chosen.append(cid)
        else:
            skipped.append(cid)
    if skipped:
        cheapest = min(skipped, key=lambda cid: cost.share([cid]))
        if abs(cost.share([*chosen, cheapest]) - target_share) < abs(cost.share(chosen) - target_share):
            chosen.append(cheapest)
    if not chosen:
        raise CostError(
            f"no component in the pool fits under a target share of {target_share:.4f}; the target "
            "set is cheaper than the cheapest single component, so there is nothing to match"
        )
    error = abs(cost.share(chosen) - target_share) / target_share
    return chosen, error

CEILING_SHARE = 0.25        # the pre-registered ceiling the quarter-FLOPs line is drawn at
MASKED_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

def report(dims: Dimensions, cost: CostModel, band: Sequence[int], fractions: Tuple[float, float]) -> Dict[str, Any]:
    """The cost model as the study's on-disk bookkeeping, with the candidate band priced

    This is the layout `from_dict` reads back, plus the human-facing lines the
    numbers exist for: the quarter-FLOPs line (how many heads reach the
    ceiling, which on every model so far is more heads than it has), and the
    band's share at each masked fraction.
    """
    candidate_heads = len(band) * dims.n_heads
    heads_share = candidate_heads * cost.head_macs / cost.total_macs
    mlps_share = len(band) * cost.mlp_macs / cost.total_macs
    heads_for_ceiling = CEILING_SHARE * cost.total_macs / cost.head_macs
    heads_in_model = dims.n_layers * dims.n_heads
    percent = round(100 * CEILING_SHARE)
    return {
        "model": {"hf_name": dims.hf_name, "d_model": dims.d_model, "n_layers": dims.n_layers,
                  "n_heads": dims.n_heads, "n_kv_heads": dims.n_kv_heads, "d_head": dims.d_head, "d_ff": dims.d_ff},
        "assumptions": {
            "context_tokens": cost.context,
            "counted": "decoder-stack MACs per token: qkv/o projections, attention scores at the stated "
                       "context, and the three MLP matmuls; embeddings and LM head excluded",
        },
        "per_component_macs": {
            "head": cost.head_macs,
            "mlp": cost.mlp_macs,
            "mlp_over_head": round(cost.mlp_macs / cost.head_macs, 2),
            "attention_share_of_layer": round(cost.attention_macs / cost.layer_macs, 4),
        },
        "totals": {
            "per_layer_macs": cost.layer_macs,
            "model_macs_per_token": cost.total_macs,
            "all_heads_share": round(dims.n_layers * cost.attention_macs / cost.total_macs, 4),
            "all_mlps_share": round(dims.n_layers * cost.mlp_macs / cost.total_macs, 4),
        },
        "quarter_flops_line": {
            f"heads_equal_to_{percent}pct": round(heads_for_ceiling, 1),
            "heads_in_model": heads_in_model,
            "note": (f"more heads than the model has would be needed to reach {percent}% of FLOPs, so a "
                     "head-only circuit can never touch the pre-registered ceiling; the ceiling binds "
                     "through MLPs, each of which costs "
                     f"{round(100 * cost.mlp_macs / cost.total_macs, 2)}% of the model on its own"
                     if heads_for_ceiling > heads_in_model else
                     f"{round(heads_for_ceiling)} heads reach {percent}% of FLOPs"),
        },
        "candidate_set": {
            "band_depth_fraction": list(fractions),
            "layers": list(band),
            "heads": candidate_heads,
            "mlps": len(band),
            "heads_macs_share": round(heads_share, 4),
            "mlps_macs_share": round(mlps_share, 4),
            "total_share": round(heads_share + mlps_share, 4),
        },
        "masked_fraction_examples": {
            f"p={p}": round(p * (heads_share + mlps_share), 4) for p in MASKED_FRACTIONS
        },
    }
