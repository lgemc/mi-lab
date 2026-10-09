"""The fine-tune's update itself, as an object to truncate and to decompose.

`diffing.py` measures how big the update between two checkpoints is and how
many directions it spreads over. This module acts on it. Both tools below keep
the *earlier* checkpoint's weights fixed and put back some part of the update
`D = W_after - W_before`, one linear map at a time, so every model they build
sits on the line between the two checkpoints: nothing of the update is the
earlier model, all of it is the later one.

- **Truncation** (`truncate`): the top-k singular directions of each D, or all
  but those. The plain-SVD answer to "is the forgetting in a few directions",
  and the baseline the decomposition has to beat (the *SVD on weight
  differences* audit, Zhong & Raghunathan 2508.00161, does this at rank one).
- **Delta parameter decomposition** (`DeltaLinear`, `install`): D written as a
  sum of rank-1 subcomponents `u_c v_c^T` plus a high-rank leftover, each
  subcomponent switched per position by a learned causal-importance gate. The
  method is Stochastic / adVersarial Parameter Decomposition (Bushnaq et al.
  2506.20790, and the VPD technical report, 2026) applied to the update instead
  of to the weights, with the earlier checkpoint held fixed as the thing every
  ablation falls back to; the leftover plays the part of the catch-all in
  targeted parameter decomposition (Vigouroux & Sharkey 2607.13047). Masking a
  subcomponent to zero reverts that piece of the update. A subcomponent whose
  gate is open on the earlier task's text and closed on the later task's is a
  piece of the update the later task did not need and the earlier one pays for.

The gates read each map's own input, through a small two-layer network per map
(SPD's `vector_mlp` causal-importance function), not VPD's global transformer:
the model here is hundreds of maps deep, and a global gate network the size of
VPD's would be larger than the update it explains.

A common pipe could be: targets | updates | DeltaLinear.from_update | install | train | gates | revert
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional

# The linear maps a transformer block owns, by the suffix of their module path.
# Embeddings and norms are left at the earlier checkpoint's values by every
# model built here; `truncate` reports their share separately.
TARGET = re.compile(r"\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)$")


class DeltaError(ValueError):
    """Two checkpoints whose update cannot be taken, said with what differs"""


def targets(model: nn.Module) -> Dict[str, nn.Linear]:
    """Every block linear map, by module path, in depth order"""
    found = {name: module for name, module in model.named_modules()
             if isinstance(module, nn.Linear) and TARGET.search(name)}
    if not found:
        raise DeltaError("no block linear maps matched; this model's layout is not the one TARGET names")
    return found


def kind(name: str) -> Tuple[int, str]:
    """(layer, short name) of a target path: (12, 'down_proj')"""
    match = TARGET.search(name)
    if match is None:
        raise DeltaError(f"{name} is not a block linear map")
    return int(match.group(1)), match.group(2).split(".")[-1]


def updates(before: nn.Module, after: nn.Module, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """W_after - W_before for every target map, in float32

    float32 because the update is a few thousandths of the weight it is added
    to, and a half-precision subtraction would round most of it away.
    """
    mine, theirs = targets(before), targets(after)
    if mine.keys() != theirs.keys():
        raise DeltaError("the two checkpoints do not have the same linear maps")
    out = {}
    for name, module in mine.items():
        other = theirs[name]
        if module.weight.shape != other.weight.shape:
            raise DeltaError(f"{name}: {tuple(module.weight.shape)} against {tuple(other.weight.shape)}")
        out[name] = other.weight.detach().to(device, torch.float32) - module.weight.detach().to(device, torch.float32)
    return out


@dataclass
class Spectrum:
    """The singular values of one update, and the counts read off them"""
    name: str
    layer: int
    kind: str
    norm: float
    relative: float
    stable_rank: float
    rank_for: Dict[str, int]
    head: List[float]

    @classmethod
    def of(cls, name: str, update: torch.Tensor, weight: torch.Tensor, keep: int = 64) -> "Spectrum":
        values = torch.linalg.svdvals(update.float())
        energy = values.pow(2)
        cumulative = (energy.cumsum(0) / energy.sum()).tolist()
        rank_for = {f"{share:.2f}": next(i + 1 for i, c in enumerate(cumulative) if c >= share)
                    for share in (0.5, 0.9, 0.99)}
        layer, short = kind(name)
        norm = float(energy.sum().sqrt())
        return cls(name=name, layer=layer, kind=short, norm=norm,
                   relative=norm / float(weight.float().norm()),
                   stable_rank=float(energy.sum() / energy[0]), rank_for=rank_for,
                   head=values[:keep].tolist())


class Factors:
    """The full SVD of every update, kept so any truncation is one matmul away"""

    def __init__(self, deltas: Dict[str, torch.Tensor], device: str):
        self.parts: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, update in deltas.items():
            u, s, vh = torch.linalg.svd(update.to(device), full_matrices=False)
            self.parts[name] = (u, s, vh)

    def piece(self, name: str, top: Optional[int], complement: bool) -> torch.Tensor:
        """The top `top` directions of one update, or everything but them; None means all of it"""
        u, s, vh = self.parts[name]
        chosen = slice(None) if top is None else slice(top, None) if complement else slice(0, top)
        return (u[:, chosen] * s[chosen]) @ vh[chosen]


@torch.no_grad()
def truncate(work: nn.Module, base: Dict[str, torch.Tensor], factors: Factors,
             top: Optional[int], complement: bool = False) -> None:
    """Set `work`'s block maps to W_before plus a truncation of the update, in place

    `base` holds the earlier checkpoint's weights in float32, so repeated
    truncations of one working copy never accumulate rounding.
    """
    for name, module in targets(work).items():
        module.weight.copy_((base[name] + factors.piece(name, top, complement)).to(module.weight.dtype))


# --- the decomposition ------------------------------------------------------


def leaky_hard_sigmoid(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """clamp(x, 0, 1) with a small slope below zero, so a closed gate still has a gradient

    The VPD report's `lower_leaky` gate: the value is a hard sigmoid, and below
    zero the gradient passes at `alpha` so a gate that closed early can reopen.
    """
    hard = x.clamp(0.0, 1.0)
    leak = alpha * x.clamp(max=0.0)
    return hard + (leak - leak.detach())


class DeltaLinear(nn.Module):
    """One linear map as W_before x + sum_c m_c (v_c . x) u_c + m_rest (D - V U) x

    `base` is the earlier checkpoint's map, frozen. `V` [d_in, C] and `U`
    [C, d_out] are the subcomponents; the leftover `D - V U` is recomputed from
    the stored update on every pass, so the three parts always sum to the later
    checkpoint exactly. `mask` [B, S, C] and `rest_mask` [B, S] are set by the
    caller before a pass; None means fully on. `capture` keeps the map's input,
    which is what the gates read.
    """

    def __init__(self, base: nn.Linear, update: torch.Tensor, components: int, hidden: int):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        d_out, d_in = update.shape
        components = min(components, d_in, d_out)
        self.register_buffer("update", update.float(), persistent=False)
        u, s, vh = torch.linalg.svd(update.float(), full_matrices=False)
        root = s[:components].sqrt()
        # Starting from the update's own leading directions makes the sum
        # faithful from step one; the gates then rotate them toward directions
        # that are used together.
        self.V = nn.Parameter((vh[:components] * root[:, None]).T.contiguous())
        self.U = nn.Parameter((u[:, :components] * root).T.contiguous())
        self.gate_in = nn.Linear(d_in, hidden)
        self.gate_out = nn.Linear(hidden, components)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, 1.0)
        self.mask: Optional[torch.Tensor] = None
        self.rest_mask: Optional[torch.Tensor] = None
        self.capture = False
        self.captured: Optional[torch.Tensor] = None
        # When set and no mask is given, each pass masks by its own gates: the
        # decomposition's account of the model, with nothing outside it.
        self.gated = False

    @property
    def components(self) -> int:
        return self.V.shape[1]

    def rest(self) -> torch.Tensor:
        return self.update - (self.V @ self.U).T

    def gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(gate, upper-leaky gate) per position and subcomponent, from this map's input"""
        pre = self.gate_out(functional.gelu(self.gate_in(x.float())))
        return leaky_hard_sigmoid(pre), pre.clamp(max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture:
            self.captured = x.detach()
        out = self.base(x)
        wide = x.float()
        inner = wide @ self.V
        full = inner @ self.U
        # The leftover's output is D x minus what the subcomponents write when
        # all of them are on; never forming D - V U keeps a pass from holding a
        # matrix per map for the backward.
        rest = functional.linear(wide, self.update) - full
        if self.mask is not None:
            inner = inner * self.mask
        elif self.gated:
            inner = inner * self.gates(x)[0]
        delta = inner @ self.U
        if self.rest_mask is not None:
            rest = rest * self.rest_mask.unsqueeze(-1)
        return out + (delta + rest).to(out.dtype)

    def rest_norm(self) -> torch.Tensor:
        """||D - (V U)^T||_F^2 without forming the difference

        ||D||^2 - 2 sum_c u_c . D v_c + sum((V^T V) * (U U^T)).
        """
        cross = ((self.U @ self.update) * self.V.T).sum()
        square = ((self.V.T @ self.V) * (self.U @ self.U.T)).sum()
        return self.update.pow(2).sum() - 2 * cross + square


def install(model: nn.Module, deltas: Dict[str, torch.Tensor], components: Dict[str, int],
            hidden: int) -> Dict[str, DeltaLinear]:
    """Replace each target map of the earlier checkpoint with its DeltaLinear, in place"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = {}
    for name, module in targets(model).items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        wrapped = DeltaLinear(module, deltas[name].to(module.weight.device), components[name], hidden)
        wrapped.to(module.weight.device)
        setattr(parent, child, wrapped)
        installed[name] = wrapped
    return installed


def wrapped(model: nn.Module) -> Dict[str, DeltaLinear]:
    """The DeltaLinear maps already installed in a model, by path"""
    return {name: module for name, module in model.named_modules() if isinstance(module, DeltaLinear)}


def set_masks(maps: Dict[str, DeltaLinear], masks: Optional[Dict[str, torch.Tensor]],
              rest: Optional[Dict[str, torch.Tensor]]) -> None:
    for name, module in maps.items():
        module.mask = None if masks is None else masks[name]
        module.rest_mask = None if rest is None else rest[name]


def gating(maps: Dict[str, DeltaLinear], on: bool) -> None:
    for module in maps.values():
        module.gated = on


def capturing(maps: Dict[str, DeltaLinear], on: bool) -> None:
    for module in maps.values():
        module.capture = on
        if not on:
            module.captured = None


def kl(target_logits: torch.Tensor, logits: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean KL(target || model) over the positions `keep` marks (all of them if None)"""
    target = target_logits.float().log_softmax(-1)
    model = logits.float().log_softmax(-1)
    per = (target.exp() * (target - model)).sum(-1)
    if keep is None:
        return per.mean()
    return (per * keep).sum() / keep.sum().clamp(min=1)


def faithfulness(maps: Dict[str, DeltaLinear]) -> torch.Tensor:
    """Share of the update's squared norm left in the leftovers: 0 when the subcomponents are all of it"""
    left = sum(module.rest_norm() for module in maps.values())
    total = sum(module.update.pow(2).sum() for module in maps.values())
    return left / total


def minimality(upper: Dict[str, torch.Tensor], p: float, keep: torch.Tensor) -> torch.Tensor:
    """sum_c |g_c|^p per position, summed over maps, averaged over the kept positions

    Taken on the upper-leaky gate (SPD's choice), so a gate pushed below zero
    is not penalised for it.
    """
    total = 0.0
    for gate in upper.values():
        total = total + (gate.clamp(min=0.0) + 1e-12).pow(p).sum(-1)
    return (total * keep).sum() / keep.sum().clamp(min=1)


def mask_from(gate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """A mask between the gate and one: closed gates may take any value, open ones must stay on"""
    return gate + (1.0 - gate) * source


def counts(maps: Dict[str, DeltaLinear]) -> Iterator[Tuple[str, int]]:
    for name, module in maps.items():
        yield name, module.components


def component_names(maps: Dict[str, DeltaLinear]) -> List[Tuple[str, int]]:
    """(map path, index) for every subcomponent, in a fixed order"""
    return [(name, c) for name, module in maps.items() for c in range(module.components)]


def schedule(step: int, total: int, start: float, end: float) -> float:
    """Linear from start to end over the run"""
    return start + (end - start) * min(step / max(total - 1, 1), 1.0)


def cosine_lr(step: int, total: int, warmup: int, final: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress))


def revert(maps: Dict[str, DeltaLinear], chosen: Sequence[Tuple[str, int]], rest_on: bool = True) -> None:
    """Fixed masks: every subcomponent on except the chosen ones, which go back to the earlier model

    Shaped [1, 1, C] so one setting serves sequences of any length.
    """
    masks, rest = {}, {}
    for name, module in maps.items():
        device = module.V.device
        masks[name] = torch.ones(1, 1, module.components, device=device)
        rest[name] = torch.full((1, 1), 1.0 if rest_on else 0.0, device=device)
    for name, c in chosen:
        masks[name][..., c] = 0.0
    set_masks(maps, masks, rest)
