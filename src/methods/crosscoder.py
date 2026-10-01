"""A crosscoder over two models' activations at one site, with the checks model diffing later found it needs

The crosscoder of Lindsey et al. (2024, "Sparse Crosscoders for Cross-Layer
Features and Model Diffing"): one encoder reads both models' activations at the
same site, side by side, into one set of latents, and each model gets its own
decoder. A latent's two decoder norms say which model uses it -- the relative
norm `|d_post| / (|d_pre| + |d_post|)` is near 0 for a latent only the first
model has, near 1 for one only the second has, and near one half for one both
share. That ratio is the whole of what "model diffing" reads off a crosscoder.

Two corrections from later work are built in rather than left to a caller:

- **BatchTopK, not an L1 penalty.** Minder et al. (2025, on chat fine-tuning
  diffed with crosscoders) found that under L1 most latents that look exclusive
  to the fine-tune are artifacts. *Complete shrinkage*: the penalty pushes a
  decoder norm to zero for a model that would in fact benefit from the latent.
  *Latent decoupling*: one concept split into an exclusive latent plus a shared
  one. A TopK budget charges a latent for being active, not for its norm, which
  removes the incentive behind the first; BatchTopK (Bussmann et al. 2024)
  spends the budget across the batch rather than per row.
- **Latent scaling as the receipt.** For a latent classed exclusive to one model,
  fit how much of the *other* model's activation its own direction explains,
  by least squares on a scalar. The ratio `nu` of the other model's coefficient
  to its own is near 0 for a latent that really is exclusive and near 1 for one
  the penalty or the budget merely shrank. A count of exclusive latents is
  reported only together with how many of them survive this.

AuxK (Gao et al. 2024) keeps latents from dying: latents silent for a while are
asked to reconstruct the main residual from their own top pre-activations.

The inputs are normalised per model to unit mean squared norm per dimension
before training, with the scales kept on the module, so the two models' losses
weigh the same whatever their activation scales are.

A common pipe could be: Crosscoder | fit_scales | train_step (repeat) | relative_norms | classify | latent_scaling
"""

import math
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn

from .common.errors import CrosscoderError

PRE, POST = 0, 1
# Relative-norm thresholds for the three classes; Lindsey et al. read the same histogram.
EXCLUSIVE = 0.1


class Crosscoder(nn.Module):
    """One encoder over [models, width] activations, one decoder per model, BatchTopK latents"""

    def __init__(self, width: int, latents: int, k: int, models: int = 2, seed: int = 0):
        super().__init__()
        if k <= 0 or k > latents:
            raise CrosscoderError(f"k is the number of active latents per row and must be in 1..{latents}, got {k}")
        generator = torch.Generator().manual_seed(seed)
        decoder = torch.randn(latents, models, width, generator=generator)
        # Decoder rows start at a small norm per model; the encoder starts as their transpose,
        # the initialisation that keeps latents alive from the first step.
        decoder = decoder / decoder.norm(dim=-1, keepdim=True) * 0.1
        self.W_dec = nn.Parameter(decoder)
        self.W_enc = nn.Parameter(decoder.permute(1, 2, 0).clone())          # [models, width, latents]
        self.b_enc = nn.Parameter(torch.zeros(latents))
        self.b_dec = nn.Parameter(torch.zeros(models, width))
        self.k = k
        self.register_buffer("scales", torch.ones(models))
        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("idle", torch.zeros(latents, dtype=torch.long))

    @property
    def latents(self) -> int:
        return self.W_dec.shape[0]

    def fit_scales(self, sample: torch.Tensor) -> None:
        """Per-model scale so that the mean squared norm per dimension is one, from a sample [rows, models, width]"""
        mean_square = (sample.float() ** 2).sum(-1).mean(0)                    # [models]
        self.scales.copy_((sample.shape[-1] / mean_square).sqrt())

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.scales.view(1, -1, 1)

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.einsum("bmw,mwl->bl", x, self.W_enc) + self.b_enc)

    def encode(self, x: torch.Tensor, batch_topk: bool = True) -> torch.Tensor:
        """Sparse latents for normalised rows: the batch's top k*rows, or the inference threshold"""
        pre = self.preactivations(x)
        if not batch_topk:
            return pre * (pre > self.threshold)
        keep = min(self.k * pre.shape[0], pre.numel())
        values, flat = pre.flatten().topk(keep)
        return torch.zeros_like(pre).flatten().scatter(0, flat, values).view_as(pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bl,lmw->bmw", f, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor, batch_topk: bool = True):
        f = self.encode(x, batch_topk)
        return self.decode(f), f


def train_step(coder: Crosscoder, optimiser, x: torch.Tensor, aux_k: int = 256, aux_weight: float = 1 / 32,
               dead_after: int = 400, threshold_rate: float = 0.01) -> Dict[str, float]:
    """One step on a batch of normalised rows [rows, models, width]; returns the loss terms and per-model FVU

    The inference threshold follows the smallest value BatchTopK kept, by a
    running average, so `encode(..., batch_topk=False)` activates about k latents
    per row without needing a batch.
    """
    f = coder.encode(x, batch_topk=True)
    reconstruction = coder.decode(f)
    error = x - reconstruction
    main = (error ** 2).sum(-1).mean()

    fired = (f > 0).any(0)
    coder.idle.add_(1).masked_fill_(fired, 0)
    dead = coder.idle > dead_after
    auxiliary = torch.zeros((), device=x.device)
    if dead.any():
        pre = coder.preactivations(x) * dead
        count = min(aux_k, int(dead.sum()))
        values, index = pre.topk(count, dim=-1)
        revived = torch.zeros_like(pre).scatter(-1, index, values)
        guess = torch.einsum("bl,lmw->bmw", revived, coder.W_dec)
        target = error.detach()
        auxiliary = ((target - guess) ** 2).sum(-1).mean() / (target ** 2).sum(-1).mean().clamp(min=1e-8)
    loss = main + aux_weight * auxiliary

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

    with torch.no_grad():
        kept = f[f > 0]
        if kept.numel():
            smallest = kept.min()
            coder.threshold.mul_(1 - threshold_rate).add_(threshold_rate * smallest)
        centred = ((x - x.mean(0, keepdim=True)) ** 2).sum((0, 2))
        fvu = (error ** 2).sum((0, 2)) / centred.clamp(min=1e-8)
    return {
        "loss": float(main.detach()),
        "auxiliary": float(auxiliary.detach()),
        "dead": int(dead.sum()),
        "fvu_pre": float(fvu[PRE]),
        "fvu_post": float(fvu[POST]),
        "l0": float((f > 0).float().sum(-1).mean()),
    }


def relative_norms(coder: Crosscoder) -> torch.Tensor:
    """|d_post| / (|d_pre| + |d_post|) per latent: 0 only the first model, 1 only the second"""
    norms = coder.W_dec.detach().norm(dim=-1)                                  # [latents, models]
    return norms[:, POST] / (norms[:, PRE] + norms[:, POST]).clamp(min=1e-12)


def decoder_cosines(coder: Crosscoder) -> torch.Tensor:
    """Cosine between a latent's two decoder directions; a shared latent can still be rotated"""
    decoder = coder.W_dec.detach()
    return torch.nn.functional.cosine_similarity(decoder[:, PRE], decoder[:, POST], dim=-1)


def classify(norms: torch.Tensor, live: torch.Tensor, exclusive: float = EXCLUSIVE) -> Dict[str, List[int]]:
    """Latents by relative norm, over the live ones only; a dead latent's norms mean nothing"""
    index = torch.arange(norms.numel(), device=norms.device)
    return {
        "pre_only": index[live & (norms < exclusive)].tolist(),
        "post_only": index[live & (norms > 1 - exclusive)].tolist(),
        "shared": index[live & (norms >= exclusive) & (norms <= 1 - exclusive)].tolist(),
    }


def latent_scaling(coder: Crosscoder, batches: Iterable[torch.Tensor], latents: Sequence[int], side: int,
                   ) -> List[Dict[str, float]]:
    """For latents classed exclusive to `side`, how much of each model they explain along their own direction

    Per latent j with direction d (its `side` decoder) and activation f_j, and per
    model m, two least-squares coefficients over all rows:

    - `error`: of what m's reconstruction leaves once j's own contribution to m is
      taken back out, `x_m - xhat_m + f_j d_{j,m}` -- what j would explain in m
      if its decoder for m were free.
    - `activation`: of m's activation itself, less the decoder bias.

    `nu = beta(other) / beta(side)`. Near 0: the other model has nothing along d
    where j fires, so j is exclusive. Near 1: the other model has it as much,
    and its decoder was shrunk, not absent -- the artifact.
    """
    if not latents:
        return []
    chosen = torch.as_tensor(list(latents), device=coder.W_dec.device)
    direction = coder.W_dec.detach()[chosen, side]                            # [J, width]
    own = coder.W_dec.detach()[chosen]                                         # [J, models, width]
    squared = (direction ** 2).sum(-1)                                         # [J]
    models = coder.W_dec.shape[1]
    numerator = {name: torch.zeros(models, len(chosen), device=direction.device) for name in ("error", "activation")}
    denominator = torch.zeros(len(chosen), device=direction.device)
    with torch.no_grad():
        for x in batches:
            reconstruction, f = coder(x, batch_topk=False)
            active = f[:, chosen]                                              # [rows, J]
            denominator += (active ** 2).sum(0) * squared
            for m in range(models):
                residual = (x[:, m] - reconstruction[:, m]) @ direction.T      # [rows, J]
                residual = residual + active * (own[:, m] * direction).sum(-1)
                numerator["error"][m] += (active * residual).sum(0)
                activation = (x[:, m] - coder.b_dec[m]) @ direction.T
                numerator["activation"][m] += (active * activation).sum(0)
    other = 1 - side
    report = []
    for position, latent in enumerate(latents):
        row = {"latent": int(latent), "activity": float(denominator[position])}
        for name, values in numerator.items():
            beta = values[:, position] / denominator[position].clamp(min=1e-12)
            row[f"beta_{name}_side"] = float(beta[side])
            row[f"beta_{name}_other"] = float(beta[other])
            row[f"nu_{name}"] = float(beta[other] / beta[side]) if abs(float(beta[side])) > 1e-12 else math.nan
        report.append(row)
    return report
