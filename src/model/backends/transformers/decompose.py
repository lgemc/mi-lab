"""
Splitting the final token's residual stream into the writes that built it.

The residual stream is a sum, so the correlational half of a circuit study is
arithmetic: what the embedding put there, what each head wrote, what each
attention bias added, what each MLP contributed. One forward pass records all
of it, and `Decomposition.remainder` is the receipt -- the parts have to sum to
the whole the model actually produced, and that check is the first thing to
look at on an architecture this backend has not met.

Two traps are handled here rather than assumed away. A head's write is computed
by calling the projection with every other head zeroed, never by slicing its
weight: GPT-2 stores a Conv1D as [in, out] and everything else stores a Linear
as [out, in], and a transposed slice is wrong in a way that still produces
plausible numbers. And the final norm's divisor is frozen from the complete
stream -- whether that norm centres is settled by asking the module, because a
LayerNorm is invariant to adding a constant to every coordinate and an RMSNorm
is not.

A common pipe could be: decompose | direct_logit_attribution | classify_heads
"""

from typing import Dict, Sequence, Tuple

import torch

from ....core.config import ConfigError
from ...adapter import Decomposition, Unembedding
from .layout import _final_norm
from .positions import _last_real


def _normalizer(norm, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Read a final norm's frozen divisor and elementwise gain off one forward pass

    Whether the norm subtracts the mean is settled by asking the module rather
    than by recognizing its class: a LayerNorm is invariant to adding the same
    constant to every coordinate and an RMSNorm is not, so two calls decide it
    for a normalization this framework has never seen.
    """
    with torch.no_grad():
        # everything below this stays on CPU on purpose, but the norm is a live module
        # on the model's device -- so the two probes, and only they, have to be moved to
        # it. Handing a CPU residual to a norm with CUDA weights is the failure.
        parameter = next(norm.parameters(), None)
        probe = residual[:1] if parameter is None else residual[:1].to(parameter.device)
        centers = bool(torch.allclose(norm(probe), norm(probe + 1.0), atol=1e-3, rtol=1e-3))

    width = residual.shape[-1]
    weight = getattr(norm, "weight", None)
    shift = getattr(norm, "bias", None)
    gain = torch.ones(width, dtype=torch.float32) if weight is None else weight.detach().float().cpu()
    bias = torch.zeros(width, dtype=torch.float32) if shift is None else shift.detach().float().cpu()
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", 1e-5)

    values = residual.float().cpu()
    centered = values - values.mean(dim=-1, keepdim=True) if centers else values
    scale = (centered.pow(2).mean(dim=-1, keepdim=True) + float(eps)).sqrt()
    return gain, bias, scale, centers

class DecomposeMixin:
    """Every write into the final token, and the frozen unembedding that scores it"""

    def _head_writes(self, index: int, merged: torch.Tensor) -> torch.Tensor:
        """What each head at this layer wrote into the residual stream, as [batch, head, d_model]

        Computed by running the projection on one copy per head with every
        other head zeroed, rather than by slicing the projection's weight. The
        weight layouts disagree -- GPT-2 stores a Conv1D as [in, out] and
        everything else stores a Linear as [out, in] -- and a transposed slice
        is wrong in a way that still produces plausible numbers. Calling the
        module cannot be transposed by accident.
        """
        batch, width = merged.shape
        selector = torch.eye(self.cfg.n_heads, dtype=merged.dtype, device=merged.device)
        isolated = (self._split_heads(merged)[:, None] * selector[None, :, :, None]).reshape(
            batch, self.cfg.n_heads, width
        )
        with torch.no_grad():
            projection = self.projections[index]
            # the projection's bias is written once per layer, not once per head, so it is subtracted back out
            return (projection(isolated) - projection(merged.new_zeros(1, width))).float().cpu()

    def _bias_write(self, index: int, width: int, dtype, device) -> torch.Tensor:
        """The attention output projection's own bias, which belongs to no head"""
        with torch.no_grad():
            return self.projections[index](torch.zeros(1, width, dtype=dtype, device=device)).float().cpu()

    def _decompose_chunk(self, ids: torch.Tensor, mask: torch.Tensor, layers: Sequence[int]) -> Dict[str, torch.Tensor]:
        """Every write into one chunk's final token, recorded in a single forward pass"""
        embedding: Dict[int, torch.Tensor] = {}
        mlp_writes: Dict[int, torch.Tensor] = {}

        def embedding_hook(module, args):
            embedding[0] = args[0].detach()

        def make_mlp_hook(index: int):
            def hook(module, args, output):
                mlp_writes[index] = (output[0] if isinstance(output, tuple) else output).detach()
            return hook

        handles = [self.blocks[0].register_forward_pre_hook(embedding_hook)]
        handles.extend(self.mlps[index].register_forward_hook(make_mlp_hook(index)) for index in layers)
        try:
            with self._record_heads(layers) as merged, self._record(layers) as residual, torch.no_grad():
                self.model(ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        last = {index: _last_real(merged[index], mask) for index in layers}
        biases = torch.cat([
            self._bias_write(index, last[index].shape[-1], last[index].dtype, last[index].device)
            for index in layers
        ])
        mlps = torch.stack([_last_real(mlp_writes[index], mask) for index in layers], dim=1)
        return {
            "heads": torch.stack([self._head_writes(index, last[index]) for index in layers], dim=1),
            "mlps": mlps.float().cpu(),
            "biases": biases.expand(ids.shape[0], -1, -1).clone(),
            "embedding": _last_real(embedding[0], mask).float().cpu(),
            "residual": _last_real(residual[layers[-1]], mask).float().cpu(),
        }

    def decompose(self, prompts: Sequence[str]) -> Decomposition:
        """Split the final token's residual stream into the writes that built it

        One forward pass, every site recorded: what the embedding put there,
        what each head wrote, what each attention bias added, what each MLP
        contributed, and what the last block handed to the final norm. The
        parts sum to the whole exactly -- `remainder` is the check, and it is
        the first thing to look at on an architecture this has not met.
        """
        if not prompts:
            raise ConfigError("decompose needs at least one prompt")
        layers = list(range(self.cfg.n_layers))
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        parts = {name: [] for name in ("heads", "mlps", "biases", "embedding", "residual")}
        for ids, mask in self._chunks(input_ids, attention_mask):
            for name, value in self._decompose_chunk(ids, mask, layers).items():
                parts[name].append(value)

        joined = {name: torch.cat(values, dim=0) for name, values in parts.items()}
        norm = _final_norm(self.model)
        gain, bias, scale, centers = _normalizer(norm, joined["residual"].to(self.model.dtype))
        unembedding = Unembedding(
            weight=self.model.get_output_embeddings().weight.detach().float().cpu(),
            gain=gain, bias=bias, scale=scale, centers=centers,
        )
        return Decomposition(unembedding=unembedding, **joined)
