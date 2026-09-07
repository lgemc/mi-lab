"""
Head-level reads: where a head looked, what it wrote, and what moving it does.

Everything here turns on one hinge, the projection that mixes the heads back
into the residual stream. Its input is the heads laid end to end -- n_heads
contiguous slices of d_head -- so reading it splits the heads apart, writing it
patches one head without reimplementing attention, and differentiating at it
gives the gradient of the readout with respect to a head's output. Three
measurements, one site, which is what lets an activation from one run and a
gradient from another multiply into an estimate of a patch nobody paid for.

Attention patterns are the exception that needs the model changed rather than
hooked: the fast kernels never build the [query, key] matrix, so reading a
pattern means running eager for the duration.

A common pipe could be: head_outputs | gradients | eap
"""

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple

import torch

from ....core.config import ConfigError
from ....core.readout import Readout, require_readout
from .positions import _last_real


class HeadMixin:
    """Attention patterns, head outputs, and gradients taken at the same site"""

    @contextmanager
    def _eager_attention(self) -> Iterator[None]:
        """Run with the attention implementation that materializes its weights

        The fast kernels never build the [query, key] matrix, so asking a model
        running under sdpa or flash attention for its patterns hands back an
        empty tuple rather than an error. Switching to eager for the duration
        is the price of seeing where a head looked.
        """
        previous = getattr(self.model.config, "_attn_implementation", None)
        if previous == "eager":
            yield
            return
        try:
            self.model.set_attn_implementation("eager")
        except (AttributeError, ValueError) as error:
            raise ConfigError(
                f"cannot switch '{self.cfg.id}' to eager attention, so its patterns cannot be read: {error}"
            ) from error
        try:
            yield
        finally:
            self.model.set_attn_implementation(previous)

    def attention(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Attention weights per head, as [batch, layer, head, query, key]

        Row q of a head's matrix is where the token at position q looked. The
        row worth almost all of the attention in a circuit study is the last
        one, because that is the position the answer is read from.
        """
        if not prompts:
            raise ConfigError("attention needs at least one prompt")
        layers = self._resolve_layers(layers if layers is not None else range(self.cfg.n_layers))
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        chunks = []
        with self._eager_attention():
            for ids, mask in self._chunks(input_ids, attention_mask):
                with torch.no_grad():
                    patterns = self.model(
                        ids, attention_mask=mask, use_cache=False, output_attentions=True
                    ).attentions
                if not patterns:
                    raise ConfigError(f"'{self.cfg.id}' returned no attention weights under eager attention")
                chunks.append(torch.stack([patterns[index] for index in layers], dim=1).float().cpu())
        return torch.cat(chunks, dim=0)

    def head_outputs(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Each head's output before the projection, as [batch, layer, head, seq, d_head]

        This is the donor a head patch reads from, and the reason patching a
        head does not mean reimplementing attention: the projection's input is
        the heads laid end to end, so a head is a slice.
        """
        if not prompts:
            raise ConfigError("head_outputs needs at least one prompt")
        layers = self._resolve_layers(layers if layers is not None else range(self.cfg.n_layers))
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        chunks = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with self._record_heads(layers) as captured, torch.no_grad():
                self.model(ids, attention_mask=mask, use_cache=False)
            stacked = torch.stack([self._split_heads(captured[index]) for index in layers], dim=1)
            chunks.append(stacked.permute(0, 1, 3, 2, 4).float().cpu())
        return torch.cat(chunks, dim=0)

    def gradients(
        self,
        inputs: Sequence[str],
        readout: Readout,
        layers: Optional[Sequence[int]] = None,
        toward: Optional[Sequence[str]] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """The readout's gradient at each head's output, as [batch, layer, head, seq, d_head]

        The gradient is taken at the projection's input -- the same site
        head_outputs reads and patch writes -- so a gradient and an activation
        from two runs multiply together into an estimate of what patching that
        head would have done, without the forward pass per head that measuring
        it would cost.

        The graph is kept rather than cut at each site: detaching there would
        silently delete every path an earlier head has to the answer *through*
        this layer's attention, leaving a gradient that looks fine and answers
        a different question. torch.autograd.grad asks only for these tensors,
        so no parameter gradient is accumulated on the way past.

        `toward` and `alpha` are what integrated gradients needs and a single
        gradient does not. With `toward` given, the forward pass runs from the
        embeddings mixed as (1 - alpha) * prompts + alpha * toward rather than
        from `prompts` alone, so a caller can walk the straight line between two
        inputs and average what it finds. alpha=0 reproduces `prompts` exactly
        and the default of 1.0 is never reached without `toward`, so the
        one-gradient path is untouched.

        The two prompt sets must tokenize to the same length, because a mixture
        of embeddings at differing lengths is a mixture of different positions.
        Circuit tasks in this repo are length-aligned by construction, and this
        raises rather than broadcasts when one is not.
        """
        prompts = list(inputs)
        if not prompts:
            raise ConfigError("gradients needs at least one prompt")
        require_readout(readout, by=f"a gradient at {self.cfg.id}'s head outputs")
        layers = self._resolve_layers(layers if layers is not None else range(self.cfg.n_layers))
        input_ids, attention_mask = self._encode(prompts, padding_side="right")
        donor_ids = None
        if toward is not None:
            if len(toward) != len(prompts):
                raise ConfigError(
                    f"{len(prompts)} prompts but {len(toward)} to interpolate toward; they index the same batch"
                )
            donor_ids, donor_mask = self._encode(toward, padding_side="right")
            if donor_ids.shape != input_ids.shape:
                raise ConfigError(
                    f"interpolating needs both prompt sets at one length, but they tokenize to "
                    f"{tuple(input_ids.shape)} and {tuple(donor_ids.shape)}. A mixture of embeddings at "
                    "two lengths mixes different positions."
                )
            if not torch.equal(attention_mask, donor_mask):
                raise ConfigError("the two prompt sets pad differently, so their positions do not correspond")
        embed = self.model.get_input_embeddings()

        chunks = []
        start = 0
        for ids, mask in self._chunks(input_ids, attention_mask):
            rows = slice(start, start + ids.shape[0])
            start += ids.shape[0]
            with self._record_heads(layers, detach=False) as captured, torch.enable_grad():
                if donor_ids is None:
                    output = self.model(ids, attention_mask=mask, use_cache=False).logits
                else:
                    here = embed(ids)
                    there = embed(donor_ids[rows].to(self.model.device))
                    mixed = here + alpha * (there - here)
                    output = self.model(inputs_embeds=mixed, attention_mask=mask, use_cache=False).logits
                final = _last_real(output, mask)
                # summed over the batch because each row's answer depends on its own
                # activations alone, so one backward pass carries every row's gradient.
                # The readout is narrowed to this chunk's rows first: it holds one
                # answer per row of the whole batch, and scoring a chunk against
                # another chunk's answers is the bug `_Patch.select` exists to stop.
                objective = readout.select(rows)(final).sum()
                gradients = torch.autograd.grad(objective, [captured[layer] for layer in layers])
            stacked = torch.stack([self._split_heads(gradient) for gradient in gradients], dim=1)
            chunks.append(stacked.permute(0, 1, 3, 2, 4).float().cpu())
        return torch.cat(chunks, dim=0)

    def gate_gradients(
        self,
        inputs: Sequence[str],
        donor: torch.Tensor,
        readout: Readout,
        gates: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
    ) -> Tuple[float, torch.Tensor]:
        """The readout with each head interpolated toward a donor by its gate, and d(that)/d(gates)

        Each head's output becomes donor + gate * (clean - donor), so a gate of
        1 leaves the head alone and a gate of 0 ablates it to the donor exactly.
        That is the same site head_outputs reads and patch writes, so a gate of
        0 here and a patch there are the same intervention; what this adds is
        that the interpolation is differentiable in the gate, which is what a
        method that *learns* which heads to keep needs and no amount of scoring
        them one at a time provides.

        The donor is a counterfactual activation rather than a mean or a zero,
        because a mean removes only what varied across the batch and a zero
        removes a direction the model never sees. It is passed in rather than
        captured here so the caller decides what "off" means and the choice is
        visible in the method that made it.

        Returns the objective and its gradient rather than keeping a graph
        alive across the call, so a caller can run an optimizer loop without
        holding one forward pass per step in memory.

        `gates` is [layer, head] over the layers asked for, and is broadcast
        over batch and position: a gate is a statement about a head, not about
        a head at a token.
        """
        prompts = list(inputs)
        if not prompts:
            raise ConfigError("gate_gradients needs at least one prompt")
        require_readout(readout, by=f"a gate gradient at {self.cfg.id}'s head outputs")
        layers = self._resolve_layers(layers if layers is not None else range(self.cfg.n_layers))
        if gates.shape != (len(layers), self.cfg.n_heads):
            raise ConfigError(
                f"gates are {tuple(gates.shape)} but {len(layers)} layers x {self.cfg.n_heads} heads "
                "were asked for; a gate names a head"
            )
        if donor.shape[1] != len(layers):
            raise ConfigError(
                f"donor covers {donor.shape[1]} layers and {len(layers)} were asked for; the donor is "
                "the activation a gate of zero falls back to, so it has to cover the same sites"
            )
        input_ids, attention_mask = self._encode(prompts, padding_side="right")
        live = gates.detach().clone().requires_grad_(True)

        total = 0.0
        accumulated = torch.zeros_like(live)
        start = 0
        for ids, mask in self._chunks(input_ids, attention_mask):
            rows = slice(start, start + ids.shape[0])
            start += ids.shape[0]
            handles = []

            def make_hook(position: int, chunk: slice):
                # position indexes `layers`. The donor rows are bound here rather
                # than read from the loop variable, for the reason _Patch.select
                # exists: a hook that closes over the enclosing scope patches
                # whichever chunk happened to run last.
                def hook(module, args):
                    heads = self._split_heads(args[0])
                    off = donor[chunk, position].permute(0, 2, 1, 3).to(heads.device, heads.dtype)
                    gate = live[position].reshape(1, 1, -1, 1).to(heads.device)
                    gated = off + gate * (heads - off)
                    return (gated.reshape(*args[0].shape), *args[1:])
                return hook

            for position, index in enumerate(layers):
                handles.append(self.projections[index].register_forward_pre_hook(make_hook(position, rows)))
            try:
                with torch.enable_grad():
                    output = self.model(ids, attention_mask=mask, use_cache=False).logits
                    final = _last_real(output, mask)
                    objective = readout.select(rows)(final).sum()
                    (gradient,) = torch.autograd.grad(objective, [live])
            finally:
                for handle in handles:
                    handle.remove()
            total += float(objective.detach())
            accumulated += gradient.detach().cpu()
        return total / len(prompts), accumulated / len(prompts)
