"""
The residual stream as a graph: who wrote what, and what each reader saw.

Because the stream is a sum, a transformer is already a graph and this is its
adjacency. `residual_sources` records every per-position write -- heads, MLPs,
projection biases, the embedding -- alongside what the two destinations in each
block actually read, and `residual_remainder` is the receipt that the two
agree. Nothing built on top of this is worth anything if that number is not
float noise: an edge intervention swaps terms in that sum, so a sum that does
not reproduce the forward pass is a graph of a different model.

`_check_edge` is the one authority on which edges exist -- attention at layer L
reads everything written before block L, the MLP at L also reads L's own
attention, and neither can read L's MLP or anything later. `edges` asks it
rather than restating the rule, so the two cannot drift apart, and both
interventions ask it before touching a hook.

A common pipe could be: residual_sources | residual_remainder | edge_patch
"""

import contextlib
from typing import Dict, List, Sequence, Tuple

import torch

from ....core.config import ConfigError
from .layout import _attention_norm, _mlp_norm


class GraphMixin:
    """The writes, the reads, the legal edges between them, and the check that they add up"""

    def _head_writes_at(self, index: int, merged: torch.Tensor) -> torch.Tensor:
        """_head_writes at every position rather than one, as [batch, seq, head, d_model]

        Same construction and the same reason for it: run the projection with
        every other head zeroed rather than slicing its weight, and subtract
        the bias back out because it is written once per layer and not once per
        head. The only difference is that the position axis is kept, which is
        what an edge needs and what a final-token attribution does not.
        """
        batch, seq, width = merged.shape
        selector = torch.eye(self.cfg.n_heads, dtype=merged.dtype, device=merged.device)
        split = self._split_heads(merged)
        isolated = (split[:, :, None] * selector[None, None, :, :, None]).reshape(
            batch, seq, self.cfg.n_heads, width
        )
        with torch.no_grad():
            projection = self.projections[index]
            bias = projection(merged.new_zeros(1, width))
            return projection(isolated) - bias

    def residual_sources(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Every per-position write into the residual stream, and what each destination read

        The residual stream is a sum, so a transformer is already a graph and
        this is its adjacency: `heads`, `mlps`, `biases` and `embedding` are
        what each source wrote at each position, and `attention_in`/`mlp_in`
        are what the two destinations in each block actually saw.

        The receipt is that they agree. A destination's input has to equal the
        embedding plus every upstream write, and `residual_remainder` reports
        the gap. Nothing here is worth anything if that number is not float
        noise: an edge ablation swaps terms in that sum, so a sum that does not
        reproduce the forward pass is a graph of a different model.

        One forward pass plus n_heads projection calls per layer, all of which
        are position-wise and cheap next to attention itself.
        """
        if not prompts:
            raise ConfigError("residual_sources needs at least one prompt")
        layers = list(range(self.cfg.n_layers))
        input_ids, attention_mask = self._encode(prompts, padding_side="right")
        collected: Dict[str, List[torch.Tensor]] = {
            name: [] for name in ("embedding", "heads", "biases", "mlps", "attention_in", "mlp_in")
        }

        for ids, mask in self._chunks(input_ids, attention_mask):
            merged: Dict[int, torch.Tensor] = {}
            mlp_out: Dict[int, torch.Tensor] = {}
            attention_in: Dict[int, torch.Tensor] = {}
            mlp_in: Dict[int, torch.Tensor] = {}
            handles = []

            def make(store, index, on_output=False):
                def hook(module, args, output=None):
                    value = output if on_output else args[0]
                    value = value[0] if isinstance(value, tuple) else value
                    store[index] = value.detach()
                return hook

            for index in layers:
                handles.append(self.projections[index].register_forward_pre_hook(make(merged, index)))
                handles.append(self.mlps[index].register_forward_hook(make(mlp_out, index, on_output=True)))
                handles.append(_attention_norm(self.blocks[index], index)
                               .register_forward_pre_hook(make(attention_in, index)))
                handles.append(_mlp_norm(self.blocks[index], index)
                               .register_forward_pre_hook(make(mlp_in, index)))
            try:
                with torch.no_grad():
                    self.model(ids, attention_mask=mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            width = merged[0].shape[-1]
            heads = torch.stack([self._head_writes_at(index, merged[index]) for index in layers], dim=1)
            with torch.no_grad():
                biases = torch.stack([
                    self.projections[index](merged[index].new_zeros(1, width)).expand(
                        merged[index].shape[0], merged[index].shape[1], width)
                    for index in layers], dim=1)
            collected["embedding"].append(attention_in[0].float().cpu())
            collected["heads"].append(heads.permute(0, 1, 3, 2, 4).float().cpu())
            collected["biases"].append(biases.float().cpu())
            collected["mlps"].append(torch.stack([mlp_out[i] for i in layers], dim=1).float().cpu())
            collected["attention_in"].append(
                torch.stack([attention_in[i] for i in layers], dim=1).float().cpu())
            collected["mlp_in"].append(torch.stack([mlp_in[i] for i in layers], dim=1).float().cpu())
        return {name: torch.cat(values, dim=0) for name, values in collected.items()}

    def residual_remainder(self, sources: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """How far the sum of the writes is from what the destinations actually read

        The check `decompose` makes at the final token, made at every position
        and every destination. Float noise means the graph is the model's;
        anything larger means a write is missing or double counted, and every
        edge intervention built on it would be intervening on a fiction.
        """
        heads, mlps = sources["heads"].sum(dim=2), sources["mlps"]
        attention_gap, mlp_gap = 0.0, 0.0
        for layer in range(heads.shape[1]):
            upstream = sources["embedding"].clone()
            for earlier in range(layer):
                upstream = upstream + heads[:, earlier] + sources["biases"][:, earlier] + mlps[:, earlier]
            attention_gap = max(attention_gap, float((upstream - sources["attention_in"][:, layer]).abs().max()))
            here = upstream + heads[:, layer] + sources["biases"][:, layer]
            mlp_gap = max(mlp_gap, float((here - sources["mlp_in"][:, layer]).abs().max()))
        scale = float(sources["attention_in"].abs().max())
        return {"attention_in": attention_gap, "mlp_in": mlp_gap,
                "relative": max(attention_gap, mlp_gap) / scale if scale else float("nan")}

    def edges(self) -> List[Tuple[str, str]]:
        """Every edge a causal residual stream admits, source before destination

        Enumerated here because which sources exist and which destinations can
        read them is architecture knowledge, and this file is where that lives.
        `_check_edge` is the authority on legality; this asks it rather than
        restating the rule, so the two cannot drift apart.
        """
        sources = ["embed"]
        for layer in range(self.cfg.n_layers):
            sources.extend(f"head:{layer}:{head}" for head in range(self.cfg.n_heads))
            sources.append(f"mlp:{layer}")
            sources.append(f"bias:{layer}")
        found = []
        for layer in range(self.cfg.n_layers):
            for destination in (f"attn:{layer}", f"mlp:{layer}"):
                for source in sources:
                    try:
                        self._check_edge(source, destination)
                    except ConfigError:
                        continue
                    found.append((source, destination))
        return found

    def _check_edge(self, source: str, destination: str) -> None:
        """Refuse an edge that does not exist in a causal residual stream"""
        kind, layer = destination.split(":")[0], int(destination.split(":")[1])
        if kind not in ("attn", "mlp"):
            raise ConfigError(f"destination must be attn:L or mlp:L, got '{destination}'")
        if source == "embed":
            return
        parts = source.split(":")
        if parts[0] not in ("head", "mlp", "bias"):
            raise ConfigError(f"source must be embed, head:L:H, mlp:L or bias:L, got '{source}'")
        at = int(parts[1])
        # attention at L reads everything written before block L; the MLP at L also
        # reads L's own attention, and neither can read L's MLP or anything later
        limit = at < layer or (kind == "mlp" and at == layer and parts[0] in ("head", "bias"))
        if not limit:
            raise ConfigError(
                f"'{source}' is not upstream of '{destination}': a residual stream is causal, so this "
                "edge does not exist and ablating it would be ablating nothing while reporting a result"
            )

    def _live_write(self, source: str, merged, mlp_out, embedded, like: torch.Tensor,
                    grad: bool = False) -> torch.Tensor:
        """What a source wrote in the run currently executing

        `grad` keeps the reconstruction inside the autograd graph. False is
        right for `edge_patch`, whose captures belong to a frozen
        counterfactual; True is required by `edge_gate`, where a gate on an
        early edge changes the residual every later component reads, and
        detaching silently reduces the gradient to its direct-path term. A
        finite-difference check put that error at 54x, and sign-flipped, on
        an edge into layer 0.
        """
        if source == "embed":
            return embedded[0].to(like.device, like.dtype)
        parts = source.split(":")
        layer = int(parts[1])
        if parts[0] == "mlp":
            return mlp_out[layer].to(like.device, like.dtype)
        width = merged[layer].shape[-1]
        with contextlib.nullcontext() if grad else torch.no_grad():
            bias = self.projections[layer](merged[layer].new_zeros(1, width))
            if parts[0] == "bias":
                return bias.expand_as(like).to(like.device, like.dtype)
            head = int(parts[2])
            selector = torch.zeros(self.cfg.n_heads, dtype=merged[layer].dtype, device=merged[layer].device)
            selector[head] = 1.0
            isolated = (self._split_heads(merged[layer]) * selector[None, None, :, None]).reshape(
                merged[layer].shape)
            return (self.projections[layer](isolated) - bias).to(like.device, like.dtype)

    def _off_write(self, source: str, off: Dict[str, torch.Tensor]) -> torch.Tensor:
        """What a source wrote in the counterfactual run this ablation falls back to"""
        if source == "embed":
            return off["embedding"]
        parts = source.split(":")
        layer = int(parts[1])
        if parts[0] == "mlp":
            return off["mlps"][:, layer]
        if parts[0] == "bias":
            return off["biases"][:, layer]
        return off["heads"][:, layer, int(parts[2])]
