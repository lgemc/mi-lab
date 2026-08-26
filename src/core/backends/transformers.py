from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from ..adapter import DTYPES, Decomposition, ModelAdapter, Unembedding, register_backend
from ..config import ConfigError, ModelConfig, Position

"""
The HuggingFace backend: the correctness oracle every faster backend is
checked against. It works on any causal LM on the hub and it is slow, and
those two facts are the same fact.

Everything here is architecture knowledge, and it is deliberately quarantined
in this file. `_blocks`, `_attention_projection`, `_mlp` and `_final_norm` are
the whole of it: four lookups that try the names the major model families use
and raise with the module's own type when none matches. Teaching this backend
a new architecture is editing those four lists, and nothing above this layer
learns that a model was a GPT-2 rather than a Pythia.

Two sites carry every operation. The residual stream is read and written at a
decoder block's output; heads are read and written at the input to the
projection that mixes them back in. Capture and patch address the *same* two
sites, which is what makes writing back what was already there exactly a
no-op -- and every causal number in this framework is a difference against
that no-op.

Capture uses forward hooks and deliberately does *not* use
output_hidden_states, which looks like the obvious way to do this and is
subtly not the same thing: that tuple's last entry is the final layer norm
applied to the residual stream, not the residual stream, so capturing at depth
1.0 would silently hand back a different quantity than capturing anywhere
else. Transformers 5 also implements output_hidden_states with its own
recorder hooks, which can observe a block's output before an intervention hook
has finished with it.

This module is named after the library it wraps, which is also the registry
key `configs/*.yaml` names. `import transformers` inside it still resolves to
the installed library: Python 3 imports are absolute unless written with a
leading dot.

A common pipe could be: load_adapter | capture | head_outputs | patch
"""

def _first_attribute(root, paths: Sequence[str]):
    """Follow the first dotted attribute path that resolves, or return None"""
    for path in paths:
        module = root
        for attribute in path.split("."):
            module = getattr(module, attribute, None)
            if module is None:
                break
        if module is not None:
            return module
    return None

def _blocks(model) -> torch.nn.ModuleList:
    """Find the list of decoder blocks, whatever this architecture calls it

    Every residual stream hook site in this backend is a block boundary, so
    this is the only piece of architecture knowledge probing needs.
    """
    found = _first_attribute(model, ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"))
    if found is None:
        raise ConfigError(f"cannot find the decoder blocks of {type(model).__name__}; teach _blocks its layout")
    return found

def _attention_projection(block, index: int):
    """The linear that mixes the heads back into the residual stream

    This is the hinge every head-level operation turns on. Its *input* is the
    heads' outputs laid end to end -- n_heads contiguous slices of d_head --
    so reading it splits the heads apart and writing it patches one head
    without reimplementing attention. Its *output* is the whole attention
    write, so nothing downstream has to be told a patch happened.
    """
    found = _first_attribute(
        block, ("attn.c_proj", "self_attn.o_proj", "attention.dense", "attn.out_proj", "self_attention.dense")
    )
    if found is None:
        raise ConfigError(
            f"cannot find the attention output projection of block {index} "
            f"({type(block).__name__}); teach _attention_projection its layout"
        )
    return found

def _mlp(block, index: int):
    """The submodule whose output is this block's MLP write into the residual stream"""
    found = _first_attribute(block, ("mlp", "feed_forward", "ffn"))
    if found is None:
        raise ConfigError(f"cannot find the MLP of block {index} ({type(block).__name__}); teach _mlp its layout")
    return found

def _final_norm(model):
    """The normalization sitting between the last block and the unembedding"""
    found = _first_attribute(
        model,
        (
            "transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm",
            "model.decoder.final_layer_norm", "model.final_layernorm", "transformer.norm_f",
        ),
    )
    if found is None:
        raise ConfigError(f"cannot find the final norm of {type(model).__name__}; teach _final_norm its layout")
    return found

def _last_real(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise pick of the final non-padding position from a [batch, seq, ...] tensor

    With right padding the final real token is at mask.sum() - 1, which is the
    last column only for the longest prompt in the batch. Reading column -1
    instead is the bug that makes a short prompt's answer come out of padding.
    """
    index = (mask.sum(dim=1) - 1).long()
    return sequence[torch.arange(sequence.shape[0], device=sequence.device), index]

def _normalizer(norm, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Read a final norm's frozen divisor and elementwise gain off one forward pass

    Whether the norm subtracts the mean is settled by asking the module rather
    than by recognizing its class: a LayerNorm is invariant to adding the same
    constant to every coordinate and an RMSNorm is not, so two calls decide it
    for a normalization this framework has never seen.
    """
    with torch.no_grad():
        probe = residual[:1]
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

@dataclass
class _Patch:
    """Donor activations to write in, and which rows of them the current chunk needs

    A patched forward pass still runs in batch_size chunks, and the donor
    tensors cover the whole prompt list, so every hook has to slice the rows
    belonging to the chunk it is firing inside. Handing a hook the full donor
    would patch the wrong prompts as soon as one batch becomes two.
    """
    residual: Dict[int, torch.Tensor] = field(default_factory=dict)
    heads: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    start: int = 0
    stop: int = 0

    def select(self, start: int, rows: int) -> None:
        self.start, self.stop = start, start + rows

    def rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.start : self.stop]

class TransformersAdapter:
    """A CircuitAdapter over a HuggingFace causal LM

    Every residual stream site is a decoder block's output, read and written
    with forward hooks; every head-level site is the input to the projection
    that mixes the heads. Capture and patch address the same two sites, so
    what a patch writes in is exactly what a capture would have read out.
    """

    def __init__(self, cfg: ModelConfig, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = _blocks(model)
        self.projections = [_attention_projection(block, index) for index, block in enumerate(self.blocks)]
        self.mlps = [_mlp(block, index) for index, block in enumerate(self.blocks)]
        self.cfg = cfg.with_sizes(
            n_layers=len(self.blocks),
            d_model=model.config.hidden_size,
            n_heads=getattr(model.config, "num_attention_heads", None),
        )
        self._patch: Optional[_Patch] = None

    def layer(self, frac: Optional[float] = None) -> int:
        """Resolve a depth fraction to an absolute layer index"""
        return self.cfg.layer(frac)

    def _resolve_layers(self, layers: Optional[Sequence[int]]) -> List[int]:
        """Default to the config's probe layer, and reject indices this model does not have"""
        if layers is None:
            return [self.layer()]
        layers = list(layers)
        if not layers:
            raise ConfigError("capture needs at least one layer")
        out_of_range = [index for index in layers if not 0 <= index < self.cfg.n_layers]
        if out_of_range:
            raise ConfigError(f"layers {out_of_range} are outside the {self.cfg.n_layers} layers of '{self.cfg.id}'")
        return layers

    def _chunks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Walk the prompts in batch_size chunks, telling any active patch which rows it is on"""
        for start in range(0, len(input_ids), self.cfg.batch_size):
            ids = input_ids[start : start + self.cfg.batch_size].to(self.model.device)
            mask = attention_mask[start : start + self.cfg.batch_size].to(self.model.device)
            if self._patch is not None:
                self._patch.select(start, ids.shape[0])
            yield ids, mask

    @contextmanager
    def _record(self, layers: Sequence[int]) -> Iterator[Dict[int, torch.Tensor]]:
        """Collect the residual stream leaving each requested block

        Recording hooks are registered after any steering hook, and PyTorch
        runs forward hooks in registration order while threading the modified
        output through them, so capturing at a steered layer sees the
        intervention rather than the value it replaced.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden.detach()
            return hook

        for index in layers:
            handles.append(self.blocks[index].register_forward_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _encode(self, prompts: Sequence[str], padding_side: str):
        """Tokenize every prompt to one uniform length, so chunks stay comparable"""
        self.tokenizer.padding_side = padding_side
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        return encoded["input_ids"], encoded["attention_mask"]

    def capture(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        position: Position = Position.LAST,
    ) -> torch.Tensor:
        """Capture the residual stream at the given layers, batched per cfg.batch_size

        Chunks are accumulated on CPU: on a unified-memory machine the device
        and host pools are the same silicon, but on a discrete GPU holding a
        thousand prompts' activations resident is how capture OOMs.
        """
        if not prompts:
            raise ConfigError("capture needs at least one prompt")
        position = Position(position)
        layers = self._resolve_layers(layers)
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        chunks = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with self._record(layers) as captured, torch.no_grad():
                self.model(ids, attention_mask=mask, use_cache=False)
            stacked = torch.stack([captured[index] for index in layers], dim=1)
            chunks.append(_reduce(stacked, mask, position).float().cpu())
        return torch.cat(chunks, dim=0)

    def generate(self, prompts: Sequence[str], max_new_tokens: Optional[int] = None, **kwargs) -> List[str]:
        """Greedily continue each prompt, returning only the new text"""
        input_ids, attention_mask = self._encode(prompts, padding_side="left")
        completions = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with torch.no_grad():
                generated = self.model.generate(
                    ids,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )
            completions.extend(self.tokenizer.batch_decode(generated[:, ids.shape[1] :], skip_special_tokens=True))
        return completions

    @contextmanager
    def steer(self, layer: int, vector: torch.Tensor, strength: float) -> Iterator[None]:
        """Add a direction to the residual stream leaving `layer`, for the duration of the block

        The vector is normalized and rescaled to the mean activation norm seen
        at that layer in this very forward pass, so a strength of 1.0 means the
        same intervention size on GPT-2 small as on a 27B. At strength 0 no hook
        fires at all, which keeps a zero-strength run byte-identical to no
        steering -- the check that tells you the hook is where you think it is.
        """
        (layer,) = self._resolve_layers([layer])
        if strength == 0:
            yield
            return

        direction = vector.to(self.model.device, self.model.dtype)
        direction = direction / direction.norm()

        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            scale = strength * hidden.norm(dim=-1, keepdim=True).mean()
            steered = hidden + scale * direction
            return (steered, *output[1:]) if is_tuple else steered

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # ----------------------------------------------------------------- circuits

    def single_token(self, text: str) -> int:
        """The id of a string that is exactly one token, or an error naming what it split into

        The whole IOI design rests on this: names have to be one token each or
        the answer is spread over several logits and a logit difference stops
        measuring the thing it is named after. Failing loudly here is what
        keeps that from being discovered as a mysteriously flat result.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            pieces = self.tokenizer.convert_ids_to_tokens(ids)
            raise ConfigError(
                f"'{text}' is {len(ids)} tokens on '{self.cfg.id}' ({pieces}), not one; "
                "single-token names are what makes a logit difference readable"
            )
        return int(ids[0])

    def tokens(self, prompt: str) -> List[str]:
        """The prompt as the strings the model actually sees, for labelling an axis"""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [self.tokenizer.decode([token]) for token in ids]

    def logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Next-token logits at each prompt's final real token, as [batch, vocab]"""
        if not prompts:
            raise ConfigError("logits needs at least one prompt")
        input_ids, attention_mask = self._encode(prompts, padding_side="right")
        chunks = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with torch.no_grad():
                output = self.model(ids, attention_mask=mask, use_cache=False).logits
            chunks.append(_last_real(output, mask).float().cpu())
        return torch.cat(chunks, dim=0)

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

    @contextmanager
    def _record_heads(self, layers: Sequence[int]) -> Iterator[Dict[int, torch.Tensor]]:
        """Collect what each head wrote, before the projection mixed them together"""
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(module, args):
                captured[index] = args[0].detach()
            return hook

        for index in layers:
            handles.append(self.projections[index].register_forward_pre_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _split_heads(self, merged: torch.Tensor) -> torch.Tensor:
        """Split a [..., n_heads * d_head] tensor into [..., n_heads, d_head]"""
        return merged.reshape(*merged.shape[:-1], self.cfg.n_heads, merged.shape[-1] // self.cfg.n_heads)

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

    @contextmanager
    def patch(
        self,
        residual: Optional[Dict[int, torch.Tensor]] = None,
        heads: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    ) -> Iterator[None]:
        """Overwrite activations with values from another run, for the duration of the block

        residual maps a layer to a [batch, seq, d_model] replacement for the
        stream *leaving* that block -- the same site capture() reads, so a
        capture from one run drops straight into a patch of another. heads maps
        a layer to {head: [batch, seq, d_head]}, replacing only those heads and
        leaving the layer's others alone.

        Donor tensors cover the whole prompt list; the hooks slice out the rows
        of whichever chunk is running. Patches do not nest: a second one while
        the first is open would silently decide an ordering nobody chose.
        """
        if self._patch is not None:
            raise ConfigError("a patch is already active; build one patch describing every site instead of nesting")

        active = _Patch(residual=dict(residual or {}), heads=dict(heads or {}))
        for index in list(active.residual) + list(active.heads):
            self._resolve_layers([index])
        handles = []

        def make_residual_hook(index: int):
            def hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                donor = active.rows(active.residual[index]).to(hidden.device, hidden.dtype)
                return (donor, *output[1:]) if is_tuple else donor
            return hook

        def make_head_hook(index: int):
            def hook(module, args):
                merged = self._split_heads(args[0]).clone()
                for head, value in active.heads[index].items():
                    merged[:, :, head] = active.rows(value).to(merged.device, merged.dtype)
                return (merged.reshape(*args[0].shape), *args[1:])
            return hook

        for index in active.residual:
            handles.append(self.blocks[index].register_forward_hook(make_residual_hook(index)))
        for index in active.heads:
            handles.append(self.projections[index].register_forward_pre_hook(make_head_hook(index)))

        self._patch = active
        try:
            yield
        finally:
            self._patch = None
            for handle in handles:
                handle.remove()

def _reduce(stacked: torch.Tensor, mask: torch.Tensor, position: Position) -> torch.Tensor:
    """Collapse a [batch, layer, seq, d_model] capture to the requested position

    Padding is never averaged into a MEAN and never mistaken for the LAST
    token: with right padding the final real token is at mask.sum() - 1, which
    is not the final column whenever a prompt is shorter than its batch.
    """
    if position is Position.ALL:
        return stacked
    if position is Position.MEAN:
        weights = mask[:, None, :, None].to(stacked.dtype)
        return (stacked * weights).sum(dim=2) / weights.sum(dim=2)
    last = (mask.sum(dim=1) - 1)[:, None, None, None].expand(-1, stacked.shape[1], -1, stacked.shape[3])
    return stacked.gather(2, last).squeeze(2)

@register_backend("transformers")
def _build_transformers(cfg: ModelConfig) -> ModelAdapter:
    """Load a HuggingFace causal LM in eval mode, with a pad token guaranteed"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    if tokenizer.pad_token is None:
        # GPT-2 and friends ship no pad token, and batching needs one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=DTYPES[cfg.dtype])
    model.to(cfg.device)
    model.eval()
    return TransformersAdapter(cfg, model, tokenizer)

