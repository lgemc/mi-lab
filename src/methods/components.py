"""The names a knockout study gives to the parts of a model, and the sets it builds from them.

Three kinds of component, each a string so it can be a JSON key, a log line
and a command-line argument without a conversion in between:

    mlp:L       the whole MLP block of layer L
    heads:L     every attention head of layer L, as one unit
    head:L:H    one attention head

Every ablation, cost and control script in the study speaks this vocabulary,
and until this module they each parsed it themselves. The parsing is trivial;
what is not trivial is the set algebra around it, and that is what lives here:
which layers a *band* of depth fractions resolves to on the model actually
loaded, which components make up a sweep plan, what the atomic pool a random
control may draw from is, what the complement of a circuit is at a given
granularity, and when two components are the same thing said twice.

Layers are a depth band rather than indices, which is invariant 1 and which
the absolute `range(27, 36)` it replaces could not honour: 27-35 exists on a
36-layer model and nowhere else, so every script that named it was silently
bound to one checkpoint. 0.75-1.0 is the same nine layers there and the
proportionally equivalent band anywhere else.

A common pipe could be: band | plan | parse | complement | redundant
"""

from typing import Iterable, List, Optional, Sequence, Tuple

from .circuits import CircuitError

KINDS = ("mlp", "heads", "head")

# The candidate band of the translation study as a depth fraction: the top
# quarter of the stack, where the neuron scan found the language-specific
# activity concentrated.
CANDIDATE_BAND = (0.75, 1.0)

class ComponentError(CircuitError):
    """A component name or set that does not describe this model"""

def parse(cid: str) -> Tuple[str, int, Optional[int]]:
    """'mlp:31' | 'heads:31' | 'head:31:7' -> (kind, layer, head)"""
    parts = cid.split(":")
    if parts[0] not in KINDS:
        raise ComponentError(f"unknown component '{cid}'; kinds are {', '.join(KINDS)}")
    if (parts[0] == "head") != (len(parts) == 3) or len(parts) < 2:
        raise ComponentError(f"malformed component '{cid}'; expected mlp:L, heads:L or head:L:H")
    try:
        layer = int(parts[1])
        head = int(parts[2]) if len(parts) == 3 else None
    except ValueError as error:
        raise ComponentError(f"malformed component '{cid}': {error}") from None
    return parts[0], layer, head

def name(kind: str, layer: int, head: Optional[int] = None) -> str:
    """The inverse of `parse`"""
    if kind == "head":
        if head is None:
            raise ComponentError("a single head needs a head index")
        return f"head:{layer}:{head}"
    if kind not in KINDS:
        raise ComponentError(f"unknown component kind '{kind}'")
    return f"{kind}:{layer}"

def band(cfg, fractions: Tuple[float, float] = CANDIDATE_BAND) -> List[int]:
    """A depth band resolved against the model actually loaded, both ends inclusive"""
    low, high = fractions
    return list(range(cfg.layer(low), cfg.layer(high) + 1))

def layer_components(layers: Iterable[int]) -> List[str]:
    """Every whole-layer component of the given layers: the MLPs first, then the head groups"""
    layers = list(layers)
    return [name("mlp", layer) for layer in layers] + [name("heads", layer) for layer in layers]

def head_components(layers: Iterable[int], n_heads: int) -> List[str]:
    """Every single head of the given layers, in layer order"""
    return [name("head", layer, head) for layer in layers for head in range(n_heads)]

def atomic_components(layers: Iterable[int], n_heads: int) -> List[str]:
    """The pool a random control draws from: one MLP or one head, never a group

    `heads:L` is a group of atoms and would make a draw coarser than the
    lattice it is matched on.
    """
    layers = list(layers)
    return [name("mlp", layer) for layer in layers] + head_components(layers, n_heads)

def plan(group: str, layers: Sequence[int], n_heads: int) -> List[str]:
    """The components a sweep group walks, in the order it should walk them

    `layers` is the whole-layer sweep. `heads-upper` and `heads-lower` are the
    single heads of the band's upper and lower halves; the upper half runs in
    descending depth so a truncated sweep has done the deepest layers first,
    which on the model the study began on was also the neuron-density order.
    """
    layers = list(layers)
    if group == "layers":
        return layer_components(layers)
    if group == "heads-upper":
        return head_components(reversed(layers[len(layers) // 2:]), n_heads)
    if group == "heads-lower":
        return head_components(layers[:len(layers) // 2], n_heads)
    raise ComponentError(f"unknown group '{group}'; groups are layers, heads-upper, heads-lower")

def complement(components: Iterable[str], n_layers: int) -> List[str]:
    """Every whole-layer component the given set does not name

    Ablating this *is* running the circuit alone, at the granularity the sweep
    worked in: whole MLPs and whole attention layers.
    """
    keep = set(components)
    return [cid for cid in layer_components(range(n_layers)) if cid not in keep]

def redundant(cid: str, chosen: Iterable[str]) -> bool:
    """Whether `cid` is already inside one of `chosen`, or contains one of them

    A head and the head group of its own layer are the same thing at two
    granularities; a greedy set that took both would count the layer twice.
    """
    kind, layer, _ = parse(cid)
    for other in chosen:
        other_kind, other_layer, _ = parse(other)
        if layer != other_layer:
            continue
        if kind == other_kind or {kind, other_kind} == {"head", "heads"}:
            return True
    return False

def heads_named(components: Iterable[str], n_heads: int) -> dict:
    """{layer: sorted head indices} covered by the head-level components, groups expanded"""
    by_layer: dict = {}
    for cid in components:
        kind, layer, head = parse(cid)
        if kind == "heads":
            by_layer.setdefault(layer, set()).update(range(n_heads))
        elif kind == "head":
            by_layer.setdefault(layer, set()).add(head)
    return {layer: sorted(heads) for layer, heads in by_layer.items()}

def mlps_named(components: Iterable[str]) -> List[int]:
    """The layers whose MLP a component set names, in order of first mention"""
    layers: List[int] = []
    for cid in components:
        kind, layer, _ = parse(cid)
        if kind == "mlp" and layer not in layers:
            layers.append(layer)
    return layers
