"""EAP over the components this study ablates: one scoring primitive, and the three choices around it.

Phase 1b ranks components by knocking each one out and regenerating the
corpus: 306 components, 7,575 seconds, one forward pass per sentence per
component. Attribution patching (Nanda 2023; Syed et al. 2023 per edge)
answers the same question -- *how much does the metric move if I intervene
here* -- with a first-order expansion instead of the intervention, at a cost
that does not depend on how many components are being scored. Here that is
one clean pass and one backward pass for the whole lattice.

The estimate is the one the study's own intervention defines. Mean ablation
sets a component's write to its counterfactual mean, so the quantity being
approximated is the metric's change under `a := mean`, and its first-order
term is

    score(c) = (a_clean(c) - a_mean(c)) . d(metric)/d(a(c))

summed over positions and coordinates, positive when ablating *costs* the
model metric -- the sign phase 1b's `dbleu` already has, so the two rankings
are comparable without either being flipped to match.

EAP is not a method that competes with the sweep. It is the scoring engine,
and everything interesting is in the three choices made around it. Each is a
parameter here rather than a hardcoded assumption, because which of them
changes the answer is a measurement and not something to be assumed --- on
the 8B translation band it was the first that mattered most and the second
that mattered only once its two sides stopped asking the same question
(`docs/methodology.tex`, Attribution as the Discovery Method):

  `steps`        gradient x delta is a first-order term, and a first order
                 term is wrong exactly where the metric is flat: a component
                 behind a saturated softmax scores zero however much it
                 matters. `steps > 1` integrates the gradient along the path
                 from the mean-ablated model to the clean one instead of
                 reading it at one end (Hanna et al. 2403.17806 for the
                 idea; see `_mixed` for what path this actually takes, which
                 is not the paper's and says so).
  `metric`       `target` is the likelihood of one continuation and `kl` is
                 the whole next-token distribution. Zhang & Nanda (2024)
                 report that the circuit you find changes substantially with
                 this choice, and this repo has already paid for it once: the
                 sheaf runs whose faithfulness was two logits where the
                 paper's was the whole vocabulary. It is a parameter so that
                 the difference is a measurement rather than a default.
  `granularity`  `node` scores a component: what does the model lose if this
                 head is gone *everywhere*. `edge` scores a (source,
                 destination) pair: what does the model lose if this one
                 reader stops seeing this head. A source that matters through
                 one path and not another is invisible to the first and
                 separable by the second, and node attribution is exactly the
                 sum of a source's edges -- `Attribution.to_nodes` performs
                 that sum, which is also how the two granularities check each
                 other.

What is *not* a parameter is the base: the units scored are the study's own
`mlp:L` / `head:L:H` lattice, the same ones `knockout.ablate` replaces and
`Means` holds a mean of. That is the deliberate limit of this module. The
open question in the field is which base to run this primitive on -- features,
learned parameter components, post-hoc weight factors -- and the answer here
is "the one phase 1b ablates", because the point of this module is to be
checkable against phase 1b's 7,575 seconds of ground truth.

A common pipe could be: teacher_forced | attribute | ranked | agreement
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..core.metrics import jaccard, measure, spearman
from ..model.passes import Span, differentiable, encode, hooked, scored_positions
from ..telemetry.observe import Progress
from . import components as comp
from .circuits import CircuitError
from .knockout import Means

READOUT = "logits"          # the destination every source has and `adapter.edges()` does not enumerate
Edge = Tuple[str, str]
Key = Any  # a component id for node granularity, an (source, destination) pair for edge

DEFAULT_STEPS = 5           # the IG budget `discovery._eap_ig` uses, kept so the two are comparable
DEFAULT_BATCH = 4           # a backward pass holds every hooked site's activations and their gradients
MAX_LENGTH = 512            # the study's few-shot prompts, matching passes.DEFAULT_MAX_LENGTH


class AttributionError(CircuitError):
    """A metric that cannot be estimated the way it was asked for, or a scope with no counterfactual"""


@dataclass(frozen=True)
class Metric:
    """A differentiable stand-in for the thing the study actually scores

    `flat_at_clean` is the property that decides whether a metric can be used
    with `steps=1` at all, and it is a fact about the metric rather than a
    preference: a metric whose minimum *is* the clean model has a zero
    gradient there, so its first-order term is identically zero for every
    component and the ranking that comes back is noise with a confident shape.
    """
    name: str
    description: str
    score: Callable[..., torch.Tensor]
    needs_reference: bool = False
    flat_at_clean: bool = False


METRICS: Dict[str, Metric] = {}


def register_metric(description: str, needs_reference: bool = False,
                    flat_at_clean: bool = False) -> Callable:
    def decorate(score: Callable) -> Callable:
        name = score.__name__.lstrip("_")
        METRICS[name] = Metric(name=name, description=description, score=score,
                               needs_reference=needs_reference, flat_at_clean=flat_at_clean)
        return score
    return decorate


def metric_names() -> List[str]:
    return sorted(METRICS)


def metric(name: str) -> Metric:
    if name not in METRICS:
        raise AttributionError(f"no metric named '{name}'; metrics are {', '.join(metric_names())}")
    return METRICS[name]


def scored(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
           reference: Optional[torch.Tensor] = None):
    """The rows a metric is actually about: one per scored token, with its target and its reference

    Position i predicts token i+1, so the weights are shifted with the logits.
    Getting that shift wrong scores the prompt's last token and calls it the
    first token of the answer.

    The selection happens *before* any softmax rather than after. A few-shot
    translation prompt is ~300 tokens of which ~20 are scored, so the discarded
    ninety-odd percent is a distribution over Qwen's 151,936 entries computed
    in float32 and then multiplied by zero. On the 8B that did not move wall
    clock or peak memory measurably -- the backward graph over the band
    dominates both -- so this is arithmetic not done rather than a speedup
    claimed.
    """
    keep = weights[:, 1:] > 0
    picked = (logits[:, :-1][keep], ids[:, 1:][keep])
    return picked if reference is None else (*picked, reference[:, :-1][keep])


@register_metric("the model's log-likelihood of the continuation being scored, per token")
def _target(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
            reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    """mean over scored positions of log p(the token that is actually there)

    The claim-about-one-answer end of the metric axis, which is what a logit
    difference is on a single-token task: a number that goes down when the
    model stops producing *this* continuation and says nothing about what it
    produces instead.

    Which continuation is a separate choice and not this function's: teacher
    force the model's own baseline generations and the score measures
    departure from what the unablated model does; teacher force the WMT
    references and it measures departure from a translation the model may
    never have produced. `translation_study` picks; both go through here.
    """
    rows, targets = scored(logits, ids, weights)
    if not rows.numel():
        raise AttributionError("no positions are being scored; every span's continuation is empty")
    return rows.float().log_softmax(dim=-1).gather(1, targets[:, None]).mean()


@register_metric("minus the KL from the unablated model's next-token distribution to this one",
                 needs_reference=True, flat_at_clean=True)
def _kl(logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
        reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    """-KL(full || this), averaged over scored positions, over the whole vocabulary

    The whole-distribution end of the metric axis. It sees a component that
    reorders the tail of the distribution without changing the argmax, which
    `target` cannot, and it is the metric the DiscoGP paper evaluates with.

    Negated so that every metric in this module is a *goodness*: higher is a
    healthier model, and a positive attribution score therefore means
    ablating hurts, for both metrics, without a per-metric sign anywhere.

    Its gradient at the clean point is zero -- the clean model is where the KL
    is minimised -- which is why it is registered `flat_at_clean` and why
    `attribute` refuses it at `steps=1` instead of returning the zeros.
    """
    if reference is None:
        raise AttributionError("the kl metric needs the unablated model's logits and was given none")
    rows, _, full = scored(logits, ids, weights, reference)
    if not rows.numel():
        raise AttributionError("no positions are being scored; every span's continuation is empty")
    target = full.float().log_softmax(dim=-1)
    return -(target.exp() * (target - rows.float().log_softmax(dim=-1))).sum(dim=-1).mean()


@dataclass(frozen=True)
class Attribution:
    """What a scoring pass produced, with everything needed to say what it is an estimate of

    The scores are in the metric's own units -- nats of log-likelihood per
    token for `target`, nats of KL for `kl` -- per example, and they are
    estimates of a *drop*: positive means the model is worse without this
    unit. They are not BLEU and are not claimed to be; `agreement` is how a
    ranking here is compared with one measured in BLEU.
    """
    scores: Dict[Key, float]
    method: str
    metric: str
    granularity: str
    units: str
    layers: List[int]
    steps: int
    examples: int
    passes: int
    seconds: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def ranked(self, count: Optional[int] = None, negative: bool = False) -> List[Tuple[Key, float]]:
        """Every unit from most to least damaging to lose, or the other end with `negative`"""
        order = sorted(self.scores.items(), key=lambda pair: pair[1], reverse=not negative)
        return order[:count] if count is not None else order

    def top(self, count: int) -> List[Key]:
        return [key for key, _ in self.ranked(count)]

    def to_nodes(self) -> "Attribution":
        """An edge attribution summed back down to its sources, which is node attribution

        A node ablation removes a source's write from every destination at
        once, so its first-order score is the sum of its edges' -- the same
        sum autograd performs for free when the gradient is read at the
        source's own site instead. Doing it explicitly is what makes the
        claim checkable, and `tests/attribution.py` checks it: two routes to
        one number is two chances to be wrong.

        The reverse does not exist. Nothing in a node score says which reader
        the damage was done to, which is the structure weight-space methods
        give up when they collapse back to per-unit scores.
        """
        if self.granularity == "node":
            return self
        summed: Dict[Key, float] = {}
        for (source, _), value in self.scores.items():
            summed[source] = summed.get(source, 0.0) + value
        return Attribution(
            scores=summed, method=self.method, metric=self.metric, granularity="node",
            units=self.units, layers=self.layers, steps=self.steps, examples=self.examples,
            passes=self.passes, seconds=self.seconds, notes={**self.notes, "summed_from": "edge"},
        )

    def edges(self) -> List[Edge]:
        if self.granularity != "edge":
            raise AttributionError("a node attribution has no edges; run with granularity='edge'")
        return [key for key, _ in self.ranked()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method, "metric": self.metric, "granularity": self.granularity,
            "units": self.units, "layers": self.layers, "steps": self.steps,
            "examples": self.examples, "passes": self.passes, "seconds": round(self.seconds, 1),
            "notes": self.notes,
            "ranked": [{"unit": _unit_id(key), "score": round(value, 6)} for key, value in self.ranked()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attribution":
        granularity = data["granularity"]
        scores = {_unit_key(row["unit"], granularity): float(row["score"]) for row in data["ranked"]}
        return cls(scores=scores, method=data["method"], metric=data["metric"], granularity=granularity,
                   units=data["units"], layers=list(data["layers"]), steps=int(data["steps"]),
                   examples=int(data["examples"]), passes=int(data["passes"]),
                   seconds=float(data.get("seconds", 0.0)), notes=dict(data.get("notes", {})))


def _unit_id(key: Key) -> str:
    return f"{key[0]}->{key[1]}" if isinstance(key, tuple) else str(key)


def _unit_key(unit: str, granularity: str) -> Key:
    if granularity != "edge":
        return unit
    source, _, destination = unit.partition("->")
    return (source, destination)


@dataclass
class _Site:
    """One place a component writes, with the value it is ablated toward

    `pre` says which end of the module it is: a head's write is read at the
    *input* to the output projection, an MLP's at its output, which is
    exactly the pair `knockout.ablate` replaces. The two hook kinds differ in
    signature and in what returning a value means, and that is the only
    reason this distinction exists here.
    """
    key: Tuple[str, int]
    module: torch.nn.Module
    pre: bool
    mean: torch.Tensor


def _sites(adapter, means: Means, layers: Sequence[int]) -> List[_Site]:
    missing = [layer for layer in layers if not means.covers([layer])]
    if missing:
        raise AttributionError(
            f"no counterfactual means for layers {sorted(set(missing))}; they were captured over "
            f"{means.layers}, and a component with no mean has nothing to be ablated toward"
        )
    found = []
    for layer in layers:
        merged = means.heads[layer].reshape(-1)       # [n_heads * d_head], the projection's input
        found.append(_Site(("heads", layer), adapter.projections[layer], True, merged))
        found.append(_Site(("mlps", layer), adapter.mlps[layer], False, means.mlps[layer]))
    return found


def _mixed(live: torch.Tensor, mean: torch.Tensor, alpha: float) -> torch.Tensor:
    """The activation at strength `alpha` along the path from mean-ablated to live

    `alpha * live + (1 - alpha) * mean`, with `live` left in the graph rather
    than detached. That last part is the whole design of this function and is
    what makes the path taken here neither of the two variants the literature
    names:

      the `inputs` variant interpolates the input embeddings, which keeps the
      model's computation intact but has no analogue when the counterfactual
      is a mean activation rather than a corrupted prompt;

      the `activations` variant writes each node's interpolated value as a
      constant, which severs the graph at every site -- and with a site at
      every layer, a component's influence on the metric *through a later
      component* is then invisible, which is not what EAP estimates.

    So the mixture is of the mean with whatever the partially-ablated model
    actually produced at this site, and the path is the model's own
    trajectory as ablation strength goes from 1 to 0. At `alpha = 1` it is
    the identity on `live` and the estimate is plain EAP, which is why
    `steps=1` needs no separate code path. Away from 1 the path is not a
    straight line in joint activation space, so this is a path-integrated
    estimate rather than the paper's integral, and `units` says `path`
    wherever it is used.
    """
    if alpha == 1.0:
        return live
    return alpha * live + (1.0 - alpha) * mean.to(live.device, live.dtype)


def _capture(sites: Sequence[_Site], store: Optional[Dict[Tuple[str, int], torch.Tensor]],
             grads: Optional[Dict[Tuple[str, int], torch.Tensor]] = None,
             alpha: float = 1.0) -> List[torch.utils.hooks.RemovableHandle]:
    """Hooks that record each site's value, mix it toward the mean, and accumulate its gradient

    `store=None` records nothing, which is what the gradient passes want: the
    displacement every score is built from is `clean - mean`, measured once on
    the unablated pass, and the interpolated values the later passes run at are
    not it. Keeping them would be keeping a copy of every site's activation
    per step for nobody to read.
    """
    handles = []

    def watch(site: _Site, value: torch.Tensor) -> torch.Tensor:
        mixed = _mixed(value, site.mean, alpha)
        if store is not None:
            store[site.key] = mixed.detach()
        if grads is not None:
            def accumulate(gradient, key=site.key):
                grads[key] = gradient.detach() if key not in grads else grads[key] + gradient.detach()
            mixed.register_hook(accumulate)
        return mixed

    for site in sites:
        if site.pre:
            def pre_hook(module, args, site=site):
                return (watch(site, args[0]), *args[1:])
            handles.append(site.module.register_forward_pre_hook(pre_hook))
        else:
            def post_hook(module, args, output, site=site):
                hidden = output[0] if isinstance(output, tuple) else output
                replaced = watch(site, hidden)
                return (replaced, *output[1:]) if isinstance(output, tuple) else replaced
            handles.append(site.module.register_forward_hook(post_hook))
    return handles


def _destination_hooks(adapter, layers: Sequence[int],
                       slots: Dict[str, torch.Tensor]) -> List[torch.utils.hooks.RemovableHandle]:
    """Hooks that open a per-reader slot at each destination, so its gradient is that reader's alone

    The obvious implementation is a tensor hook on what the destination read,
    and it is wrong in a way that looks right. A block's pre-attention norm is
    handed the residual stream, and the residual stream is *also* the skip
    connection and the input to everything after it; the gradient of that
    tensor is the total over all of those uses. Summing such gradients over
    destinations counts every downstream path once per destination upstream of
    it, and the node score an edge set adds up to comes out several times too
    large -- a discrepancy that is easy to read as a linearization error and
    is not one.

    What an edge needs is the *partial* derivative through one reader, which
    is what `edge_patch` intervenes on: one destination's input changes and
    nobody else's does. So a zero tensor is added to the value on its way into
    the reader and nowhere else, and its gradient is by construction the part
    of d(metric) that arrived through this reader. Adding zero changes no
    number in the forward pass; it changes which paths the backward pass can
    tell apart.

    Only destinations at or after the first scored layer are opened, plus
    `logits`, which every source reaches: a destination upstream of every
    scored source reads none of them and would contribute a column of zeros.
    """
    handles = []
    first = min(layers)
    for destination, module in adapter.destinations().items():
        if destination != READOUT and int(destination.split(":")[1]) < first:
            continue

        def pre_hook(module, args, destination=destination):
            value = args[0]
            slot = torch.zeros_like(value, requires_grad=True)
            slots[destination] = slot
            return (value + slot, *args[1:])
        handles.append(module.register_forward_pre_hook(pre_hook))
    return handles


def _node_scores(adapter, sites: Sequence[_Site], clean: Dict[Tuple[str, int], torch.Tensor],
                 grads: Dict[Tuple[str, int], torch.Tensor], mask: torch.Tensor,
                 into: Dict[Key, float]) -> None:
    """Accumulate (clean - mean) . gradient per component, over the batch's real tokens

    Padding is dropped before the sum. A padded position's activation is
    whatever the model made of a pad token, and its gradient is real: summing
    over it adds a number that depends on how long the *other* sentences in
    the batch were.
    """
    width = adapter.cfg.d_head
    weights = mask[..., None].float()
    for site in sites:
        kind, layer = site.key
        if site.key not in grads:
            continue
        delta = (clean[site.key].float() - site.mean.to(clean[site.key].device).float()) * weights
        contribution = delta * grads[site.key].float()
        if kind == "mlps":
            into[comp.name("mlp", layer)] = into.get(comp.name("mlp", layer), 0.0) + float(contribution.sum())
            continue
        per_head = contribution.reshape(*contribution.shape[:-1], adapter.cfg.n_heads, width).sum(dim=(0, 1, 3))
        for head in range(adapter.cfg.n_heads):
            key = comp.name("head", layer, head)
            into[key] = into.get(key, 0.0) + float(per_head[head])


def _edge_scores(adapter, sites: Sequence[_Site], clean: Dict[Tuple[str, int], torch.Tensor],
                 grads: Dict[str, torch.Tensor], mask: torch.Tensor, layers: Sequence[int],
                 into: Dict[Key, float]) -> None:
    """Accumulate (clean write - mean write) . d(metric)/d(destination input) per legal edge

    A source's write into the residual stream, not its activation at the site:
    a head's mean lives at the input to the output projection and is not a
    residual-stream vector until it has been through it, which is what
    `adapter.head_write` is for. An MLP's site *is* its write and needs no
    conversion.

    Every destination's gradient is stacked once and each source is contracted
    against all of them at once, because the alternative -- a dot product per
    (source, destination) pair -- is thousands of small kernel launches for
    one einsum's worth of arithmetic.

    Which pairs are legal is asked of `adapter.edges()` rather than decided
    here. A residual stream is causal and layer L's attention cannot read
    layer L's MLP; that rule is architecture knowledge and the backend is
    where it lives, so a second copy of it in this file is a second copy that
    can disagree. The one edge added on top is `-> logits`, which every source
    has and `edges()` deliberately does not list (see `destinations`): without
    it a source's edges do not sum to its node score, and the gap is the
    direct path to the output.
    """
    weights = mask[..., None].float()
    names = list(grads)
    if not names:
        return
    stacked = torch.stack([grads[name].float() for name in names])       # [destination, batch, seq, d_model]
    width = adapter.cfg.d_head
    index_of = {name: index for index, name in enumerate(names)}
    reachable: Dict[str, List[int]] = {}
    if READOUT in index_of:
        for layer in layers:
            for source in [comp.name("mlp", layer)] + [comp.name("head", layer, head)
                                                       for head in range(adapter.cfg.n_heads)]:
                reachable.setdefault(source, []).append(index_of[READOUT])
    for source, destination in adapter.edges():
        if destination in index_of:
            reachable.setdefault(source, []).append(index_of[destination])

    def spread(source: str, delta: torch.Tensor) -> None:
        legal = reachable.get(source, [])
        if not legal:
            return
        values = torch.einsum("dbsm,bsm->d", stacked[legal], delta * weights)
        for offset, index in enumerate(legal):
            key = (source, names[index])
            into[key] = into.get(key, 0.0) + float(values[offset])

    # No gradient is wanted from this arithmetic, and `head_write` runs the
    # live output projection: without this the writes come back attached to a
    # graph nobody backpropagates, held alive until the next batch frees them.
    with torch.no_grad():
        for site in sites:
            kind, layer = site.key
            if site.key not in clean:
                continue
            delta = clean[site.key].float() - site.mean.to(clean[site.key].device).float()
            if kind == "mlps":
                spread(comp.name("mlp", layer), delta)
                continue
            for head in range(adapter.cfg.n_heads):
                isolated = torch.zeros_like(delta)
                isolated[..., head * width : (head + 1) * width] = delta[..., head * width : (head + 1) * width]
                spread(comp.name("head", layer, head),
                       adapter.head_write(layer, isolated.to(stacked.dtype)).float())


def attribute(adapter, means: Means, spans: Sequence[Span], layers: Sequence[int],
              metric_name: str = "target", steps: int = 1, granularity: str = "node",
              batch_size: int = DEFAULT_BATCH, max_length: int = MAX_LENGTH,
              label: str = "attribution") -> Attribution:
    """Score every unit of `layers` by the first-order effect of mean-ablating it

    One clean pass per batch to record what each site holds and, for `kl`,
    what the unablated model predicts; then `steps` passes along the path
    from the mean-ablated model back to it, each one backward-differentiating
    the metric and accumulating the gradient at every site at once. The cost
    is `1 + steps` passes per batch *whatever the lattice is* -- which is the
    whole reason this exists next to a sweep that pays per component.

    Gradients are summed over the batch and over real tokens and divided by
    the number of examples at the end, so a score means the same thing at 20
    sentences and at 200 and the two can be compared.
    """
    if granularity not in ("node", "edge"):
        raise AttributionError(f"granularity is 'node' or 'edge', got '{granularity}'")
    if steps < 1:
        raise AttributionError(f"attribution needs at least one step, got {steps}")
    chosen = sorted({int(layer) for layer in layers})
    if not chosen:
        raise AttributionError("no layers to attribute over")
    wanted = metric(metric_name)
    if wanted.flat_at_clean and steps == 1:
        raise AttributionError(
            f"the '{metric_name}' metric is at its optimum on the unablated model, so its gradient "
            "there is zero and every first-order score would be zero; run it with steps > 1, which "
            "reads the gradient away from the clean point, or use a metric that is not flat there "
            f"({', '.join(name for name in metric_names() if not METRICS[name].flat_at_clean)})"
        )
    if granularity == "edge" and not hasattr(adapter, "destinations"):
        raise AttributionError(
            f"backend '{adapter.cfg.backend}' does not expose destinations(); edge attribution needs "
            "to know what each reader saw, and only a backend with the residual-stream surface does"
        )

    sites = _sites(adapter, means, chosen)
    totals: Dict[Key, float] = {}
    batches = [list(spans[start : start + batch_size]) for start in range(0, len(spans), batch_size)]
    bar = Progress(len(batches), label, every=max(1, len(batches) // 10))
    with measure(items=len(spans)) as cost, differentiable(adapter):
        for batch in batches:
            ids, mask = encode(adapter, [span.text for span in batch], max_length)
            weights = scored_positions(batch, mask)
            clean: Dict[Tuple[str, int], torch.Tensor] = {}
            with hooked(_capture(sites, clean)), torch.no_grad():
                # only `kl` needs the unablated distribution kept; for the others
                # these are [batch, seq, vocab] held for the whole batch to be ignored
                output = adapter.model(ids, attention_mask=mask, use_cache=False)
                reference = output.logits.detach() if wanted.needs_reference else None
                del output

            grads: Dict[Tuple[str, int], torch.Tensor] = {}
            destinations: Dict[str, torch.Tensor] = {}
            for step_index in range(1, steps + 1):
                alpha = step_index / steps
                slots: Dict[str, torch.Tensor] = {}
                handles = _capture(sites, None, grads, alpha)
                if granularity == "edge":
                    handles += _destination_hooks(adapter, chosen, slots)
                with hooked(handles), torch.enable_grad():
                    logits = adapter.model(ids, attention_mask=mask, use_cache=False).logits
                    wanted.score(logits, ids, weights, reference).backward()
                for name, slot in slots.items():
                    if slot.grad is not None:
                        gradient = slot.grad.detach()
                        destinations[name] = gradient if name not in destinations else destinations[name] + gradient
                adapter.model.zero_grad(set_to_none=True)
                del logits, slots

            averaged = {key: value / steps for key, value in grads.items()}
            if granularity == "node":
                _node_scores(adapter, sites, clean, averaged, mask, totals)
            else:
                _edge_scores(adapter, sites, clean,
                             {name: value / steps for name, value in destinations.items()},
                             mask, chosen, totals)
            bar.tick(f"{len(totals)} units")
    bar.finish()

    integrated = steps > 1
    return Attribution(
        scores={key: value / max(1, len(spans)) for key, value in totals.items()},
        method="eap_ig_path" if integrated else "eap",
        metric=metric_name,
        granularity=granularity,
        units=f"{wanted.name} drop per example" + (" (integrated, ablation path)" if integrated else " (first order)"),
        layers=chosen,
        steps=steps,
        examples=len(spans),
        passes=len(batches) * (1 + steps),
        seconds=cost[0].seconds,
        notes={"metric_description": wanted.description},
    )


def with_layer_groups(estimate: Attribution, n_heads: int) -> Attribution:
    """The same scores plus a `heads:L` entry per layer, so a group component can be compared

    The knockout sweep scores `heads:L` -- a whole layer's attention ablated
    at once -- alongside single heads, because a solo head rarely moves corpus
    BLEU by more than noise and the group does. Attribution has no such
    problem and scores every head separately, so to be compared on the study's
    own vocabulary it has to be able to answer for the group too.

    The group's score is the sum of its heads', which is not a convenience:
    first-order attribution is linear in the intervention, so the estimate for
    ablating a set is exactly the sum of the estimates for ablating its
    members. That is also the assumption the sum is worth testing against --
    where the group's *measured* effect is far from the sum of its members',
    the linearity is what broke.
    """
    grouped = dict(estimate.scores)
    for layer in estimate.layers:
        heads = [estimate.scores.get(comp.name("head", layer, head), 0.0) for head in range(n_heads)]
        grouped[comp.name("heads", layer)] = sum(heads)
    return Attribution(
        scores=grouped, method=estimate.method, metric=estimate.metric, granularity=estimate.granularity,
        units=estimate.units, layers=estimate.layers, steps=estimate.steps, examples=estimate.examples,
        passes=estimate.passes, seconds=estimate.seconds,
        notes={**estimate.notes, "layer_groups": "heads:L is the sum of its heads"},
    )


def agreement(estimate: Attribution, measured: Dict[str, float], at: Sequence[int] = (5, 10, 20)) -> Dict[str, Any]:
    """How well an attribution ranking reproduces a ranking that was actually measured

    `measured` is component id -> the real number, phase 1b's `dbleu` being
    the one this was written for: the drop in corpus BLEU when that component
    is mean-ablated, at one generation pass per component. Both rankings are
    restricted to the components they share, because a Spearman over a union
    padded with zeros is a correlation with the padding.

    Two numbers, deliberately. The correlation says whether the estimate
    tracks the measurement over the whole lattice; the overlap at k says
    whether it puts the same components at the top, which is the only part a
    greedy walk ever reads. An estimate can be good at one and poor at the
    other, and which one matters depends on what the ranking is for.

    Both numbers are only as good as `measured`, and that is the trap this
    comparison sets. The sweep's own noise floor is visible in its file: 288
    of its 306 components are single heads scored on half the corpus, with a
    dBLEU spread of 0.21 around zero and ten of them outside +-0.5. A
    correlation taken over all of them is mostly a correlation against that
    spread, and it comes back near zero whatever the estimate says. Pass the
    part of the sweep that was measured on the full set -- `sweep_measured`
    takes the filter -- and the question becomes answerable.
    """
    scores = estimate.to_nodes().scores
    shared = sorted(set(scores) & set(measured))
    if not shared:
        raise AttributionError(
            f"the estimate and the measurement share no components (estimate has {len(scores)}, "
            f"measurement {len(measured)}); they were probably scored over different layers"
        )
    estimated_order = [key for key, _ in sorted(scores.items(), key=lambda pair: -pair[1]) if key in shared]
    measured_order = [key for key, _ in sorted(measured.items(), key=lambda pair: -pair[1]) if key in shared]
    overlap = {
        f"top{count}": round(jaccard(estimated_order[:count], measured_order[:count]), 4)
        for count in at if count <= len(shared)
    }
    return {
        "components": len(shared),
        "spearman": round(spearman([scores[key] for key in shared], [measured[key] for key in shared]), 4),
        "overlap": overlap,
        "estimate_top": estimated_order[: max(at)],
        "measured_top": measured_order[: max(at)],
        "method": estimate.method, "metric": estimate.metric, "steps": estimate.steps,
        "passes": estimate.passes, "seconds": round(estimate.seconds, 1),
    }


def disagreement(first: Attribution, second: Attribution, at: int = 20) -> Dict[str, Any]:
    """How much two attributions of the same lattice disagree, which is the point of varying a choice

    Written for the metric axis -- the same components, the same base, scored
    under `target` and under `kl` -- where the claim being tested is Zhang &
    Nanda's: that the circuit you report depends on a choice most papers make
    in one line and do not revisit. It is the same measurement for the
    integration axis, and it is a measurement rather than an argument.
    """
    left, right = first.to_nodes(), second.to_nodes()
    shared = sorted(set(left.scores) & set(right.scores))
    if not shared:
        raise AttributionError("the two attributions share no units; they are not of the same lattice")
    return {
        "units": len(shared),
        "spearman": round(spearman([left.scores[key] for key in shared],
                                   [right.scores[key] for key in shared]), 4),
        f"overlap_top{at}": round(jaccard(left.top(at), right.top(at)), 4),
        "left": {"method": left.method, "metric": left.metric, "steps": left.steps, "top": left.top(at)},
        "right": {"method": right.method, "metric": right.metric, "steps": right.steps, "top": right.top(at)},
    }


def split(estimate: Attribution, at: int = 20) -> Dict[str, Any]:
    """What an edge attribution says that its own node summary cannot

    `wiring.split` asks this of a trained edge circuit: a source kept for one
    reader and cut for another is a fact about an edge and disappears in any
    per-component summary. Here the same question is asked of scores rather
    than of a mask -- for each source in the top `at` edges, how much of its
    total attribution one destination accounts for. A source whose share is
    near 1 is a node wearing an edge's clothes; a source spread over several
    readers is the structure node attribution loses.
    """
    if estimate.granularity != "edge":
        raise AttributionError("split is about edges; a node attribution has no destinations to split over")
    per_source: Dict[str, Dict[str, float]] = {}
    for (source, destination), value in estimate.scores.items():
        per_source.setdefault(source, {})[destination] = value
    rows = []
    for source in {source for source, _ in estimate.top(at)}:
        edges = per_source.get(source, {})
        total = sum(abs(value) for value in edges.values())
        best, share = max(edges.items(), key=lambda pair: abs(pair[1]))
        rows.append({"source": source, "destinations": len(edges),
                     "largest": best, "largest_share": round(abs(share) / total, 4) if total else 0.0,
                     "total": round(sum(edges.values()), 6)})
    rows.sort(key=lambda row: -abs(row["total"]))
    return {"sources": len(rows), "rows": rows}


def as_components(estimate: Attribution, count: Optional[int] = None,
                  positive_only: bool = True) -> List[str]:
    """The ranking as the component ids the rest of the study argues in

    `positive_only` drops the units the estimate says the model is *better*
    without. They are a real finding and they are not a circuit: a greedy
    walk handed one adds a component to improve its own score, which is how a
    candidate set grows past the frontier without anything being wrong with
    any single step.
    """
    ranked = estimate.to_nodes().ranked()
    chosen = [key for key, value in ranked if value > 0 or not positive_only]
    return chosen[:count] if count is not None else chosen
