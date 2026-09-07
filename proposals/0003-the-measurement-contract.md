# **Proposal: the measurement contract — what `methods/` is allowed to know**

| Field          | Value                                                                          |
| :------------- | :------------------------------------------------------------------------------ |
| **Status**     | **Implemented** — all eight steps of §VII                                        |
| **Date**       | September 2026                                                                   |
| **Reference**  | `model/adapter.py`, `data/tasks.py`, `methods/common/span.py`, `share/definitions.py` |
| **Enables**    | [0002 — the second axis](0002-multi-domain-architecture.md), phases 3 onward       |
| **Supersedes** | nothing. `CircuitTask` grows; `IOIDataset` and the four `TemplateTask`s keep working |

---

## 0. What landed, and where it differs from this document

All eight steps of §VII are in the tree, and the three numerical receipts in
§VI did not move.

| Decision | What this document said | What was built, and why |
| :--- | :--- | :--- |
| Where `Score` lives | `methods/common/` | `core/readout.py`. `data/tasks.py` has to name the type `CircuitTask.readout` returns and `data` sits below `methods`, so the proposed home was a layering violation. Core's charter is "what a model is and how a number is scored", which is exactly this |
| The `Readout` marker | a Protocol with no members | A `differentiable` **field** on `Score`, with `Readout` kept as the name for it in a signature. §VIII.1 already noticed the problem and proposed a smoke gradient; the real issue is worse than it says — a Protocol with no members of its own is satisfied *structurally by everything*, so `isinstance(bleu, Readout)` is True and the refusal never fires. `require_readout` decides on the field |
| `Score.select` | not in the proposal | Added, and it is load-bearing. Every backend runs a batch in chunks and the readout holds one answer per row of the *whole* batch; the old code sliced its id lists per chunk explicitly. Without it a chunked gradient scores each chunk against the first chunk's answers |
| `DirectScore` | §IV said only "check the receipt still closes" | Added. Direct attribution does not score an output, it scores each write *into* one, and what the write is dotted with is the readout's business. `require_direct` names a readout with no such form |
| `quality.py` | "BLEU/chrF/COMET become `Score`s" | `TextQuality` is a `Score` over *sentence*-level BLEU/chrF with `corpus()` beside it. Corpus BLEU pools n-gram counts before dividing and is not a mean of sentence scores, so the two are kept apart and the knockout tables' numbers did not move |
| `training.py`'s faith kinds | "three `Readout`s" | Four `Faithfulness` objects in a new `methods/sheaves/faith.py` (`pair` exists too), each carrying the sentence that says when it lies. `whole` — does this kind need the whole vocabulary — is read off the object rather than recomputed as `faith_kind in ("kl", "nll", "gold")` at three separate call sites |
| `Behaviour.logit_difference` → `.score` | §VIII.3 asks whether it is serialized | It is not. `experiment/run.py` records `clean_logit_difference` from the IOI report, not from `Behaviour`, and no converter touches the field. A rename, not a migration |
| `head_gate_gradients` | not mentioned | Renamed to `gate_gradients` and given the same readout treatment. It carried the same `positive`/`negative` pair and would have been the last one left |

One numerical change worth naming: the gradient objective now goes through
`core.metrics.logit_difference`, which casts to float64, where the backend used
to index the logits itself in the model's dtype. That is one implementation of
one equation instead of two, and the finite-difference receipt passes unchanged.

---

## I. The claim

> Everything in `methods/` consumes exactly three things: **a batch of inputs**, **a set
> of addressable internal sites**, and **one scalar per example**. Two of the three are
> already abstract. The third is hardcoded as a logit difference over token ids, and
> that single fact is what makes twenty-three otherwise modality-free modules into
> language modules.

This proposal names the third thing, gives it a type, and hands it to the task that
knows what it should be.

It is worth doing on its own merits before any second modality exists, for the reason
in §V: `share/schema/metric.py` requires that *a metric carries the definition that
produced it*, and today that definition comes from a lookup table keyed by name, which
can drift from the code that computed the number. A readout object closes that gap by
construction.

---

## II. The three leaks, with their cost

### II.1 `CircuitAdapter.logits` — a vocabulary in the contract

`model/adapter.py:195`

```python
def logits(self, prompts: Sequence[str]) -> torch.Tensor:
    """Next-token logits at each prompt's final real token, as [batch, vocab]"""
```

Two commitments in one signature: inputs are strings, and the output is indexed by a
vocabulary. `single_token` and `tokens` follow from the same commitment. A model
scored by cosine similarity to an embedding, or by return over a rollout, cannot
satisfy this and has no reason to want to.

### II.2 `methods/common/span.py` — the readout, hardcoded, twice

`methods/common/span.py:41` and `:63`

```python
def mean_logit_difference(adapter, prompts, io, subject) -> float: ...

def baselines(adapter, dataset: CircuitTask) -> Baselines:
    io, subject = dataset.answers(adapter)
    clean     = mean_logit_difference(adapter, dataset.clean, io, subject)
    corrupted = mean_logit_difference(adapter, dataset.corrupted, io, subject)
```

`Baselines` then *carries* `io: List[int]` and `subject: List[int]` — token ids — into
every downstream consumer, and `Behaviour` has a field literally named
`logit_difference`. `Baselines.recovery` and `Baselines.span` are pure arithmetic and
would survive any readout; the two lists are the whole leak, and they are structural
rather than cosmetic because every technique that needs a gradient receives those two
lists and passes them to `head_gradients(prompts, positive, negative, ...)`.

### II.3 `CircuitTask` — answers as token ids

`data/tasks.py:42`

```python
clean: List[str]
corrupted: List[str]
def answers(self, adapter) -> Tuple[List[int], List[int]]: ...
def token_labels(self, adapter) -> List[str]: ...
```

The docstring already contains the correct instinct — the ids are read off the model
in hand rather than cached, "because the ids are the tokenizer's opinion and a task
carrying them would be silently wrong on the next model". The generalization is one
step further: *the whole scoring rule* is the task's opinion about this model, not
just the ids in it.

---

## III. The contract

Four types. Three are new; one is `CircuitTask` with two members changed.

### III.1 `Score` and `Readout` — and why they are two things

```python
@runtime_checkable
class Score(Protocol):
    """One number per example, in stated units, with the definition that produced it

    `definition` is not documentation. share/schema/metric.py refuses a Metric with
    an empty definition, and share/definitions.py is a lookup table keyed by name
    that can disagree with the code. A Score carries its own, so an artifact's
    definition is written by the thing that computed the number.
    """
    name: str
    units: str
    definition: str

    def __call__(self, output: Any) -> torch.Tensor:
        """[batch] floats, one per example, higher meaning more of the behaviour"""


@runtime_checkable
class Readout(Score, Protocol):
    """A Score that a gradient can be taken through

    Marker only: it adds no members. What it promises is that `__call__` is a
    differentiable function of `output` with the graph intact -- which is the
    thing eap, eap_ig and mask training need and the thing BLEU cannot give.
    """
```

**Why the split is not ceremony.** `methods/knockout/quality.py` scores generations
with BLEU, chrF and COMET. Those are perfectly good scores for ablation — you take a
component away and the sentences get worse — and there is no gradient through them.
Meanwhile `eap` is "patching's first-order expansion at a constant five passes", which
is a claim about a derivative.

Today the two never meet because the circuit half and the knockout half each hardcode
their own scoring. Under one abstract `Score` they would meet, and the failure would
be a `RuntimeError` about a tensor with no `grad_fn`, several minutes into a run.

So: `techniques.py`'s registry entry grows one field.

```python
@register_technique("eap", passes=5, needs_gradient=True)
```

and the registry refuses at selection time, in the `require_circuits` style —
naming the readout and the technique, before the model loads:

```
eap needs a gradient through the score, and the readout for task 'translation-bleu'
is BLEU, which has none. Techniques that work with this readout: patching, ablation,
random. Techniques that need a differentiable readout: attribution, eap, eap_ig, mask.
```

### III.2 `CircuitTask`, generalized

```python
class CircuitTask(Protocol):
    name: str
    modality: str                     # "text" | "image" | "trajectory" -- for the .mia card

    @property
    def clean(self) -> Sequence[Any]:      # was List[str]
    @property
    def corrupted(self) -> Sequence[Any]:  # was List[str]

    def readout(self, adapter) -> Score:
        """How a number comes out of this model on this task, bound to these examples

        Built against the adapter in hand for the reason `answers` was: the answer
        ids, the class embedding, the success predicate are all this model's opinion
        and a task carrying them is silently wrong on the next one.
        """

    def labels(self, adapter) -> List[str]:   # was token_labels
        """The first example's positions as strings, for labelling a position axis"""

    def landmarks(self, adapter) -> Dict[str, int]: ...
    def subset(self, indices: Sequence[int]) -> "CircuitTask": ...
    def __len__(self) -> int: ...
```

`answers()` does not disappear — it moves *inside* the LM readout's constructor, which
is the only place it was ever meaningful.

### III.3 `CircuitAdapter`, generalized

Two members change; the rest are already sites rather than words.

```python
def outputs(self, inputs: Sequence[Any]) -> Any:
    """Whatever this model produces for these inputs, in whatever form the readout
    for this domain expects. For a decoder: next-token logits at each input's final
    real token. The measurement layer never indexes this -- only a Score does."""

def gradients(
    self,
    inputs: Sequence[Any],
    readout: Readout,
    layers: Optional[Sequence[int]] = None,
    toward: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """d readout(outputs(inputs)) / d (head output), as [batch, layer, head, seq, d_head]

    Same site head_outputs reads and patch writes, differentiated rather than
    recorded, and the graph is kept through the block rather than detached --
    cutting it deletes every path an earlier head has to the answer through this
    layer's attention, leaving a gradient that looks fine and answers a different
    question. That note survives this change unaltered; only what is being
    differentiated moves from a fixed logit difference to the readout in hand.
    """
```

`logits`, `single_token` and `tokens` do not vanish. They move to an LM-specific
protocol that the decoder backend also satisfies:

```python
class TokenAdapter(CircuitAdapter, Protocol):
    def logits(self, prompts) -> torch.Tensor: ...
    def single_token(self, text: str) -> int: ...
    def tokens(self, prompt: str) -> List[str]: ...
```

`require_tokens(adapter)` beside `require_circuits(adapter)`, same message shape. The
consumers are `domains/lm/` and the parts of the CLI that print tokens — which is
exactly the set that should have to ask.

### III.4 `Baselines` and `Behaviour`, unpinned

```python
def mean_score(adapter, inputs: Sequence[Any], score: Score) -> float:
    return float(score(adapter.outputs(list(inputs))).mean())


@dataclass(frozen=True)
class Baselines:
    clean: float
    corrupted: float
    readout: Score                    # replaces io / subject

    @property
    def span(self) -> float:          # unchanged
        return self.clean - self.corrupted

    def recovery(self, patched: float) -> float:   # unchanged
        return recovery(patched, self.clean, self.corrupted)


@dataclass(frozen=True)
class Behaviour:
    score: float                      # was `logit_difference`
    units: str                        # new: what `score` is in
    accuracy: float
    n: int
```

The span-near-zero refusal is unchanged and its message improves: it can now name the
readout that failed to move rather than saying "logit difference" on a task whose
number is something else.

---

## IV. What changes, module by module

| Module | Change | Risk |
| :--- | :--- | :--- |
| `model/adapter.py` | `logits` → `outputs`; `head_gradients` → `gradients(readout=)`; new `TokenAdapter` | Mechanical. The protocol is `runtime_checkable`, so the refusal path is already there |
| `model/backends/transformers/outputs.py`, `heads.py` | Implement both; `outputs` delegates to today's `logits` | None — the decoder's `outputs` *is* logits |
| `data/tasks.py` | Protocol change; `TemplateTask.readout` builds a `LogitDifference` from today's `answers` | Low. `TemplateTask` already computes the ids |
| `data/ioi.py` | `IOIDataset` gains `readout`, keeps `answers` | Low |
| `methods/common/span.py` | §III.4 | **The one to be careful with.** Every downstream `.io`/`.subject` read must move |
| `methods/circuits/patching.py`, `ablation.py`, `search.py`, `verify.py` | `baselines.io, baselines.subject` → `baselines.readout` | Mechanical, and the type checker finds all of them |
| `methods/circuits/attribution.py` | `Attribution.residual` receipt is against the readout now | **Check the receipt still closes at ~1e-6.** It should: the decomposition is unchanged, only what the final vector is dotted with moves |
| `methods/circuits/techniques.py` | `needs_gradient` on the registry entry; the refusal in §III.1 | New behaviour, needs a test |
| `methods/circuits/faithfulness.py`, `comparison.py` | Units come from the readout instead of being assumed | Low, and it fixes a real hazard: `compare_techniques` refuses to subtract scores not in the same units, and would now know |
| `methods/sheaves/training.py` | `--faith kl` / `nll` / `gold` are **three readouts chosen by a flag today**. They become three `Readout`s | Medium, and it is the payoff: the flag stops being a special case in the loss and becomes an object with a definition |
| `methods/knockout/quality.py` | BLEU/chrF/COMET become `Score`s | Low. They gain the ability to be a knockout target through the ordinary path |
| `share/converters/*.py` | `Metric.definition` read off the readout instead of `definitions.py` | Low, and see §V |
| `experiment/runner.py`, `cli/`, `ie/`, `viz/` | Column headers and labels come from `score.units` | Cosmetic, several sites |

**Not changed:** `spec_hash` (invariant 4 — a readout that determines a result is
named in the spec already, by task name), the `.mia` wire format's numbers, the
journal, `Cost`, every threshold, every test's expected value.

---

## V. The reason to do this even with one modality

`share/schema/metric.py` refuses a `Metric` with an empty `definition`, because
"`faithfulness` here is a logit-difference recovery under restoration; in the
faithfulness literature it is a normalized KL reproduction. Both report near 0.9."
That refusal is one of the six rules the format enforces, and it is currently
satisfied by `share/definitions.py` — one table, keyed by metric name, read by both
the converters and the migration.

A table keyed by name is a second opinion about what the number is. Nothing checks
that the entry for `"faithfulness"` describes the code that produced this artifact's
`faithfulness`, and the day `--faith` changes from `nll` to `kl` the table does not
move. That is the same class of bug as a `.pt` file identified by filename, which
`ActivationDataset` exists to prevent.

Under this proposal the definition travels with the callable that computed the number,
`definitions.py` becomes the fallback for *migrating old artifacts* (which is what it
is genuinely good at), and the converter reads `score.definition`.

Second smaller payoff: `methods/sheaves/training.py`'s `--faith {nll,kl,gold}` is
already three readouts, discovered the expensive way — a run against the full model's
argmax collapsed to emitting quotes at 95–98% density while faith read ~0. Three
readout objects, each carrying the sentence that explains when it lies, is where that
lesson belongs.

---

## VI. Tests

| Test | What it pins |
| :--- | :--- |
| `tests/methods/span.py::test_baselines_are_readout_agnostic` | A trivial `Score` (mean of the output) gives a well-formed `Baselines` with no tokenizer touched |
| `tests/methods/techniques.py::test_a_gradient_technique_refuses_a_non_differentiable_readout` | The §III.1 message, before the model loads |
| `tests/methods/discovery.py` (existing finite-difference check) | **Unchanged expected values.** This is the receipt that `gradients(readout=)` differentiates the same quantity `head_gradients` did |
| `tests/methods/circuits.py` (existing `remainder`/`residual` ~1e-6) | **Unchanged expected values** |
| `tests/model/adapter.py` golden capture | **Unchanged.** If it moves, the refactor touched the model and not just the plumbing |
| `tests/share/artifact.py::test_definition_comes_from_the_readout` | A converter writes the readout's definition, not the table's |
| `tests/data/tasks.py::test_every_registered_task_builds_a_readout` | The protocol is satisfied by all five |

The three "unchanged expected values" rows are the actual safety argument for this
refactor: it is a signature change with three numerical receipts already in place that
must not move, and one of them (the finite difference on `head_gradients`) is
specifically designed to catch a gradient taken at the wrong site.

---

## VII. Order of work

1. `Score` / `Readout` in `methods/common/` — new file, nothing imports it yet.
2. `LogitDifference` as the first `Readout`, in what will become `domains/lm/readout.py`.
   Assert it equals `core.metrics.logit_difference` on the IOI dataset.
3. `CircuitTask.readout` on `TemplateTask` and `IOIDataset`, alongside `answers`.
4. `span.py` to §III.4. Run the full suite: the three receipts must not move.
5. `adapter.py` — `outputs`, `gradients`, `TokenAdapter`. Run the golden capture.
6. `techniques.py` — `needs_gradient` and the refusal.
7. `quality.py` and `training.py` readouts.
8. `share/converters/` — definition from the readout.

Steps 1–3 are additive and cannot break anything. Step 4 is the one commit that is
worth reviewing carefully, and 0002's phase 2 grep test is what tells you when the
job is finished: the allow-list empties.

---

## VIII. Sharp edges

1. **`Readout` is a marker with no members, so nothing enforces differentiability at
   registration.** The check is at first use and it is a `RuntimeError` about
   `grad_fn`. Mitigate at construction: a one-example smoke gradient when a technique
   with `needs_gradient` is selected, before the batch is built. Cheap, and it moves
   the failure from minutes in to seconds in.
2. **The graph must still not be detached.** `head_gradients`'s existing docstring is
   the most important sentence in `adapter.py` and it survives verbatim. A refactor
   that routes the readout through a `.detach()` for tidiness produces a gradient that
   looks fine and answers a different question, and only `tests/methods/discovery.py`
   would say so.
3. **`Behaviour.logit_difference` → `Behaviour.score` is a field rename in something
   that gets serialized.** Check `experiment/run.py` and every `share/converters/`
   entry before renaming, and if it appears in a written `run.json`, this is a
   migration and not a rename.
4. **A `Score` bound to an adapter must not outlive it.** `task.readout(adapter)`
   closes over answer ids or an embedding read off *that* checkpoint. Held past a model
   swap it is `Means.check`'s failure mode with no `check` — so give it the same:
   stamp the config id in and refuse another. `telemetry/results.py::guard` and
   `methods/knockout/ablate.py::Means` are the two existing precedents and they agree.
5. **Units are now data, so a chart can be wrong quietly.** `viz/` reads `score.units`
   for an axis label. A readout with `units = ""` produces an unlabelled axis rather
   than an error. Refuse an empty `units` the way `Metric` refuses an empty
   `definition` — same rule, same reason.
6. **Do not generalize `landmarks` in this proposal.** "The positions worth reading a
   patching grid at, by name" is meaningful for a sentence, arguable for a patch grid,
   and probably wrong for a trajectory. It stays as it is and gets revisited when
   `domains/vision/` has a real task, rather than being designed against an imagined
   one.

---

## IX. Open question

`methods/probing/` is not covered here. A probe consumes activations and labels, not a
readout, so it is untouched by this proposal — but `LabeledPrompts` is as text-shaped
as `CircuitTask` was, and its `groups` invariant (a contrast pair must not straddle a
split) needs a non-text analogue before `domains/vision/` can probe anything. That is
either a small third proposal or a section added here once there is a vision dataset
to look at. Writing it now would be designing against an imagined dataset, which is
the mistake §VIII.6 refuses for `landmarks`.
