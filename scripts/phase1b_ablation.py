"""Phase 1b deliverable 4: the mean-ablation sweep over the candidate components.

Mean-ablation knockout (Wang et al. 2211.00593 3, which is also the
validation step Zhang et al. 2502.11806 4.3 use -- their *discovery* method is
subspace-intervened path patching, which this sweep does not implement).
Priced for this GPU: each candidate component (an attention head or an MLP
block in layers 27-35) is replaced by its mean activation over the
counterfactual reference distribution, the ES->EN translation is regenerated
on the 200-sentence WMT shortlist, and the drop in corpus BLEU against the
un-ablated baseline is the component's score.

The reference distribution is the counterfactual prompt form: the same
few-shot skeleton, the same shots and the same Spanish sources as the eval
prompts, with only the translation logic removed (see
src/data/translation.py). It replaced an earlier version that averaged over
raw English reference sentences -- that distribution differs from the eval
prompts in the prompt format and the in-context task as well as in
translation, so ablating toward it removed the model's ability to continue the
prompt at all, and a whole-layer knockout degenerated into repeated tokens
rather than into bad translation. Results measured against the old mean are
archived beside this file and are not comparable.

Engineering constraints, all deliberate:
- resumable: every finished component is flushed to
  results/phase1b-ablation-progress.json, and a --budget wall-clock limit
  makes each invocation exit cleanly, so the sweep runs as a chain of
  foreground calls with nothing orphanable in the background;
- layer-level components (a whole layer's heads, one MLP) are scored on all
  200 sentences; single heads on 100 -- a solo head rarely moves corpus BLEU
  by more than noise, so the halved pass buys double the coverage and the
  greedy stage re-verifies everything it uses at 200;
- the counterfactual means are captured once and cached beside the progress file.

Stages: sweep (default) walks a component group; assemble writes the ranked
results/phase1b-ablation-sweep.json; comet adds dCOMET for the top solo
components.

Run: uv run python -m scripts.phase1b_ablation qwen3-8b sweep layers
     uv run python -m scripts.phase1b_ablation qwen3-8b sweep heads32-35
     uv run python -m scripts.phase1b_ablation qwen3-8b assemble
     uv run python -m scripts.phase1b_ablation qwen3-8b comet
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import sacrebleu
import torch

from scripts.observe import Budget, Progress, banner, duration, gpu, log, preview, set_log_file, step
from scripts.paths import guard, result
from scripts.phase0_promptform import SHOTS, clean_completion
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

PROGRESS = result("phase1b-ablation-progress.json")
SWEEP = result("phase1b-ablation-sweep.json")
MEANS = result("phase1b-counterfactual-means.pt")

# The candidate band as a depth fraction, which is what invariant 1 asks for and
# what the absolute `range(27, 36)` it replaces could not honour: 27-35 exists on
# a 36-layer model and nowhere else, so every phase1b script was silently bound to
# one checkpoint. 0.75-1.0 is the same nine layers on the 8B (cfg.layer(0.75) is
# 27, cfg.layer(1.0) is the last) and the proportionally equivalent band anywhere
# else -- seven layers, 21-27, on a 28-layer model.
CANDIDATE_BAND = (0.75, 1.0)

def candidate_layers(cfg) -> list:
    """The candidate band resolved against the model actually loaded"""
    return list(range(cfg.layer(CANDIDATE_BAND[0]), cfg.layer(CANDIDATE_BAND[1]) + 1))

# The 8B band, for the bookkeeping that has no adapter to ask. Anything holding a
# cfg must call candidate_layers(cfg) instead: this is the old answer, kept only
# where the new question cannot be asked.
CANDIDATE_LAYERS = list(range(27, 36))
# How many sentences a score is computed on, and the reason it is settable.
# At 200 the 95% CI on a 1.35 dBLEU effect is 2.33 -- the interval nearly touches
# zero, which is why the strongest component in the whole sweep only reached
# p=0.012 and why nothing survived correction for the 18 tested. The shortlist
# holds 497 usable pairs and the CI falls as 1/sqrt(n): 497 takes it to 1.48.
#
# The default stays at 200 so that every score already on disk keeps its meaning
# and the analysis stages over them keep running. A better-powered protocol is a
# different experiment, not a correction to this one, and it gets its own results
# directory rather than overwriting the numbers it supersedes:
#
#   MI_LAB_RESULTS=results/qwen3-1.7b MI_LAB_EVAL_SENTENCES=497 \
#       uv run python -m scripts.phase1b_ablation qwen3-1.7b sweep layers 3600
#
# Heads are scored on half, as they always were.
EVAL_SENTENCES = int(os.environ.get("MI_LAB_EVAL_SENTENCES", "200"))
HEAD_SENTENCES = EVAL_SENTENCES // 2
# One generation pass costs the weights once and the KV cache once per row, and
# on this GB10 the weights dominate: 15.3 GiB of them against 144 KiB/token of
# KV. At 32 a 200-sentence pass re-streamed those 15.3 GiB seven times to carry
# 1.2 GiB of cache, which is the whole pass paying bandwidth for the model and
# almost none for the data. 100 makes it two passes at ~3.6 GiB of KV, and the
# ceiling is not arithmetic but the neighbours: this box shares its unified
# memory with the sglang and vLLM servers, and MemAvailable sat near 13 GiB
# while this was chosen. 200 (one pass, ~7.3 GiB) fits the arithmetic and not
# the margin. Raise it only after reading `gpu()` from a live run.
#
# Changing this changes the padding of every batch, and bf16 padding changes
# generations, so a baseline captured at one value is not the baseline for
# another: the progress files cache theirs, and the cache has to be dropped
# with the constant or dBLEU spans two numeric regimes.
GENERATION_BATCH = 100
# The mean capture does not get to ride on GENERATION_BATCH. Its hooks hold one
# activation per layer for every layer at once -- 2 x [batch, 256, 4096] over 36
# layers -- so its peak scales with the batch where a generation's KV does, only
# steeper: the whole-stack capture is ~0.14 GiB per row against generation's
# ~0.036. Inheriting a raised generation batch would silently triple a pass that
# already sits at 4.5 GiB of live activations, and it buys nothing, because the
# capture runs once and is cached to disk.
MEANS_BATCH = 32
MAX_NEW_TOKENS = 64

# A dBLEU is a difference of two corpus scores on 200 sentences, and the split-half
# check (phase1b_splithalf) measured that difference's own spread at roughly +-1.5
# for a layer-level component -- wider than every effect this sweep found. So a
# ranking by dBLEU alone ranks noise, and nothing downstream may consume it before
# a paired bootstrap has said which entries are distinguishable from the baseline
# at all. Paired because the ablated and un-ablated systems translate the *same*
# sentences: the pairing removes the sentence-sampling variance that dominates the
# unpaired comparison, and it is the only reason 200 sentences can say anything.
SIGNIFICANCE_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 1000

def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"components": {}}

def save_progress(state: dict) -> None:
    PROGRESS.parent.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

def eval_data():
    wmt = load_pairs(str(default_pairs_path("wmt-newstest2013-es-en-500")))
    shots = wmt[-SHOTS:]
    eval_pairs = wmt[:EVAL_SENTENCES]
    prompts = [translation_prompt(spanish, form="few_shot", shots=shots) for spanish, _ in eval_pairs]
    references = [english for _, english in eval_pairs]
    return wmt, prompts, references

def reference_prompts(wmt):
    """The counterfactual distribution the ablation means are averaged over

    Every non-shot source in the dev subset, wrapped in the counterfactual form
    -- the eval prompt with its translation logic taken out and nothing else
    changed. The mean has to strip the task and preserve everything else.
    """
    shots = wmt[-SHOTS:]
    return [translation_prompt(spanish, form="counterfactual", shots=shots) for spanish, _ in wmt[:-SHOTS]]

def means_path(layers) -> Path:
    """Where a captured mean lands, keyed by the layers it covers

    The candidate band keeps the original filename so the cache captured for
    the sweep stays valid; any other span gets its own file, because a mean
    over one set of layers is not a mean over another and silently reusing the
    cache would ablate a layer toward a number never measured on it.
    """
    if list(layers) == CANDIDATE_LAYERS:
        return MEANS
    return MEANS.with_name(f"phase1b-counterfactual-means-L{min(layers)}-{max(layers)}.pt")

def _geometry(adapter) -> dict:
    """What a mean is a mean *of*, so a cache cannot be read onto the wrong model"""
    return {"id": adapter.cfg.id, "n_layers": adapter.cfg.n_layers,
            "n_heads": adapter.cfg.n_heads, "d_model": adapter.cfg.d_model}

def _checked(means: dict, adapter, source) -> dict:
    """A cached mean is only this model's if the model it was captured on matches

    The reuse rule below matches caches by *layer index*, which is a coincidence
    of numbering rather than a statement about the network: a 36-layer cache
    covers layer 27 and so does any other model with 28 or more layers. Loading
    one across models ablates toward activations of a different width and a
    different head count, and where the widths happen to agree it produces a
    number instead of an error. The geometry is stamped at capture and checked
    here so that stops being possible.

    A cache written before the stamp existed has no geometry and is taken on
    trust with a warning: the alternative is invalidating a capture that is
    almost certainly fine, and the directory it lives in is now guarded by
    config anyway.
    """
    stored = means.get("model")
    if stored is None:
        log(f"warning: {source} predates the model stamp; assuming it belongs to "
            f"'{adapter.cfg.id}' because {source.parent}/ is stamped for it")
        return means
    here = _geometry(adapter)
    if stored != here:
        raise SystemExit(
            f"{source} holds means captured on {stored}, and this is {here}. Ablating toward another "
            "model's activations is not an experiment with a worse number in it, it is a different "
            "quantity -- delete the cache or point MI_LAB_RESULTS somewhere else."
        )
    return means

def capture_means(adapter, texts, layers=None) -> dict:
    """Mean head output [n_heads, d_head] and mean MLP write [d_model] per layer, over real tokens

    `layers` defaults to the candidate band. The random-component control arm
    passes the whole stack, because a control drawn only from the layers 1a
    already flagged asks "are these the right components in the right layers"
    rather than "is the localization real at all".
    """
    layers = CANDIDATE_LAYERS if layers is None else list(layers)
    cache = means_path(layers)
    if cache.exists():
        return _checked(torch.load(cache, weights_only=True), adapter, cache)
    # a cache covering more layers already answers this question: slice it rather
    # than spending another capture pass. It also keeps two scopes comparable --
    # means captured in separate passes over the same prompts are the same number
    # twice, but only one of them is the number the other scope was scored against
    for other in sorted(MEANS.parent.glob("phase1b-counterfactual-means*.pt")):
        candidate = _checked(torch.load(other, weights_only=True), adapter, other)
        if all(layer in candidate["heads"] and layer in candidate["mlps"] for layer in layers):
            log(f"reusing {other} for layers {min(layers)}-{max(layers)} (superset on disk)")
            return {
                "heads": {layer: candidate["heads"][layer] for layer in layers},
                "mlps": {layer: candidate["mlps"][layer] for layer in layers},
                "tokens": candidate["tokens"],
            }
    sums = {"heads": {}, "mlps": {}}
    tokens = 0
    grabbed = {}
    handles = []

    def head_hook(layer):
        def hook(module, args):
            grabbed["heads", layer] = args[0].detach()
        return hook

    def mlp_hook(layer):
        def hook(module, args, output):
            grabbed["mlps", layer] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    for layer in layers:
        handles.append(adapter.projections[layer].register_forward_pre_hook(head_hook(layer)))
        handles.append(adapter.mlps[layer].register_forward_hook(mlp_hook(layer)))
    batches = (len(texts) + MEANS_BATCH - 1) // MEANS_BATCH
    log(f"capturing counterfactual means: {len(texts)} prompts, layers "
        f"{min(layers)}-{max(layers)}, {batches} batches of {MEANS_BATCH} -> {cache}")
    bar = Progress(batches, "means", every=max(1, batches // 10))
    try:
        for start in range(0, len(texts), MEANS_BATCH):
            batch = texts[start : start + MEANS_BATCH]
            adapter.tokenizer.padding_side = "right"
            encoded = adapter.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            ids = encoded["input_ids"].to(adapter.model.device)
            mask = encoded["attention_mask"].to(adapter.model.device)
            with torch.no_grad():
                adapter.model(ids, attention_mask=mask, use_cache=False)
            weights = mask[..., None].float()
            for layer in layers:
                for kind in ("heads", "mlps"):
                    value = (grabbed[kind, layer].float() * weights).sum(dim=(0, 1)).cpu()
                    sums[kind][layer] = sums[kind].get(layer, torch.zeros_like(value)) + value.double()
            tokens += int(mask.sum())
            bar.tick(f"{tokens} tokens")
    finally:
        for handle in handles:
            handle.remove()
    means = {
        "heads": {layer: (sums["heads"][layer] / tokens).float().reshape(adapter.cfg.n_heads, adapter.cfg.d_head)
                  for layer in layers},
        "mlps": {layer: (sums["mlps"][layer] / tokens).float() for layer in layers},
        "tokens": tokens,
        "model": _geometry(adapter),
    }
    bar.finish()
    torch.save(means, cache)
    log(f"means saved: {tokens} tokens over {len(layers)} layers -> {cache} "
        f"({cache.stat().st_size / 1024 ** 2:.0f} MiB)")
    return means

def benjamini_hochberg(pvalues: dict) -> dict:
    """FDR-corrected q-values, because 18 tests at alpha .05 buy ~1 false positive for free

    Reported beside the raw p rather than instead of it. The raw value answers
    "would this component look real if it were the only one tested", which is
    the question a reader of a single row asks; the q-value answers "does it
    look real given that 18 rows were screened to find it", which is the
    question the sweep actually poses. They disagree here, and a table showing
    only one of them would be arguing for a conclusion rather than reporting.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    qvalues = {}
    running = 1.0
    for rank in range(total, 0, -1):
        cid, pvalue = ordered[rank - 1]
        running = min(running, pvalue * total / rank)
        qvalues[cid] = round(running, 5)
    return qvalues

def stage_significance(config: str) -> None:
    """Paired bootstrap of every component with stored generations against the baseline

    No GPU: the generations are already on disk, and a significance test on them
    is arithmetic. Single heads are absent by construction -- stage_sweep stores
    no hypotheses for them -- and that absence is the finding rather than a gap
    to fill: phase1b_splithalf scored 30 of them on a disjoint half and the
    ranking did not reproduce (rho -0.22), so there is nothing there to test.
    """
    from sacrebleu.metrics import BLEU
    from sacrebleu.significance import PairedTest

    set_log_file(result("phase1b-ablation.log"))
    state = load_progress()
    if "baseline" not in state:
        raise SystemExit(f"{PROGRESS} has no baseline; run the sweep first")
    _, _, references = eval_data()
    testable = {cid: record["hypotheses"] for cid, record in state["components"].items()
                if len(record.get("hypotheses") or []) >= EVAL_SENTENCES}
    if not testable:
        raise SystemExit("no component stored a full set of generations, so nothing can be tested")
    banner("phase1b ablation significance", {
        "config": config,
        "testable": f"{len(testable)} of {len(state['components'])} components have stored generations",
        "not testable": f"{len(state['components']) - len(testable)} single heads (no generations stored; "
                        "phase1b_splithalf found their ranking does not reproduce)",
        "test": f"paired bootstrap, {BOOTSTRAP_RESAMPLES} resamples, {EVAL_SENTENCES} sentences",
        "alpha": SIGNIFICANCE_ALPHA,
    })
    names = sorted(testable, key=lambda cid: -state["components"][cid]["dbleu"])
    outcome = PairedTest(
        [("baseline", state["baseline"]["hypotheses"][:EVAL_SENTENCES]),
         *[(cid, testable[cid][:EVAL_SENTENCES]) for cid in names]],
        {"bleu": BLEU()}, references=[references[:EVAL_SENTENCES]],
        test_type="bs", n_samples=BOOTSTRAP_RESAMPLES,
    )()
    scores = outcome[1]["BLEU"]
    raw = {cid: float(entry.p_value) for cid, entry in zip(names, scores[1:], strict=True)
           if entry.p_value is not None}
    qvalues = benjamini_hochberg(raw)
    significance = {}
    for cid, entry in zip(names, scores[1:], strict=True):
        significance[cid] = {
            "bleu": round(float(entry.score), 2),
            "dbleu": round(float(scores[0].score) - float(entry.score), 2),
            "p": round(raw[cid], 5) if cid in raw else None,
            "q": qvalues.get(cid),
            "significant": bool(cid in raw and raw[cid] < SIGNIFICANCE_ALPHA),
            "significant_fdr": bool(cid in qvalues and qvalues[cid] < SIGNIFICANCE_ALPHA),
        }
    state["significance"] = {
        "alpha": SIGNIFICANCE_ALPHA,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sentences": EVAL_SENTENCES,
        "baseline_bleu": round(float(scores[0].score), 2),
        "components": significance,
    }
    save_progress(state)
    log(f"{'component':<12} {'dBLEU':>7} {'p':>8} {'q(FDR)':>8}   verdict")
    for cid in names:
        record = significance[cid]
        verdict = ("significant (FDR)" if record["significant_fdr"]
                   else "raw only" if record["significant"] else "not significant")
        log(f"{cid:<12} {record['dbleu']:>7.2f} {record['p']!s:>8} {record['q']!s:>8}   {verdict}")
    passing = [cid for cid in names if significance[cid]["significant"]]
    fdr = [cid for cid in names if significance[cid]["significant_fdr"]]
    log(f"raw p < {SIGNIFICANCE_ALPHA}: {len(passing)} -> {passing}")
    log(f"FDR q < {SIGNIFICANCE_ALPHA}: {len(fdr)} -> {fdr}")
    if not fdr:
        log("!! nothing survives correction for the number of components screened. The raw-p survivors are "
            "the best candidates this sweep has, and they are not an established result -- treat them as a "
            "hypothesis to test on more data, not as a discovered circuit.")

def parse_component(cid: str):
    """'mlp:31' | 'heads:31' | 'head:31:7' -> the hook targets it names"""
    parts = cid.split(":")
    kind, layer = parts[0], int(parts[1])
    head = int(parts[2]) if len(parts) > 2 else None
    return kind, layer, head

@contextmanager
def ablate(adapter, means: dict, components):
    """Replace each named component's activation with its counterfactual mean, for the duration"""
    handles = []
    by_layer_heads: dict = {}
    mlp_layers = []
    for cid in components:
        kind, layer, head = parse_component(cid)
        if kind == "mlp":
            mlp_layers.append(layer)
        elif kind == "heads":
            by_layer_heads.setdefault(layer, set()).update(range(adapter.cfg.n_heads))
        elif kind == "head":
            by_layer_heads.setdefault(layer, set()).add(head)
        else:
            raise ValueError(f"unknown component '{cid}'")

    def head_hook(layer, heads):
        mean = means["heads"][layer]
        def hook(module, args):
            merged = args[0].clone()
            width = adapter.cfg.d_head
            for head in heads:
                merged[..., head * width : (head + 1) * width] = mean[head].to(merged.device, merged.dtype)
            return (merged, *args[1:])
        return hook

    def mlp_hook(layer):
        mean = means["mlps"][layer]
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            replaced = mean.to(hidden.device, hidden.dtype).expand_as(hidden)
            return (replaced, *output[1:]) if isinstance(output, tuple) else replaced
        return hook

    for layer, heads in by_layer_heads.items():
        handles.append(adapter.projections[layer].register_forward_pre_hook(head_hook(layer, sorted(heads))))
    for layer in mlp_layers:
        handles.append(adapter.mlps[layer].register_forward_hook(mlp_hook(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def translate(adapter, prompts, label: str = "translate"):
    """Generate, chunked here rather than inside the adapter, so the pass reports progress

    adapter.generate batches internally and returns only when every prompt is
    done, which makes a 200-sentence pass a single silent minute. Chunking at
    this level produces the same completions in the same order and lets the
    loop say where it is.
    """
    chunks = [prompts[start : start + GENERATION_BATCH] for start in range(0, len(prompts), GENERATION_BATCH)]
    bar = Progress(len(chunks), label, indent=2)
    done = []
    for chunk in chunks:
        done.extend(clean_completion(text) for text in adapter.generate(chunk, max_new_tokens=MAX_NEW_TOKENS))
        bar.tick(f"{len(done)}/{len(prompts)} sentences")
    return done

def bleu_of(hypotheses, references):
    return round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)

def component_plan(group: str, layers=None, heads: int = 32):
    layers = list(CANDIDATE_LAYERS) if layers is None else list(layers)
    if group == "layers":
        return [f"mlp:{layer}" for layer in layers] + [f"heads:{layer}" for layer in layers]
    if group == "heads32-35":
        # 1a neuron-density order (33: 325 flagged, 34: 317, 35: 210, 32: 180), so a
        # truncated sweep has done the most promising layers first. The order is a
        # fact about the 8B's own 1a scan and does not transfer; on another model
        # this is the upper half of the band in descending depth, which is the
        # nearest thing to it that means anything.
        upper = layers[len(layers) // 2:]
        return [f"head:{layer}:{head}" for layer in reversed(upper) for head in range(heads)]
    if group == "heads27-31":
        return [f"head:{layer}:{head}" for layer in layers[:len(layers) // 2] for head in range(heads)]
    raise SystemExit(f"unknown group '{group}'; groups are layers, heads32-35, heads27-31")

def _migrate_baseline(baseline: dict) -> dict:
    """Read a baseline written before the counts were part of the record

    The old keys named the numbers they held -- bleu_200, bleu_100 -- which was
    honest while the counts were constants and became a lie the moment they were
    not. Everything already on disk was scored at 200/100, so that is what the
    old names are read as.
    """
    if "bleu_eval" not in baseline and "bleu_200" in baseline:
        baseline = {**baseline, "bleu_eval": baseline["bleu_200"],
                    "bleu_head": baseline["bleu_100"], "sentences": {"eval": 200, "head": 100}}
    return baseline

def ensure_baseline(state, adapter, prompts, references):
    if "baseline" in state:
        state["baseline"] = _migrate_baseline(state["baseline"])
        counts = state["baseline"].get("sentences", {})
        if counts and (counts["eval"], counts["head"]) != (EVAL_SENTENCES, HEAD_SENTENCES):
            raise SystemExit(
                f"the cached baseline was scored on {counts['eval']}/{counts['head']} sentences and this "
                f"run wants {EVAL_SENTENCES}/{HEAD_SENTENCES}. Every component score in this file is a "
                f"difference against that baseline, so mixing the two would compare drops measured on "
                f"different corpora. Point MI_LAB_RESULTS at a fresh directory for the new protocol."
            )
        log(f"baseline already scored: BLEU {state['baseline']['bleu_eval']} (cached)")
        return
    start = time.time()
    with step(f"baseline: {len(prompts)} sentences, un-ablated"):
        hypotheses = translate(adapter, prompts, label="baseline")
    state["baseline"] = {
        "bleu_eval": bleu_of(hypotheses, references),
        "bleu_head": bleu_of(hypotheses[:HEAD_SENTENCES], references[:HEAD_SENTENCES]),
        "sentences": {"eval": EVAL_SENTENCES, "head": HEAD_SENTENCES},
        "seconds": round(time.time() - start, 1),
        "hypotheses": hypotheses,
    }
    save_progress(state)
    log(f"baseline BLEU {state['baseline']['bleu_eval']} at {EVAL_SENTENCES} / "
        f"{state['baseline']['bleu_head']} at {HEAD_SENTENCES} "
        f"({duration(state['baseline']['seconds'])}/pass)")
    preview(hypotheses, "baseline")

def stage_sweep(config: str, group: str, budget: float) -> None:
    set_log_file(result("phase1b-ablation.log"))
    allowance = Budget(budget)
    state = load_progress()
    # the plan is resolved twice: once against the 8B band so the banner can be
    # printed before a checkpoint load blocks for minutes, and again against the
    # model once it is up. They agree on the 8B and differ everywhere else, and
    # the second one is the one that runs.
    whole = component_plan(group)
    plan = [cid for cid in whole if cid not in state["components"]]
    banner("phase1b ablation sweep", {
        "config": config,
        "group": group,
        "components": f"{len(plan)} to do of {len(whole)} in group ({len(state['components'])} already scored)",
        "budget": duration(budget),
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    if not plan:
        log(f"group '{group}' already swept -- nothing to do")
        return
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    layers = candidate_layers(adapter.cfg)
    whole = component_plan(group, layers, adapter.cfg.n_heads)
    plan = [cid for cid in whole if cid not in state["components"]]
    log(f"candidate band {CANDIDATE_BAND[0]:.2f}-{CANDIDATE_BAND[1]:.2f} of depth on "
        f"{adapter.cfg.n_layers} layers -> {layers[0]}-{layers[-1]} ({len(layers)} layers, "
        f"{adapter.cfg.n_heads} heads each) · {len(plan)} components to do")
    if not plan:
        log(f"group '{group}' already swept -- nothing to do")
        return
    wmt, prompts, references = eval_data()
    means = capture_means(adapter, reference_prompts(wmt), layers=layers)
    ensure_baseline(state, adapter, prompts, references)

    seen = []
    for index, cid in enumerate(plan):
        left = [c for c in plan if c not in state["components"]]
        estimate = sum(seen) / len(seen) if seen else state["baseline"]["seconds"]
        if not allowance.fits(estimate):
            log(f"stopping cleanly: {allowance.state()}, next component needs ~{duration(estimate)}")
            log(f"{len(left)} of {len(plan)} left in group '{group}' -- re-run the same command to resume")
            return
        kind, _, _ = parse_component(cid)
        count = EVAL_SENTENCES if kind in ("mlp", "heads") else HEAD_SENTENCES
        start = time.time()
        log(f"[{index + 1}/{len(plan)}] {cid} · {count} sentences · {allowance.state()} · "
            f"eta for group {duration(estimate * len(left))}")
        with ablate(adapter, means, [cid]):
            hypotheses = translate(adapter, prompts[:count], label=cid)
        seen.append(time.time() - start)
        bleu = bleu_of(hypotheses, references[:count])
        base = state["baseline"]["bleu_eval"] if count == EVAL_SENTENCES else state["baseline"]["bleu_head"]
        state["components"][cid] = {
            "bleu": bleu,
            "dbleu": round(base - bleu, 2),
            "sentences": count,
            "seconds": round(time.time() - start, 1),
            "hypotheses": hypotheses if kind in ("mlp", "heads") else hypotheses[:0],
        }
        save_progress(state)
        log(f"{cid}: BLEU {bleu} (d {round(base - bleu, 2)}) in {duration(time.time() - start)} · {gpu()}")
        preview(hypotheses, cid)
    log(f"group '{group}' done: {len(plan)} components in {duration(allowance.spent)}")

def stage_assemble(command: str) -> None:
    state = load_progress()
    ranked = sorted(
        (
            {"component": cid, **{key: value for key, value in record.items() if key != "hypotheses"}}
            for cid, record in state["components"].items()
        ),
        key=lambda record: -record["dbleu"],
    )
    SWEEP.write_text(json.dumps({
        "protocol": "mean ablation (Wang et al. 2211.00593 knockout) per component, mean taken over the "
                    "counterfactual prompt distribution -- eval prompt with the translation logic removed; "
                    "greedy translation of the WMT-200 shortlist (few-shot, 64 new tokens); "
                    "score = baseline corpus BLEU minus ablated corpus BLEU. Layer-level components on "
                    f"{EVAL_SENTENCES} sentences, single heads on {HEAD_SENTENCES}.",
        "baseline": {key: value for key, value in state["baseline"].items() if key != "hypotheses"},
        "counterfactual_mean_tokens": int(torch.load(MEANS, weights_only=True)["tokens"]) if MEANS.exists() else None,
        "components_scored": len(ranked),
        "wall_seconds_total": round(sum(record["seconds"] for record in ranked)
                                    + state["baseline"]["seconds"], 1),
        "ranked": ranked,
        "comet_top": state.get("comet_top"),
        "command": command,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(ranked)} components -> {SWEEP}; top 5: "
          + ", ".join(f"{r['component']} d{r['dbleu']}" for r in ranked[:5]))

def stage_comet(config: str, top: int) -> None:
    """dCOMET for the strongest solo components: regenerate at 200 sentences, score against baseline"""
    state = load_progress()
    ranked = sorted(state["components"].items(), key=lambda item: -item[1]["dbleu"])[:top]
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    sources = [spanish for spanish, _ in wmt[:EVAL_SENTENCES]]
    means = capture_means(adapter, reference_prompts(wmt))

    from comet import download_model, load_from_checkpoint
    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def score(hypotheses):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, hypotheses, references, strict=True)]
        return round(float(comet.predict(data, batch_size=16, gpus=1).system_score), 4)

    results = {"baseline": score(state["baseline"]["hypotheses"])}
    for cid, record in ranked:
        hypotheses = record["hypotheses"]
        if len(hypotheses) < EVAL_SENTENCES:
            with ablate(adapter, means, [cid]):
                hypotheses = translate(adapter, prompts)
        results[cid] = score(hypotheses)
        print(cid, results[cid], flush=True)
    state["comet_top"] = {
        "scores": results,
        "dcomet": {cid: round(results["baseline"] - value, 4) for cid, value in results.items() if cid != "baseline"},
    }
    save_progress(state)

def main() -> None:
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    stage = sys.argv[2] if len(sys.argv) > 2 else "sweep"
    guard(config)
    if stage == "sweep":
        group = sys.argv[3] if len(sys.argv) > 3 else "layers"
        budget = float(sys.argv[4]) if len(sys.argv) > 4 else 420.0
        stage_sweep(config, group, budget)
    elif stage == "assemble":
        stage_assemble(f"uv run python -m scripts.phase1b_ablation {config} sweep <group> (chained), then assemble")
    elif stage == "significance":
        stage_significance(config)
    elif stage == "comet":
        stage_comet(config, top=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        raise SystemExit(f"unknown stage '{stage}'")

if __name__ == "__main__":
    main()
