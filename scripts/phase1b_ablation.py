"""Phase 1b deliverable 4: the mean-ablation sweep over the candidate components.

Zhang et al. protocol, priced for this GPU: each candidate component (an
attention head or an MLP block in layers 27-35) is replaced by its mean
activation under the EN-only control condition (the English side of the
500-sentence WMT dev subset), the ES->EN translation is regenerated on the
200-sentence WMT shortlist, and the drop in corpus BLEU against the
un-ablated baseline is the component's score.

Engineering constraints, all deliberate:
- resumable: every finished component is flushed to
  results/phase1b-ablation-progress.json, and a --budget wall-clock limit
  makes each invocation exit cleanly, so the sweep runs as a chain of
  foreground calls with nothing orphanable in the background;
- layer-level components (a whole layer's heads, one MLP) are scored on all
  200 sentences; single heads on 100 -- a solo head rarely moves corpus BLEU
  by more than noise, so the halved pass buys double the coverage and the
  greedy stage re-verifies everything it uses at 200;
- the EN-control means are captured once and cached beside the progress file.

Stages: sweep (default) walks a component group; assemble writes the ranked
results/phase1b-ablation-sweep.json; comet adds dCOMET for the top solo
components.

Run: uv run python -m scripts.phase1b_ablation qwen3-8b sweep layers
     uv run python -m scripts.phase1b_ablation qwen3-8b sweep heads32-35
     uv run python -m scripts.phase1b_ablation qwen3-8b assemble
     uv run python -m scripts.phase1b_ablation qwen3-8b comet
"""

import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import sacrebleu
import torch

from scripts.phase0_promptform import SHOTS, clean_completion
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

PROGRESS = Path("results/phase1b-ablation-progress.json")
SWEEP = Path("results/phase1b-ablation-sweep.json")
MEANS = Path("results/phase1b-en-means.pt")

CANDIDATE_LAYERS = list(range(27, 36))
EVAL_SENTENCES = 200
HEAD_SENTENCES = 100
GENERATION_BATCH = 32
MAX_NEW_TOKENS = 64

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

def capture_means(adapter, texts) -> dict:
    """Mean head output [n_heads, d_head] and mean MLP write [d_model] per candidate layer, over real tokens"""
    if MEANS.exists():
        return torch.load(MEANS, weights_only=True)
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

    for layer in CANDIDATE_LAYERS:
        handles.append(adapter.projections[layer].register_forward_pre_hook(head_hook(layer)))
        handles.append(adapter.mlps[layer].register_forward_hook(mlp_hook(layer)))
    try:
        for start in range(0, len(texts), adapter.cfg.batch_size):
            batch = texts[start : start + adapter.cfg.batch_size]
            adapter.tokenizer.padding_side = "right"
            encoded = adapter.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            ids = encoded["input_ids"].to(adapter.model.device)
            mask = encoded["attention_mask"].to(adapter.model.device)
            with torch.no_grad():
                adapter.model(ids, attention_mask=mask, use_cache=False)
            weights = mask[..., None].float()
            for layer in CANDIDATE_LAYERS:
                for kind in ("heads", "mlps"):
                    value = (grabbed[kind, layer].float() * weights).sum(dim=(0, 1)).cpu()
                    sums[kind][layer] = sums[kind].get(layer, torch.zeros_like(value)) + value.double()
            tokens += int(mask.sum())
    finally:
        for handle in handles:
            handle.remove()
    means = {
        "heads": {layer: (sums["heads"][layer] / tokens).float().reshape(adapter.cfg.n_heads, adapter.cfg.d_head)
                  for layer in CANDIDATE_LAYERS},
        "mlps": {layer: (sums["mlps"][layer] / tokens).float() for layer in CANDIDATE_LAYERS},
        "tokens": tokens,
    }
    torch.save(means, MEANS)
    return means

def parse_component(cid: str):
    """'mlp:31' | 'heads:31' | 'head:31:7' -> the hook targets it names"""
    parts = cid.split(":")
    kind, layer = parts[0], int(parts[1])
    head = int(parts[2]) if len(parts) > 2 else None
    return kind, layer, head

@contextmanager
def ablate(adapter, means: dict, components):
    """Replace each named component's activation with its EN-control mean, for the duration"""
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

def translate(adapter, prompts):
    return [clean_completion(text) for text in adapter.generate(prompts, max_new_tokens=MAX_NEW_TOKENS)]

def bleu_of(hypotheses, references):
    return round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)

def component_plan(group: str):
    if group == "layers":
        return [f"mlp:{layer}" for layer in CANDIDATE_LAYERS] + [f"heads:{layer}" for layer in CANDIDATE_LAYERS]
    if group == "heads32-35":
        # 1a neuron-density order (33: 325 flagged, 34: 317, 35: 210, 32: 180), so a
        # truncated sweep has done the most promising layers first
        return [f"head:{layer}:{head}" for layer in (33, 34, 35, 32) for head in range(32)]
    if group == "heads27-31":
        return [f"head:{layer}:{head}" for layer in range(27, 32) for head in range(32)]
    raise SystemExit(f"unknown group '{group}'; groups are layers, heads32-35, heads27-31")

def ensure_baseline(state, adapter, prompts, references):
    if "baseline" in state:
        return
    start = time.time()
    hypotheses = translate(adapter, prompts)
    state["baseline"] = {
        "bleu_200": bleu_of(hypotheses, references),
        "bleu_100": bleu_of(hypotheses[:HEAD_SENTENCES], references[:HEAD_SENTENCES]),
        "seconds": round(time.time() - start, 1),
        "hypotheses": hypotheses,
    }
    save_progress(state)
    print(f"baseline BLEU {state['baseline']['bleu_200']} ({state['baseline']['seconds']}s/pass)", flush=True)

def stage_sweep(config: str, group: str, budget: float) -> None:
    start_all = time.time()
    state = load_progress()
    plan = [cid for cid in component_plan(group) if cid not in state["components"]]
    if not plan:
        print(f"group '{group}' already swept")
        return
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    means = capture_means(adapter, [english for _, english in wmt])
    ensure_baseline(state, adapter, prompts, references)

    for cid in plan:
        if time.time() - start_all > budget:
            print(f"budget reached with {len([c for c in plan if c not in state['components']])} left", flush=True)
            return
        kind, _, _ = parse_component(cid)
        count = EVAL_SENTENCES if kind in ("mlp", "heads") else HEAD_SENTENCES
        start = time.time()
        with ablate(adapter, means, [cid]):
            hypotheses = translate(adapter, prompts[:count])
        bleu = bleu_of(hypotheses, references[:count])
        base = state["baseline"]["bleu_200"] if count == EVAL_SENTENCES else state["baseline"]["bleu_100"]
        state["components"][cid] = {
            "bleu": bleu,
            "dbleu": round(base - bleu, 2),
            "sentences": count,
            "seconds": round(time.time() - start, 1),
            "hypotheses": hypotheses if kind in ("mlp", "heads") else hypotheses[:0],
        }
        save_progress(state)
        print(f"{cid}: BLEU {bleu} (d {round(base - bleu, 2)}) in {round(time.time() - start)}s", flush=True)
    print("group done", flush=True)

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
        "protocol": "mean ablation (EN-control mean from the WMT-500 English side) per component; "
                    "greedy translation of the WMT-200 shortlist (few-shot, 64 new tokens); "
                    "score = baseline corpus BLEU minus ablated corpus BLEU. Layer-level components on "
                    f"{EVAL_SENTENCES} sentences, single heads on {HEAD_SENTENCES}.",
        "baseline": {key: value for key, value in state["baseline"].items() if key != "hypotheses"},
        "en_control_mean_tokens": int(torch.load(MEANS, weights_only=True)["tokens"]) if MEANS.exists() else None,
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
    means = capture_means(adapter, [english for _, english in wmt])

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
    if stage == "sweep":
        group = sys.argv[3] if len(sys.argv) > 3 else "layers"
        budget = float(sys.argv[4]) if len(sys.argv) > 4 else 420.0
        stage_sweep(config, group, budget)
    elif stage == "assemble":
        stage_assemble(f"uv run python -m scripts.phase1b_ablation {config} sweep <group> (chained), then assemble")
    elif stage == "comet":
        stage_comet(config, top=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        raise SystemExit(f"unknown stage '{stage}'")

if __name__ == "__main__":
    main()
