"""Phase 3: decompose a stage's update into pieces, and find the pieces the forgetting is made of

The parameter-decomposition answer to the question `phase3_delta_rank` asks in
the plain singular basis. The science stage's update D = W_science - W_tooluse
is written, map by map, as rank-1 subcomponents plus a leftover, each
subcomponent switched per position by a learned gate (`analysis/delta.py`). The
earlier checkpoint stays fixed underneath: every subcomponent off is the
tool-use model, every one on is the science model.

**train**: on a mix of both tasks' training demonstrations and wikitext, the
decomposed model has to reproduce the science model's next-token distribution
(KL) under three kinds of mask --

- *stochastic*: each subcomponent's mask drawn between its gate and one, the
  leftover's between zero and one (SPD);
- *adversarial*: the same range, with the draw pushed by a few signed-gradient
  steps toward the masks that hurt most, shared across the batch so the
  adversary finds flaws in the decomposition rather than in one sequence (VPD);
- with a penalty on how many gates are open (sum |g|^p, p annealed down).

**analyse**: on the held-out eval answers --

- how often each subcomponent's gate opens on the earlier task, the later task
  and prose; a subcomponent open on tool-use answers and shut on science ones is
  a piece of the update the science stage did not need there, which the
  tool-use task nonetheless runs through;
- faithfulness: the gated model (each pass masked by its own gates), the
  subcomponents without the leftover, the leftover alone;
- **reverting**: the n subcomponents most specific to the earlier task set back
  to the tool-use model, everything else left at the science model, scored for
  forgetting kept and learning kept (`analysis/stage.py`); against the same
  number chosen at random among the live ones, and chosen by use on the earlier
  task alone. If the specific ones take the forgetting back and leave the
  learning, the forgetting is separable from the learning in parameter space,
  and these are the pieces that separate it;
- for the most specific subcomponents, the tool-use tokens their gates open on.

Run: uv run python -m scripts.phase3_delta_pd qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science
     ... --stage=analyse    (reuse a trained decomposition)
     ... --stage=export --export_n=256 --export_by=used    (a reverted checkpoint for `make eval`)
     ... --steps=200 --components=32    (a smoke run)
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List

from src.core.config import load_config
from src.domains.lm.analysis import delta, stage
from src.domains.lm.data.sdft import training_texts
from src.domains.lm.data.wikitext import passages
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import track

# The generation budget the saved eval answers were made under; part of their directory name.
BUDGET = 2048
# Wikitext passages the scoring sets use come first; training takes the ones after.
SCORED_PROSE = 16

DEFAULTS: Dict[str, Any] = {
    "stage": "all", "components": 128, "hidden": 32, "steps": 3000, "batch": 4, "seq": 512,
    "lr": 3e-4, "warmup": 100, "imp": 1e-4, "p_start": 2.0, "p_end": 0.5, "pgd": 1.0, "pgd_steps": 1,
    "pgd_lr": 0.5, "faith": 0.1, "log_every": 25, "later": 80, "seed": 0,
    "export_n": 256, "export_by": "used",
}


def options(argv: List[str]) -> Dict[str, Any]:
    configs = [word for word in argv if not word.startswith("--")]
    if len(configs) != 2:
        raise SystemExit("give two configs: the model before the stage and after it")
    chosen = dict(DEFAULTS, before=configs[0], after=configs[1])
    for word in argv:
        if word.startswith("--"):
            key, _, value = word[2:].partition("=")
            key = key.replace("-", "_")
            if key not in DEFAULTS:
                raise SystemExit(f"unknown flag --{key}; known: {', '.join(DEFAULTS)}")
            chosen[key] = type(DEFAULTS[key])(value)
    return chosen


def build(opts: Dict[str, Any], device):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(load_config(opts["before"]).hf_name)
    model = stage.load(opts["before"], str(device))
    after = stage.load(opts["after"], str(device))
    deltas = delta.updates(model, after, device=str(device))
    del after
    maps = delta.install(model, deltas, dict.fromkeys(deltas, opts["components"]), opts["hidden"])
    return tokenizer, model, maps


def corpus(tokenizer, opts: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Token ids per source, each cut to the training length"""
    def encode(texts):
        return [tokenizer(text, add_special_tokens=False, truncation=True,
                          max_length=opts["seq"], return_tensors="pt")["input_ids"][0] for text in texts]
    tool = encode(training_texts(stage.SDFT_REPO, stage.EARLIER, tokenizer))
    science = encode(training_texts(stage.SDFT_REPO, stage.LATER, tokenizer))
    prose = encode(passages(SCORED_PROSE + max(len(tool), 400))[SCORED_PROSE:])
    return {"earlier": tool, "later": science, "prose": prose}


def batches(sources: Dict[str, List[Any]], size: int, device, generator: random.Random):
    """Right-padded batches drawing from the three sources in turn"""
    import torch

    names = list(sources)
    turn = 0
    while True:
        rows = []
        for _ in range(size):
            pool = sources[names[turn % len(names)]]
            rows.append(pool[generator.randrange(len(pool))])
            turn += 1
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        keep = torch.zeros(len(rows), width)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            keep[i, : len(r)] = 1
        yield ids.to(device), keep.to(device)


def train(model, maps, tokenizer, opts: Dict[str, Any], tracker, device) -> None:
    import torch

    generator = random.Random(opts["seed"])
    torch.manual_seed(opts["seed"])
    with step("training text"):
        sources = corpus(tokenizer, opts)
    log("training sequences: " + ", ".join(f"{k} {len(v)}" for k, v in sources.items()))
    parameters = [p for m in maps.values() for p in m.parameters() if p.requires_grad]
    log(f"trainable parameters: {sum(p.numel() for p in parameters) / 1e6:.1f}M over {len(maps)} maps, "
        f"{sum(m.components for m in maps.values())} subcomponents")
    optimizer = torch.optim.AdamW(parameters, lr=opts["lr"], weight_decay=0.0)
    stream = batches(sources, opts["batch"], device, generator)
    started = time.time()
    for at in range(opts["steps"]):
        scale = delta.cosine_lr(at, opts["steps"], opts["warmup"], 0.1)
        for group in optimizer.param_groups:
            group["lr"] = opts["lr"] * scale
        p = delta.schedule(at, opts["steps"], opts["p_start"], opts["p_end"])
        ids, keep = next(stream)

        with torch.no_grad():
            delta.set_masks(maps, None, None)
            delta.capturing(maps, True)
            target = model(input_ids=ids, attention_mask=keep.long()).logits
        gates, upper = {}, {}
        for name, module in maps.items():
            gates[name], upper[name] = module.gates(module.captured)
        delta.capturing(maps, False)

        def run(masks, rest, target=target, ids=ids, keep=keep):
            delta.set_masks(maps, masks, rest)
            return delta.kl(target, model(input_ids=ids, attention_mask=keep.long()).logits, keep)

        # Stochastic: closed gates take any mask in [g, 1], the leftover any in [0, 1].
        stochastic = run({n: delta.mask_from(g, torch.rand_like(g)) for n, g in gates.items()},
                         {n: torch.rand(ids.shape, device=device) for n in maps})

        # Adversarial: one source per subcomponent, shared across batch and positions.
        sources_c = {n: torch.rand(m.components, device=device, requires_grad=True) for n, m in maps.items()}
        sources_r = {n: torch.rand(1, device=device, requires_grad=True) for n in maps}
        for _ in range(opts["pgd_steps"]):
            loss = run({n: delta.mask_from(g.detach(), sources_c[n]) for n, g in gates.items()},
                       {n: sources_r[n].expand(ids.shape) for n in maps})
            grads = torch.autograd.grad(loss, [*sources_c.values(), *sources_r.values()])
            with torch.no_grad():
                for source, grad in zip([*sources_c.values(), *sources_r.values()], grads, strict=True):
                    source.add_(opts["pgd_lr"] * grad.sign()).clamp_(0.0, 1.0)
        adversarial = run({n: delta.mask_from(g, sources_c[n].detach()) for n, g in gates.items()},
                          {n: sources_r[n].detach().expand(ids.shape) for n in maps})

        importance = delta.minimality(upper, p, keep)
        faithful = delta.faithfulness(maps)
        total = stochastic + opts["pgd"] * adversarial + opts["imp"] * importance + opts["faith"] * faithful
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        delta.set_masks(maps, None, None)

        if at % opts["log_every"] == 0 or at == opts["steps"] - 1:
            with torch.no_grad():
                open_ = sum(((g > 0.5).float().sum(-1) * keep).sum() for g in gates.values()) / keep.sum()
                live = sum(int((g > 0.5).any(0).any(0).sum()) for g in gates.values())
            metrics = {"loss": float(total), "kl_stochastic": float(stochastic), "kl_adversarial": float(adversarial),
                       "importance": float(importance), "leftover_share": float(faithful),
                       "open_per_position": float(open_), "live_in_batch": live, "p": p, "lr_scale": scale}
            tracker.log(at, metrics)
            rate = (at + 1) / (time.time() - started)
            log(f"step {at:5d}  kl stoch {metrics['kl_stochastic']:.4f}  adv {metrics['kl_adversarial']:.4f}  "
                f"imp {metrics['importance']:.0f}  open/pos {metrics['open_per_position']:.0f}  live {live}  "
                f"leftover {metrics['leftover_share']:.3f}  p {p:.2f}  {rate:.2f} step/s")


def save(maps, path) -> None:
    from safetensors.torch import save_file

    state = {f"{name}.{key}": value.detach().contiguous().cpu()
             for name, module in maps.items() for key, value in module.state_dict().items()
             if not key.startswith("base.")}
    save_file(state, str(path))


def restore(maps, path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path))
    for name, module in maps.items():
        mine = {key[len(name) + 1:]: value for key, value in state.items() if key.startswith(name + ".")}
        missing = module.load_state_dict(mine, strict=False)
        unexpected = [k for k in missing.missing_keys if not k.startswith("base.")]
        if unexpected or missing.unexpected_keys:
            raise SystemExit(f"{path} does not match this decomposition at {name}: "
                             f"{unexpected or missing.unexpected_keys}")


def eval_groups(sets) -> Dict[str, List]:
    """The scoring sets as (ids, first position) per group: answers only on the two tasks"""
    return {"earlier": [(e.ids, e.start - 1) for e in sets.earlier],
            "later": [(e.ids, e.start - 1) for e in sets.later],
            "prose": [(ids, 0) for ids in sets.prose]}


def ranking_groups(tokenizer, opts: Dict[str, Any], device, count: int = 64) -> Dict[str, List]:
    """Training demonstrations the ranking reads, so the eval items stay out of the choice

    Ranking subcomponents on the eval answers and then scoring the revert on
    the same answers would select on the test set; these come from both tasks'
    training splits and from wikitext past the scored passages.
    """
    generator = random.Random(opts["seed"] + 1)
    sources = corpus(tokenizer, opts)
    return {name: [(ids.unsqueeze(0).to(device), 0) for ids in generator.sample(pool, min(count, len(pool)))]
            for name, pool in sources.items()}


def gate_statistics(model, maps, groups: Dict[str, List]) -> Dict[str, Any]:
    """Per subcomponent: mean gate and share of positions with the gate past one half, per group"""
    import torch

    names = delta.component_names(maps)
    out = {}
    delta.set_masks(maps, None, None)
    for group, rows in groups.items():
        total = torch.zeros(len(names), device=next(model.parameters()).device)
        opened = torch.zeros_like(total)
        positions = 0
        for ids, start in rows:
            delta.capturing(maps, True)
            with torch.no_grad():
                model(input_ids=ids)
                gates = torch.cat([m.gates(m.captured)[0][0, start:] for m in maps.values()], dim=-1)
            delta.capturing(maps, False)
            total += gates.sum(0)
            opened += (gates > 0.5).float().sum(0)
            positions += gates.shape[0]
        out[group] = {"mean": (total / positions).tolist(), "open": (opened / positions).tolist(),
                      "positions": positions}
    return out


def where_open(model, maps, sets, tokenizer, chosen, per: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """For each chosen subcomponent, the earlier-task positions its gate opens widest on, with context"""
    import torch

    offsets, at = {}, 0
    for name, module in maps.items():
        offsets[name] = at
        at += module.components
    best: Dict[str, List] = {f"{n}:{c}": [] for n, c in chosen}
    for index, entry in enumerate(sets.earlier):
        delta.capturing(maps, True)
        with torch.no_grad():
            model(input_ids=entry.ids)
            for name, c in chosen:
                gate = maps[name].gates(maps[name].captured)[0][0, :, c]
                top = gate.topk(min(per, gate.shape[0]))
                for value, position in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                    best[f"{name}:{c}"].append((value, index, position))
        delta.capturing(maps, False)
    report = {}
    for key, rows in best.items():
        rows.sort(reverse=True)
        shown = []
        for value, index, position in rows[:per]:
            ids = sets.earlier[index].ids[0]
            left = tokenizer.decode(ids[max(0, position - 8): position])
            here = tokenizer.decode(ids[position: position + 1])
            shown.append({"gate": round(value, 3), "item": index, "in_answer": position >= sets.earlier[index].start,
                          "context": f"{left}[[{here}]]"})
        report[key] = shown
    return report


def analyse(model, maps, tokenizer, opts: Dict[str, Any], tracker, device) -> Dict[str, Any]:
    import torch

    sets = stage.build(opts["before"], opts["after"], tokenizer, device, BUDGET,
                       later_limit=opts["later"], prose=SCORED_PROSE)
    log(f"sets: {sets.meta}")
    names = delta.component_names(maps)
    with step("gate statistics"):
        stats = gate_statistics(model, maps, ranking_groups(tokenizer, opts, device))
        held_out = gate_statistics(model, maps, eval_groups(sets))
    f_earlier = torch.tensor(stats["earlier"]["open"])
    f_later = torch.tensor(stats["later"]["open"])
    f_prose = torch.tensor(stats["prose"]["open"])
    alive = ((f_earlier + f_later + f_prose) > 0).nonzero().flatten().tolist()
    log(f"alive on ranking text: {len(alive)} of {len(names)}; open per position: "
        + ", ".join(f"{g} {sum(stats[g]['open']):.0f}" for g in stats))

    rows: Dict[str, Dict[str, List[float]]] = {}
    with torch.no_grad():
        delta.set_masks(maps, None, None)
        ref = stage.reference(model, sets)

    def scored(name: str) -> None:
        with torch.no_grad():
            rows[name] = stage.score(model, sets, ref)
        delta.set_masks(maps, None, None)
        delta.gating(maps, False)

    off = {n: torch.zeros(1, 1, m.components, device=device) for n, m in maps.items()}
    on = {n: torch.ones(1, 1, m.components, device=device) for n, m in maps.items()}
    zero = {n: torch.zeros(1, 1, device=device) for n in maps}
    one = {n: torch.ones(1, 1, device=device) for n in maps}
    with step("faithfulness"):
        delta.set_masks(maps, off, zero)
        scored("before")
        scored("after")
        delta.set_masks(maps, on, zero)
        scored("subcomponents, no leftover")
        delta.set_masks(maps, off, one)
        scored("leftover alone")
        delta.set_masks(maps, None, zero)
        delta.gating(maps, True)
        scored("gated, no leftover")
        delta.gating(maps, True)
        scored("gated, with leftover")

    specificity = f_earlier - f_later
    ranked_specific = [i for i in specificity.argsort(descending=True).tolist() if specificity[i] > 0]
    ranked_earlier = [i for i in f_earlier.argsort(descending=True).tolist() if f_earlier[i] > 0]
    generator = random.Random(opts["seed"])
    sizes = [n for n in (16, 64, 256, 1024, 1568, 4096) if n <= len(alive)]
    plans = []
    for n in sizes:
        if n <= len(ranked_specific):
            plans.append((f"revert {n} earlier-specific", ranked_specific[:n]))
        if n <= len(ranked_earlier):
            plans.append((f"revert {n} most used on earlier", ranked_earlier[:n]))
        plans.append((f"revert {n} random alive", generator.sample(alive, n)))
    reverts = {}
    for label, chosen in plans:
        with step(label):
            delta.revert(maps, [names[i] for i in chosen])
            scored(label)
            entry = stage.fractions(rows, [label], sets, draws=200)[label]
            log(stage.line(label, entry))
            reverts[label] = [f"{names[i][0]}:{names[i][1]}" for i in chosen[:64]]

    variants = [name for name in rows if name not in ("before", "after")]
    fractions = stage.fractions(rows, variants, sets)
    for name in variants:
        tracker.set_tags({f"result.{name}": json.dumps({k: round(v, 4) for k, v in fractions[name].items()
                                                         if isinstance(v, float)})})
    top = [names[i] for i in ranked_specific[:30]]
    with step("where the most specific subcomponents open"):
        contexts = where_open(model, maps, sets, tokenizer, top)
    table = []
    for i in ranked_specific[:30]:
        name, c = names[i]
        layer, short = delta.kind(name)
        module = maps[name]
        weight = float(module.U[c].norm() * module.V[:, c].norm())
        table.append({"component": f"{name}:{c}", "layer": layer, "kind": short, "norm": weight,
                      "open_earlier": float(f_earlier[i]), "open_later": float(f_later[i]),
                      "open_prose": float(f_prose[i]), "where": contexts[f"{name}:{c}"]})
    by_kind: Dict[str, Dict[str, float]] = {}
    for i in ranked_specific[:256]:
        layer, short = delta.kind(names[i][0])
        by_kind.setdefault(short, {"count": 0})["count"] += 1
    by_layer: Dict[int, int] = {}
    for i in ranked_specific[:256]:
        layer, _ = delta.kind(names[i][0])
        by_layer[layer] = by_layer.get(layer, 0) + 1
    return {"sets": sets.meta, "alive": len(alive), "subcomponents": len(names),
            "open_per_position": {g: sum(stats[g]["open"]) for g in stats},
            "variants": fractions, "reverted": reverts, "most_specific": table,
            "top256_specific_by_kind": by_kind, "top256_specific_by_layer": dict(sorted(by_layer.items())),
            "open_per_position_held_out": {g: sum(held_out[g]["open"]) for g in held_out},
            "ranked_on": "training demonstrations and unscored wikitext",
            "per_item": rows, "gate_open": {g: stats[g]["open"] for g in stats}}


def export(model, maps, tokenizer, opts: Dict[str, Any], device) -> None:
    """Write the decomposition with `export_n` subcomponents reverted as a checkpoint for `make eval`

    Chosen on training text by `export_by`: "used" (most often open on the
    earlier task) or "specific" (open on the earlier task more than on the later
    one), the two rankings `analyse` compares.
    """
    from pathlib import Path

    import torch

    names = delta.component_names(maps)
    with step("ranking on training text"):
        stats = gate_statistics(model, maps, ranking_groups(tokenizer, opts, device))
    f_earlier = torch.tensor(stats["earlier"]["open"])
    score = f_earlier - torch.tensor(stats["later"]["open"]) if opts["export_by"] == "specific" else f_earlier
    chosen = [names[i] for i in score.argsort(descending=True)[: opts["export_n"]].tolist()]
    weights = delta.merged(maps, chosen)
    plain = stage.load(opts["before"], str(device))
    source = Path(load_config(opts["after"]).hf_name)
    target = stage.SDFT_REPO / "runs" / "derived" / (
        f"{source.parent.name}-{source.name}-pd-revert{opts['export_n']}-{opts['export_by']}")
    with step(f"exporting to {target}"), torch.no_grad():
        for name, module in delta.targets(plain).items():
            module.weight.copy_(weights[name].to(module.weight.dtype))
        plain.save_pretrained(target)
        tokenizer.save_pretrained(target)
    (target / "reverted.json").write_text(json.dumps([f"{n}:{c}" for n, c in chosen]))
    log(f"evaluate with: make eval DATASET=tooluse CKPT={target.relative_to(stage.SDFT_REPO)} "
        f"(and DATASET=science) in {stage.SDFT_REPO}")


def main(argv: List[str]) -> int:
    import torch

    opts = options(argv)
    # The update's path runs in float32 (it is thousandths of the weight it is
    # added to); TF32 keeps ten mantissa bits, far below that scale, and is
    # several times faster on this hardware.
    torch.backends.cuda.matmul.allow_tf32 = True
    pair = f"{opts['before']}..{opts['after']}"
    artifact, weights = result("phase3-delta-pd.json"), result("phase3-delta-pd.safetensors")
    set_log_file(result("phase3-delta-pd.log"))
    guard(pair)
    banner("phase 3 decomposing the update", {**opts, "artifact": artifact})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    with track(f"phase3_delta_pd {pair} {opts['stage']}", "mi-lab-diffing", params=opts,
               outputs=[results_root()], tags={"script": "phase3_delta_pd"}) as tracker:
        with step("models and the update"):
            tokenizer, model, maps = build(opts, device)
        if opts["stage"] in ("all", "train"):
            train(model, maps, tokenizer, opts, tracker, device)
            save(maps, weights)
            log(f"decomposition saved to {weights}")
        else:
            restore(maps, weights)
        if opts["stage"] == "export":
            export(model, maps, tokenizer, opts, device)
        if opts["stage"] in ("all", "analyse"):
            report = {"options": opts, **analyse(model, maps, tokenizer, opts, tracker, device)}
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(report, indent=1))
            log("--- final, 1000-draw intervals ---")
            for name, entry in report["variants"].items():
                log(stage.line(name, entry))
        log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MI_LAB_RESULTS", stage.pair_dir(*[w for w in sys.argv[1:] if not w.startswith("--")][:2]))
    sys.exit(main(sys.argv[1:]))
