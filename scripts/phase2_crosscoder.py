"""Phase 2: a BatchTopK crosscoder over two checkpoints' residual stream at one depth, read as a model diff

The dictionary half of the proposal that the frozen transcoders cannot do: a
dictionary trained on *both* checkpoints at once, so a direction only the
fine-tune uses can exist in it at all (`src/methods/crosscoder.py` has the
construction, the two artifacts that made L1 crosscoders unreliable for this,
and the latent-scaling check against them).

On the residual stream, after one block, so it sees what attention changed as
well as the MLPs -- the weight diff found the two moved alike. The depth is a
fraction (invariant 1); the default sits where the weight diff moved most.

Training stream, identical for both checkpoints: half wikitext-103 train, half
the two tasks' training demonstrations as templated conversations, packed into
fixed-length chunks so there is no padding to mask. Position 0 of every chunk
is dropped: its norm is the attention sink's and would dominate the loss.

Evaluation, on text the crosscoder never trained on: wikitext test passages and
the two eval sets (prompt plus the base checkpoint's answer, answer positions).
Per set and model, the fraction of variance unexplained; then the latents by
relative decoder norm, the latent-scaling ratios of every exclusive latent, how
many survive, where they fire, and the contexts they fire hardest in.

A shared latent can still be *rotated* -- the same feature read along a bent
direction in the second model -- and that, not exclusivity, is where the first
pair of runs found the methods differ. So the report also keeps every live
latent's decoder cosine, relative norm and firing frequency per text set, and
reads back the contexts of the most rotated active ones. `--evaluate-only`
reloads a trained crosscoder and redoes only this half.

Run: uv run python -m scripts.phase2_crosscoder qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science
     uv run python -m scripts.phase2_crosscoder qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science --evaluate-only
"""

import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from src.core.config import load_config
from src.domains.lm.backend.layout import _blocks
from src.domains.lm.data.sdft import TASKS, task_texts, training_texts
from src.methods import crosscoder as cc
from src.telemetry.observe import banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-crosscoder.json")
WEIGHTS = result("phase2-crosscoder.safetensors")
LOG = result("phase2-crosscoder.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
TEXT_SOURCE = "Qwen/Qwen3-0.6B"
BUDGET = 2048
CHUNK = 1024
CHUNKS_PER_PASS = 16
MAX_POSITIONS = 3072
# An exclusive latent survives latent scaling if the other model explains at most this share along it.
SURVIVES = 0.3
# A rotated latent is read back only if it fires on at least this share of some text set's rows.
ACTIVE = 1e-3
ROTATED = 15


class _StopError(Exception):
    """Raised by the capture hook once the site has been read, so the rest of the stack never runs"""


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def stream_at(model, ids, layer: int):
    """The residual stream after block `layer` for a batch [rows, positions], stopping the pass there"""
    import torch

    seen = {}

    def keep(_module, _inputs, out):
        seen["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise _StopError

    handle = _blocks(model)[layer].register_forward_hook(keep)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    except _StopError:
        pass
    finally:
        handle.remove()
    return seen["h"]


def chunks(tokenizer, seed: int) -> Iterator[List[int]]:
    """Fixed-length chunks, alternating wikitext train and the demonstrations, each reshuffled per pass"""
    from datasets import load_dataset

    prose = [row for row in load_dataset("wikitext", "wikitext-103-raw-v1", split="train")["text"] if len(row) > 200]
    chat = training_texts(SDFT_REPO, "tooluse", tokenizer) + training_texts(SDFT_REPO, "science", tokenizer)
    generator = random.Random(seed)
    eos = tokenizer.eos_token_id

    def packed(texts: List[str]) -> Iterator[List[int]]:
        buffer: List[int] = []
        while True:
            order = list(range(len(texts)))
            generator.shuffle(order)
            for index in order:
                buffer.extend(tokenizer(texts[index], add_special_tokens=False)["input_ids"] + [eos])
                while len(buffer) >= CHUNK:
                    yield buffer[:CHUNK]
                    buffer = buffer[CHUNK:]

    sources = [packed(prose), packed(chat)]
    turn = 0
    while True:
        yield next(sources[turn])
        turn = 1 - turn


def fill(pre, post, source: Iterator[List[int]], layer: int, rows: int, device):
    """`rows` paired rows [rows, 2, width] from the stream, position 0 of every chunk dropped"""
    import torch

    parts, have = [], 0
    while have < rows:
        ids = torch.tensor([next(source) for _ in range(CHUNKS_PER_PASS)], device=device)
        h_pre, h_post = stream_at(pre, ids, layer)[:, 1:], stream_at(post, ids, layer)[:, 1:]
        pair = torch.stack([h_pre, h_post], dim=2).flatten(0, 1).to(torch.bfloat16)   # [rows, 2, width]
        parts.append(pair)
        have += pair.shape[0]
    buffer = torch.cat(parts)[:rows]
    return buffer[torch.randperm(buffer.shape[0], device=device)]


def evaluation_sets(pre, post, tokenizer, layer: int, device) -> Dict[str, Dict[str, Any]]:
    """Held-out rows per text set, with where each row came from so a latent's top rows can be read back"""
    import torch

    from src.domains.lm.data.wikitext import passages

    sequences: Dict[str, List[Tuple[Any, int]]] = {"wikitext": []}
    for text in passages(64):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=CHUNK)["input_ids"]
        sequences["wikitext"].append((ids, 1))
    for task in TASKS:
        sequences[task] = []
        for item in task_texts(SDFT_REPO, TEXT_SOURCE, task, tokenizer, BUDGET):
            prompt = tokenizer(item.prompt, add_special_tokens=False)["input_ids"]
            answer = tokenizer(item.answer, add_special_tokens=False)["input_ids"]
            ids = (prompt + answer)[:MAX_POSITIONS]
            if len(ids) - len(prompt) >= 2:
                sequences[task].append((torch.tensor([ids]), len(prompt)))
    sets = {}
    for name, items in sequences.items():
        rows, origin = [], []
        for index, (ids, start) in enumerate(items):
            ids = ids.to(device)
            h_pre, h_post = stream_at(pre, ids, layer)[0, start:], stream_at(post, ids, layer)[0, start:]
            rows.append(torch.stack([h_pre, h_post], dim=1).to(torch.bfloat16))
            origin.extend((index, position) for position in range(start, ids.shape[1]))
        sets[name] = {"rows": torch.cat(rows), "origin": origin, "ids": [ids for ids, _ in items]}
    return sets


def batches(rows, size: int):
    for begin in range(0, rows.shape[0], size):
        yield rows[begin : begin + size]


def context(tokenizer, ids, position: int, width: int = 12) -> str:
    sequence = ids[0].tolist()
    before = tokenizer.decode(sequence[max(0, position - width) : position])
    here = tokenizer.decode(sequence[position : position + 1])
    after = tokenizer.decode(sequence[position + 1 : position + 4])
    return f"{before}[[{here}]]{after}"


def evaluate(coder, sets, tokenizer, batch: int) -> Dict[str, Any]:
    import torch

    report: Dict[str, Any] = {"sets": {}}
    activity = {}
    for name, entry in sets.items():
        residual = torch.zeros(2, device=coder.W_dec.device)
        centred = torch.zeros(2, device=coder.W_dec.device)
        fired = torch.zeros(coder.latents, device=coder.W_dec.device)
        mean = coder.normalise(entry["rows"]).mean(0, keepdim=True)
        with torch.no_grad():
            for rows in batches(entry["rows"], batch):
                x = coder.normalise(rows)
                reconstruction, f = coder(x, batch_topk=False)
                residual += ((x - reconstruction) ** 2).sum((0, 2))
                centred += ((x - mean) ** 2).sum((0, 2))
                fired += (f > 0).float().sum(0)
        activity[name] = fired / entry["rows"].shape[0]
        report["sets"][name] = {"rows": entry["rows"].shape[0], "fvu_pre": float(residual[0] / centred[0]),
                                "fvu_post": float(residual[1] / centred[1]),
                                "l0": float(activity[name].sum())}

    live = sum(activity.values()) > 0
    norms = cc.relative_norms(coder)
    cosines = cc.decoder_cosines(coder)
    classes = cc.classify(norms, live)
    report["live"] = int(live.sum())
    report["relative_norm_histogram"] = torch.histc(norms[live].float(), bins=20, min=0, max=1).tolist()
    report["shared_cosine_quartiles"] = quartiles(cosines[classes["shared"]].tolist()) if classes["shared"] else []
    report["classes"] = {name: len(index) for name, index in classes.items()}

    live_index = live.nonzero().flatten().tolist()
    report["latents"] = {
        "index": live_index,
        "relative_norm": norms[live].tolist(),
        "cosine": cosines[live].tolist(),
        "frequency": {name: activity[name][live].tolist() for name in sets},
    }
    busy = torch.stack([activity[name] for name in sets]).amax(0) >= ACTIVE
    candidates = [i for i in classes["shared"] if busy[i]]
    candidates.sort(key=lambda i: float(cosines[i]))
    report["rotated"] = []
    for latent in candidates[:ROTATED]:
        report["rotated"].append({
            "latent": latent, "cosine": float(cosines[latent]), "relative_norm": float(norms[latent]),
            "frequency": {name: float(activity[name][latent]) for name in sets},
            "contexts": top_contexts(coder, sets, tokenizer, latent, batch),
        })
    # Where the rotation is: mean frequency per text set of the most rotated tenth of busy shared latents.
    if candidates:
        tenth = candidates[: max(1, len(candidates) // 10)]
        report["rotated_tenth_frequency"] = {name: float(activity[name][tenth].mean()) for name in sets}
        report["all_busy_frequency"] = {name: float(activity[name][candidates].mean()) for name in sets}

    everything = torch.cat([entry["rows"] for entry in sets.values()])
    report["exclusive"] = {}
    for name, side in (("pre_only", cc.PRE), ("post_only", cc.POST)):
        scaled = cc.latent_scaling(coder, (coder.normalise(rows) for rows in batches(everything, batch)),
                                   classes[name], side)
        survivors = [row for row in scaled
                     if row["nu_error"] == row["nu_error"] and abs(row["nu_error"]) <= SURVIVES
                     and abs(row["nu_activation"]) <= SURVIVES]
        for row in survivors:
            row["frequency"] = {set_name: float(activity[set_name][row["latent"]]) for set_name in sets}
        survivors.sort(key=lambda row: -row["activity"])
        for row in survivors[:10]:
            row["contexts"] = top_contexts(coder, sets, tokenizer, row["latent"], batch)
        report["exclusive"][name] = {
            "classed": len(scaled),
            "survive": len(survivors),
            "nu_error_quartiles": quartiles([row["nu_error"] for row in scaled]),
            "survivors": survivors,
        }
    return report


def quartiles(values: List[float]) -> List[float]:
    values = sorted(v for v in values if v == v)
    if not values:
        return []
    return [values[int(q * (len(values) - 1))] for q in (0.25, 0.5, 0.75)]


def top_contexts(coder, sets, tokenizer, latent: int, batch: int, count: int = 6) -> List[Dict[str, Any]]:
    import torch

    best: List[Tuple[float, str, int]] = []
    for name, entry in sets.items():
        values = []
        with torch.no_grad():
            for rows in batches(entry["rows"], batch):
                values.append(coder.encode(coder.normalise(rows), batch_topk=False)[:, latent])
        values = torch.cat(values)
        top = values.topk(min(count, values.numel()))
        best.extend((float(v), name, int(i)) for v, i in zip(top.values, top.indices, strict=True) if v > 0)
    best.sort(reverse=True)
    out = []
    for value, name, row in best[:count]:
        sequence, position = sets[name]["origin"][row]
        out.append({"set": name, "activation": value,
                    "context": context(tokenizer, sets[name]["ids"][sequence], position)})
    return out


def parse(argv: List[str]) -> Dict[str, Any]:
    positional, flags, index = [], {}, 0
    while index < len(argv):
        word = argv[index]
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else None
            if following is not None and not following.startswith("--"):
                flags[word[2:]] = following
                index += 1
            else:
                flags[word[2:]] = "1"
        else:
            positional.append(word)
        index += 1
    if len(positional) != 2:
        raise SystemExit("give two configs, pre and post")
    return {
        "pre": positional[0], "post": positional[1],
        "depth": float(flags.get("depth", 0.55)),
        "positions": int(float(flags.get("positions", 2e7))),
        "latents": int(flags.get("latents", 16384)),
        "k": int(flags.get("k", 64)),
        "batch": int(flags.get("batch", 4096)),
        "buffer": int(flags.get("buffer", 2 ** 19)),
        "lr": float(flags.get("lr", 2e-4)),
        "seed": int(flags.get("seed", 0)),
        "evaluate_only": "evaluate-only" in flags,
    }


def main(argv: List[str]) -> int:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    options = parse(argv)
    set_log_file(LOG)
    guard(f"{options['pre']}..{options['post']}")
    torch.manual_seed(options["seed"])
    tokenizer = AutoTokenizer.from_pretrained(load_config(options["pre"]).hf_name)
    with step("loading both checkpoints"):
        pre, post = load(options["pre"]), load(options["post"])
    device = next(pre.parameters()).device
    layers = pre.config.num_hidden_layers
    layer = dataclasses.replace(load_config(options["pre"]), n_layers=layers).layer(options["depth"])
    width = pre.config.hidden_size
    banner("phase 2 crosscoder", {**options, "layer": layer, "width": width, "artifact": ARTIFACT})
    started = time.time()

    coder = cc.Crosscoder(width, options["latents"], options["k"], seed=options["seed"]).to(device)
    if options["evaluate_only"]:
        from safetensors.torch import load_file

        coder.load_state_dict(load_file(str(WEIGHTS), device=str(device)))
        previous = json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else {}
        with step("held-out evaluation of the saved crosscoder"):
            sets = evaluation_sets(pre, post, tokenizer, layer, device)
            report = evaluate(coder, sets, tokenizer, options["batch"])
        report.update({key: previous[key] for key in ("history", "minutes") if key in previous})
        report.update({"options": options, "layer": layer, "width": width, "survive_threshold": SURVIVES})
        ARTIFACT.write_text(json.dumps(report, indent=1))
        log(f"re-evaluated: live {report['live']}, classes {report['classes']}, "
            f"rotated read back {len(report['rotated'])}")
        return 0
    optimiser = torch.optim.Adam(coder.parameters(), lr=options["lr"], betas=(0.9, 0.999))
    total = options["positions"] // options["batch"]
    warmup, decay_from = min(200, total // 10), int(0.8 * total)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup) if s < decay_from
                                                 else max(0.0, (total - s) / max(1, total - decay_from)))
    source = chunks(tokenizer, options["seed"])
    done, history, training = 0, [], time.time()
    while done < total:
        with step(f"filling {options['buffer']} rows at layer {layer}") as facts:
            buffer = fill(pre, post, source, layer, options["buffer"], device)
            facts["gpu"] = gpu()
        if done == 0:
            coder.fit_scales(buffer[: 2 ** 16])
        for rows in batches(buffer, options["batch"]):
            if done >= total or rows.shape[0] < options["batch"]:
                break
            metrics = cc.train_step(coder, optimiser, coder.normalise(rows))
            schedule.step()
            done += 1
            if done % 200 == 0 or done == total:
                metrics["step"] = done
                history.append(metrics)
                log(f"step {done}/{total}  loss {metrics['loss']:.3f}  fvu {metrics['fvu_pre']:.3f}/"
                    f"{metrics['fvu_post']:.3f}  l0 {metrics['l0']:.0f}  dead {metrics['dead']}  "
                    f"eta {duration((time.time() - training) / done * (total - done))}", indent=1)
        del buffer
    save_file({name: value.detach().cpu().contiguous() for name, value in coder.state_dict().items()}, str(WEIGHTS))

    with step("held-out evaluation"):
        sets = evaluation_sets(pre, post, tokenizer, layer, device)
        report = evaluate(coder, sets, tokenizer, options["batch"])
    report.update({"options": options, "layer": layer, "width": width, "history": history,
                   "survive_threshold": SURVIVES, "minutes": (time.time() - started) / 60})
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=1))
    for name, entry in report["sets"].items():
        log(f"{name:9s} fvu pre {entry['fvu_pre']:.3f} post {entry['fvu_post']:.3f}  l0 {entry['l0']:.1f}")
    survival = [f"{name} survive {entry['survive']}/{entry['classed']}" for name, entry in report["exclusive"].items()]
    log(f"live {report['live']}, classes {report['classes']}, " + ", ".join(survival))
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
