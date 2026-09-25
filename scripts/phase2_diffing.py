"""Phase 2: what a fine-tune did to the weights, before any dictionary is involved.

The first strategy of `~/main/m/proposals/model-diffing-sdft-transcoders.md`, and
the one that costs nothing: no forward pass, no transcoder, no training run.
Two checkpoints, one subtraction per tensor, and three numbers per tensor
(`src/methods/diffing.py` says what they are and why those three).

It is here rather than under `phase1` because it is the first thing in this
repository that takes *two* checkpoints. That breaks an assumption `guard`
encodes -- one results directory belongs to one config -- so the stamp for these
artifacts is the pair, `<pre>..<post>`. A directory holding a diff of two models
is bound to those two and refuses a third, which is the same guarantee one
config wide.

What it is for: the proposal's experiment 0 and 4. Whether the change landed in
attention or the feed-forward blocks, at what depth, and over how many
directions -- the last of these being the claim a dictionary-based diff has to
beat, since a change that already lives in a handful of directions does not need
twenty thousand features to describe it.

The pair it was developed against is `qwen3-1.7b-base` -> `qwen3-1.7b`, which is
Qwen's own post-training rather than a fine-tune of ours. That pair is a
scale-setter and not a result: a full post-training is a much larger event than
learning one skill, so read its numbers as the ceiling an SDFT diff should come
in well under.

Run: uv run python -m scripts.phase2_diffing qwen3-1.7b-base qwen3-1.7b check
     uv run python -m scripts.phase2_diffing qwen3-1.7b-base qwen3-1.7b weights
     uv run python -m scripts.phase2_diffing qwen3-1.7b-base qwen3-1.7b dictionary
     uv run python -m scripts.phase2_diffing qwen3-1.7b-base qwen3-1.7b summarise
"""

import sys
import time
from typing import Any, Dict, List

from src.core.config import load_config
from src.methods import diffing
from src.model.replacement import load_replacement
from src.telemetry.observe import Progress, banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, load_state, result, save_state

ARTIFACTS = {
    "weights": "phase2-weight-diff.json",
    "dictionary": "phase2-dictionary-fit.json",
}
LOG = result("phase2-diffing.log")

# Tensors, not seconds: the loop's cost per item is one SVD of a fixed shape, so
# a count is the honest unit and the ETA follows from it.
PROGRESS_EVERY = 20


class DiffingRunError(diffing.DiffingError):
    """A phase 2 run that cannot proceed, said with the argument that would fix it"""


def artifact(key: str, **fields) -> Any:
    return result(ARTIFACTS[key].format(**fields))


def pair(pre: str, post: str) -> str:
    """The stamp these artifacts carry: a directory bound to two configs, not one"""
    return f"{pre}..{post}"


def checkpoints(pre: str, post: str) -> Dict[str, Any]:
    """Both configs and both resolved checkpoint directories

    A config rather than a bare path on purpose: `hf_name` is where a fine-tune
    announces itself (`configs/qwen3-0.6b-sdft.yaml` says to point it at a
    trained output directory), so naming configs keeps the diff described by the
    same files as every other measurement on these checkpoints.
    """
    before, after = load_config(pre), load_config(post)
    if before.hf_name == after.hf_name:
        raise DiffingRunError(
            f"'{pre}' and '{post}' both name {before.hf_name}, so their diff is zero by construction. "
            "Point one config's hf_name at the fine-tuned output directory"
        )
    with step(f"resolving {before.hf_name}") as facts:
        pre_path = diffing.resolve(before.hf_name)
        facts["path"] = str(pre_path)
    with step(f"resolving {after.hf_name}") as facts:
        post_path = diffing.resolve(after.hf_name)
        facts["path"] = str(post_path)
    return {"pre": before, "post": after, "pre_path": pre_path, "post_path": post_path}


def stage_check(options: Dict[str, Any]) -> None:
    """What the two checkpoints hold, and what the diff will cost, without doing it"""
    found = checkpoints(options["pre"], options["post"])
    before = diffing.index(found["pre_path"])
    after = diffing.index(found["post_path"])

    shared = sorted(set(before) & set(after))
    sizes = diffing.shapes(before)
    matrices = sum(1 for name in shared if len(sizes[name]) == 2)

    log(f"{len(before)} tensors in pre, {len(after)} in post, {len(shared)} shared")
    log(f"{matrices} of the shared ones are matrices and will take a singular value decomposition", indent=1)
    for name in sorted(set(before) ^ set(after)):
        log(f"unshared: {name}", indent=1)
    if len(shared) < min(len(before), len(after)):
        log("note: the unshared tensors are reported in the artifact's `skipped`, with a reason each", indent=1)


def stage_weights(options: Dict[str, Any]) -> None:
    """The diff itself: three numbers per tensor, written whole and printed as a table"""
    found = checkpoints(options["pre"], options["post"])
    total = len(set(diffing.index(found["pre_path"])) & set(diffing.index(found["post_path"])))
    progress = Progress(total, "tensors", every=PROGRESS_EVERY)

    with step(f"diffing {total} tensors on {options['device']}") as facts:
        report = diffing.checkpoint_delta(
            found["pre_path"],
            found["post_path"],
            device=options["device"],
            estimate=options["estimate"],
            on_tensor=lambda delta: progress.tick(delta.name),
            labels=(options["pre"], options["post"]),
        )
        progress.finish()
        facts["gpu"] = gpu()
        facts["estimated"] = options["estimate"]

    # The resolved paths go in beside the config ids because a hub path ends in
    # the commit hash of the revision that was actually read, and a diff is a
    # claim about two specific sets of weights rather than about two names.
    save_state(artifact("weights"), {
        **report.as_dict(),
        "pre_name": found["pre"].hf_name,
        "post_name": found["post"].hf_name,
        "pre_path": str(found["pre_path"]),
        "post_path": str(found["post_path"]),
    })
    log(str(report))
    for line in diffing.table(report):
        log(line, indent=1)


def stage_summarise(options: Dict[str, Any]) -> None:
    """The saved diff, read back and tabulated, so a summary needs no accelerator"""
    path = artifact("weights")
    if not path.exists():
        raise DiffingRunError(f"{path} does not exist yet; run the 'weights' stage first")
    report = diffing.DeltaReport.from_dict(load_state(path))
    log(str(report))
    for line in diffing.table(report):
        log(line, indent=1)
    if report.skipped:
        log("skipped:")
        for name, why in sorted(report.skipped.items()):
            log(f"{name}: {why}", indent=1)


def passages(count: int, min_chars: int = 1200) -> List[str]:
    """Long stretches of ordinary prose, for measuring a variance over

    Wikitext rather than this study's translation corpus, and long rather than
    short, for two different reasons. Long because FVU divides by the spread of
    a layer's MLP output across positions, and over a six-token prompt there is
    barely any spread to divide by -- the same dictionary reports an FVU half
    again as large on short prompts as on these. Neutral prose because the two
    checkpoints being compared are a base model and an instruction-tuned one,
    and any text shaped like an instruction is text one of them was trained on
    and the other was not.

    It lives in this script because one stage uses it. A second caller moves it
    to `src/data/`.
    """
    from datasets import load_dataset

    rows = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]
    return [row for row in rows if len(row) > min_chars][:count]


def measure_fit(config: str, release: str, texts: List[str], options: Dict[str, Any]):
    """One checkpoint's fit under one dictionary, loaded and then dropped again"""
    import torch

    cfg = load_config(config)
    model, backend = load_replacement(
        cfg.hf_name,
        release,
        dtype=cfg.dtype,
        backend=options["backend"],
        note=lambda message: log(message, indent=1),
        wrap=step,
    )
    with step(f"measuring dictionary fit on {config}") as facts:
        report = diffing.dictionary_fit(model, texts, label=config, release=release)
        facts["backend"] = backend
        facts["fvu"] = round(report.fvu, 4)
        facts["gpu"] = gpu()
    # Dropped before the second checkpoint loads: two of these plus two copies
    # of the dictionary's encoders do not need to be resident at once.
    del model
    torch.cuda.empty_cache()
    return report


def stage_dictionary(options: Dict[str, Any]) -> None:
    """Does the dictionary fitted for `pre` still describe `post`?

    The gate on every strategy that reads features rather than weights. One
    dictionary -- whichever `pre`'s config names -- applied to both checkpoints
    over identical text, reported per layer.

    There is deliberately no config pairing `post`'s weights with this
    dictionary. A config's `transcoder:` block records what was fitted *for that
    checkpoint*, and inventing one that claims otherwise would file the
    experiment's question as though it were an answer.
    """
    before = load_config(options["pre"])
    if before.transcoder is None:
        raise DiffingRunError(
            f"'{options['pre']}' names no transcoder, and this stage transfers one. "
            "configs/qwen3-1.7b-base.yaml is the config in this repo that has a dictionary"
        )
    release = before.transcoder.release
    texts = passages(options["passages"])
    log(f"{len(texts)} passages, dictionary {release} held fixed across both checkpoints")

    reports = {
        side: measure_fit(options[side], release, texts, options) for side in ("pre", "post")
    }
    shift = diffing.fit_shift(reports["pre"], reports["post"])

    save_state(artifact("dictionary"), {
        "release": release,
        "passages": len(texts),
        "pre_config": options["pre"],
        "post_config": options["post"],
        "pre": reports["pre"].as_dict(),
        "post": reports["post"].as_dict(),
        "shift": shift,
        "recorded_variance_unexplained": before.transcoder.variance_unexplained,
    })

    for side in ("pre", "post"):
        log(str(reports[side]))
    log(
        f"mean fvu {reports['pre'].fvu:.3f} -> {reports['post'].fvu:.3f} "
        f"({reports['post'].fvu - reports['pre'].fvu:+.3f}), "
        f"and {before.transcoder.variance_unexplained} is what the release records for {options['pre']}"
    )
    log(f"{'layer':>6}{'pre':>9}{'post':>9}{'delta':>9}", indent=1)
    for row in sorted(shift, key=lambda item: item["layer"]):
        log(f"{row['layer']:>6}{row['before']:>9.3f}{row['after']:>9.3f}{row['delta']:>+9.3f}", indent=1)


STAGES = {
    "check": stage_check,
    "weights": stage_weights,
    "dictionary": stage_dictionary,
    "summarise": stage_summarise,
}


def parse(argv: List[str]) -> Dict[str, Any]:
    """Two configs, a stage, and the flags every stage reads the same way"""
    positional = [word for word in argv if not word.startswith("--")]
    flags = {}
    for index, word in enumerate(argv):
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else "1"
            flags[word[2:]] = following if not following.startswith("--") else "1"
    return {
        "pre": positional[0] if positional else "qwen3-1.7b-base",
        "post": positional[1] if len(positional) > 1 else "qwen3-1.7b",
        "stage": positional[2] if len(positional) > 2 else "check",
        "device": flags.get("device", "cuda" if _accelerated() else "cpu"),
        "backend": flags.get("backend", "transformerlens"),
        "passages": int(flags.get("passages", 8)),
        "estimate": "estimate" in flags,
    }


def _accelerated() -> bool:
    """Whether there is an accelerator to put the decompositions on"""
    import torch

    return bool(torch.cuda.is_available())


def main(argv: List[str]) -> int:
    options = parse(argv)
    if options["stage"] not in STAGES:
        log(f"unknown stage '{options['stage']}'; stages are {', '.join(STAGES)}")
        return 2
    set_log_file(LOG)
    guard(pair(options["pre"], options["post"]))
    banner(f"phase 2 weight diffing: {options['stage']}", {
        "pre": options["pre"],
        "post": options["post"],
        "device": options["device"],
        "singular values": "estimated" if options["estimate"] else "exact",
        "artifact": artifact("weights"),
    })
    started = time.time()
    STAGES[options["stage"]](options)
    log(f"stage '{options['stage']}' done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
