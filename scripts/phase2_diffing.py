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
     uv run python -m scripts.phase2_diffing qwen3-1.7b-base qwen3-1.7b summarise
"""

import sys
import time
from typing import Any, Dict, List

from src.core.config import load_config
from src.methods import diffing
from src.telemetry.observe import Progress, banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, load_state, result, save_state

ARTIFACTS = {
    "weights": "phase2-weight-diff.json",
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


STAGES = {"check": stage_check, "weights": stage_weights, "summarise": stage_summarise}


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
