import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

"""
The Self-Distillation runs, as files on the Hugging Face Hub.

The SDFT-vs-SFT comparison is trained in the Self-Distillation repository,
not here: `make train-seq` writes `runs/seq-<model>/<i>-<dataset>/` and
`make sft-seq` writes `runs/sft-seq-<model>/...`, one directory per stage,
each starting from the stage before it. This module turns that tree into one
weights file per stage under a name that says what the model was trained on:

    self-distill-vs-sft/qwen3-0.6b/sdft/tool-use.safetensors
    self-distill-vs-sft/qwen3-0.6b/sdft/tool-use-then-science.safetensors
    self-distill-vs-sft/qwen3-0.6b/sft/...

A stage is named by the whole sequence that produced it, not by its last
dataset, because `science` alone would be read as "trained on science" and
the forgetting result is precisely that it was trained on tool use first.

Only the final model a run saved is published -- the `model.safetensors` that
`trainer.save_model` writes at the top of the stage directory. A stage without
one did not finish, and falling back to its newest `checkpoint-N/` would ship
a half-trained model under the finished model's name, so it is refused.
The whole plan is refused if any stage is missing, rather than pushing the
half that exists: a repository holding SDFT's final stage and not SFT's is a
comparison with one side.

The crosscoders trained on pairs of those checkpoints (saved by the crosscoder
training script, one per results directory) go to a folder of
their own, named by the same stage names so a crosscoder and the two models it
was trained on can be matched by eye:

    crosscoders/qwen3-0.6b/sdft/base--tool-use-then-science/crosscoder.safetensors
    crosscoders/qwen3-0.6b/sdft/tool-use--tool-use-then-science/crosscoder.json

The JSON beside the weights is the run's own report, and it is not optional:
it carries the layer, the width, the latent count and `k`, without which the
weights are a set of tensors nobody can load into a `Crosscoder`.

A common pipe could be: plan | crosscoder_plan | push
"""

PREFIX = "self-distill-vs-sft"
CROSSCODER_PREFIX = "crosscoders"
#: The file stem the crosscoder training script saves under (`scripts/phase2_crosscoder.py`). Only
#: used to find the files; on the Hub they are renamed `crosscoder.safetensors` / `crosscoder.json`.
SAVED_CROSSCODER = "phase2-crosscoder"
#: How a checkpoint that is not a Self-Distillation stage is named: the model it all starts from.
BASE = "base"
WEIGHTS = "model.safetensors"
INDEX = "model.safetensors.index.json"

#: The run root names `make train-seq` and `make sft-seq` write, by method.
METHODS = {"seq": "sdft", "sft-seq": "sft"}

#: Dataset names as the Makefile spells them, against how they read in a path.
DATASETS = {"tooluse": "tool-use"}

STAGE = re.compile(r"^(\d+)-(.+)$")


class HubError(ValueError):
    """A run tree that cannot be published as it stands"""


@dataclass(frozen=True)
class Upload:
    """One stage's weights and where they land in the repository"""

    method: str
    model: str
    stage: str
    sources: tuple
    path_in_repo: str

    @property
    def size(self) -> int:
        return sum(Path(source).stat().st_size for source in self.sources)


def stage_name(datasets: Sequence[str]) -> str:
    """`tooluse`, `science` -> `tool-use-then-science`: the sequence, not the last step"""
    return "-then-".join(DATASETS.get(name, name) for name in datasets)


def _weights(stage: Path) -> List[Path]:
    """The final model's weight files in a stage directory, one or its shards"""
    single = stage / WEIGHTS
    if single.is_file():
        return [single]
    index = stage / INDEX
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        missing = [shard for shard in shards if not (stage / shard).is_file()]
        if missing:
            raise HubError(f"{index} names shards that are not there: {missing}")
        return [stage / shard for shard in shards]
    checkpoints = sorted(stage.glob("checkpoint-*"))
    hint = f"; it has {checkpoints[-1].name}, but a checkpoint is not the finished model" if checkpoints else ""
    raise HubError(f"{stage} has no {WEIGHTS}: the stage did not finish{hint}")


def _stages(root: Path) -> List[Tuple[int, str, Path]]:
    """A run's `<i>-<dataset>` directories in order, refusing a gap in the numbering"""
    stages = sorted(
        (int(match.group(1)), match.group(2), child)
        for child in root.iterdir()
        if child.is_dir() and (match := STAGE.match(child.name))
    )
    if not stages:
        raise HubError(f"{root} holds no <i>-<dataset> stage directories")
    indices = [index for index, _, _ in stages]
    if indices != list(range(1, len(stages) + 1)):
        raise HubError(f"{root} stages are numbered {indices}; a gap means a stage is missing")
    return stages


def checkpoint_name(hf_name: str) -> Tuple[str, Optional[str], str]:
    """(model, method, stage) for what a config's `hf_name` points at

    A Self-Distillation stage directory is named the way `plan` names it, which
    takes its earlier siblings -- `2-science` is `tool-use-then-science` only
    because `1-tooluse` came first. Anything else is a Hub id, and the one these
    runs start from, so it is `base` with no method.
    """
    path = Path(hf_name).expanduser()
    if not path.is_dir():
        return Path(hf_name).name.lower(), None, BASE
    match, root = STAGE.match(path.name), path.parent
    key = next((key for key in METHODS if root.name.startswith(f"{key}-")), None)
    if match is None or key is None:
        raise HubError(f"{path} is a directory but not a <seq|sft-seq>-<model>/<i>-<dataset> stage")
    index = int(match.group(1))
    datasets = [dataset for i, dataset, _ in _stages(root) if i <= index]
    return root.name[len(key) + 1 :].lower(), METHODS[key], stage_name(datasets)


def plan(runs: Path, prefix: str = PREFIX) -> List[Upload]:
    """Every finished stage under a Self-Distillation `runs/` directory, named for the Hub

    Looks for `seq-<model>` and `sft-seq-<model>` and refuses a tree where
    either method is absent or any stage lacks its final weights.
    """
    runs = Path(runs)
    if not runs.is_dir():
        raise HubError(f"{runs} is not a directory; point this at Self-Distillation's runs/")
    uploads: List[Upload] = []
    found: Dict[str, List[str]] = {}
    for root in sorted(runs.iterdir()):
        key = next((key for key in METHODS if root.name.startswith(f"{key}-")), None)
        if key is None or not root.is_dir():
            continue
        method, model = METHODS[key], root.name[len(key) + 1 :].lower()
        stages = _stages(root)
        datasets: List[str] = []
        for _, dataset, directory in stages:
            datasets.append(dataset)
            name = stage_name(datasets)
            uploads.append(
                Upload(
                    method=method,
                    model=model,
                    stage=name,
                    sources=tuple(str(p) for p in _weights(directory)),
                    path_in_repo=f"{prefix}/{model}/{method}/{name}.safetensors",
                )
            )
        found.setdefault(model, []).append(method)
    if not uploads:
        raise HubError(f"{runs} holds no seq-<model> or sft-seq-<model> run")
    for model, methods in found.items():
        absent = sorted(set(METHODS.values()) - set(methods))
        if absent:
            raise HubError(f"{model} has no {', '.join(absent)} run under {runs}; that is half a comparison")
    return uploads


def _hf_name(config: str) -> str:
    from ..core.config import ConfigError, load_config

    try:
        return load_config(config).hf_name
    except ConfigError as error:
        hint = "run from a checkout whose configs/ has it"
        raise HubError(f"cannot resolve config '{config}': {error}; {hint}") from error


def crosscoder_plan(
    results: Path,
    prefix: str = CROSSCODER_PREFIX,
    resolve: Callable[[str], str] = _hf_name,
) -> List[Upload]:
    """Every trained crosscoder under a results root, with its report, named by the pair it was trained on

    `resolve` turns a config name into its `hf_name`; it is a parameter so the
    naming can be tested without the configs that point at real checkpoints.
    """
    results = Path(results)
    if not results.is_dir():
        raise HubError(f"{results} is not a directory; point this at the results root the crosscoders were written to")
    uploads: List[Upload] = []
    for weights in sorted(results.rglob(f"{SAVED_CROSSCODER}.safetensors")):
        card = weights.with_suffix(".json")
        if not card.is_file():
            raise HubError(f"{weights} has no {card.name} beside it, and without it the layer, width and k are unknown")
        options = json.loads(card.read_text())["options"]
        (pre_model, pre_method, pre_stage), (post_model, post_method, post_stage) = (
            checkpoint_name(resolve(options[side])) for side in ("pre", "post")
        )
        if post_method is None:
            raise HubError(f"{weights}: post checkpoint '{options['post']}' is not a Self-Distillation stage")
        if pre_method not in (None, post_method) or pre_model != post_model:
            raise HubError(
                f"{weights} pairs {pre_model}/{pre_method} with {post_model}/{post_method}; "
                "a crosscoder across two methods or two models has no folder in this layout"
            )
        pair = f"{pre_stage}--{post_stage}"
        folder = f"{prefix}/{post_model}/{post_method}/{pair}"
        for source, name in ((weights, "crosscoder.safetensors"), (card, "crosscoder.json")):
            uploads.append(Upload(post_method, post_model, pair, (str(source),), f"{folder}/{name}"))
    if not uploads:
        raise HubError(f"{results} holds no {SAVED_CROSSCODER}.safetensors")
    destinations = [upload.path_in_repo for upload in uploads]
    clashes = sorted({path for path in destinations if destinations.count(path) > 1})
    if clashes:
        sources = [u.sources[0] for u in uploads if u.path_in_repo in clashes]
        raise HubError(f"two crosscoders for one pair would overwrite each other at {clashes}: {sources}")
    return uploads


def single_file(upload: Upload, directory: Path) -> Path:
    """The upload's weights as one file, merging shards into `directory` if there are several"""
    if len(upload.sources) == 1:
        return Path(upload.sources[0])
    from safetensors.torch import load_file, save_file

    tensors = {}
    for source in upload.sources:
        tensors.update(load_file(source))
    merged = Path(directory) / f"{upload.method}-{upload.stage}.safetensors"
    save_file(tensors, str(merged), metadata={"format": "pt"})
    return merged


def push(uploads: Sequence[Upload], repo: str, message: Optional[str] = None, revision: Optional[str] = None) -> str:
    """Upload every stage in one commit and return its URL

    One commit, so the repository never holds one method's files without
    the other's -- an interrupted push leaves the previous state, not half of
    this one.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    with tempfile.TemporaryDirectory() as scratch:
        operations = [
            CommitOperationAdd(
                path_in_repo=upload.path_in_repo,
                path_or_fileobj=str(single_file(upload, Path(scratch))),
            )
            for upload in uploads
        ]
        info = HfApi().create_commit(
            repo_id=repo,
            operations=operations,
            commit_message=message or f"Add {len(uploads)} files",
            revision=revision,
        )
    return info.commit_url
