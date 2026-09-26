"""
Putting a checkpoint's weights on the device without a second copy of them anywhere.

`from_pretrained(device_map=...)` looks like it streams and does not quite: on
transformers 5 it stages tensors on the CPU on their way to the device, and
the host copy of a Qwen3.6-27B peaked around 1.3x the weights before it was
freed -- measured on the 1.7B, where the loader's anonymous memory reached
4.3 GB for 3.2 GB of weights. On a discrete GPU that is host RAM spent and
returned. On the GB10 host and device are the same 121 GB, so the model and
its staging copy are two draws on one pool, and a load that ends at 50 GB
passes through ~100: one such load ran the machine out on the OOM killer, and
the next came within 7 GB of doing it again.

So `stream` does the load itself. The model is built from its config
directly on the device, with its ordinary initialization -- skipping init
would be faster and would leave buffers such as rotary frequencies, which
transformers 5 fills during init, as whatever the allocator handed back.
Then each safetensors file that holds a tensor the model wants is opened on
the device, copied in one tensor at a time, closed, and dropped from the page
cache. The peak is the model plus one tensor.

It refuses rather than guesses. Every parameter and persistent buffer must be
loaded exactly once from a tensor of its own shape; a model key the
checkpoint lacks, a shape that disagrees, or a tensor loaded twice is an
error naming it. A loader that silently kept a randomly initialized tensor
would produce a model that runs, generates text, and is wrong. Checkpoint
tensors the model has no use for -- a vision tower, an MTP head, GPT-2's old
causal-mask buffers -- are counted and not refused: they cannot corrupt the
model, and any rename that went wrong shows up as a *missing* model tensor.

Checkpoint keys are matched to the model's by name, with two renames (see
`model_key`): a multimodal checkpoint's nested language model, and a
base-model checkpoint saved without its head class's prefix. A checkpoint
needing more is refused by the rule above.

A common pipe could be: config | build on device | stream | load_adapter
"""

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

NESTED = "model.language_model."


class LoadError(ValueError):
    """A checkpoint that does not fill the model exactly, said with what is off"""


def checkpoint_files(name: str) -> Optional[Dict[str, List[str]]]:
    """Each safetensors file of a checkpoint -- a hub name or a local directory -- and the keys in it

    None when the checkpoint is not safetensors, which sends the caller back
    to `from_pretrained`.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    def fetch(filename: str) -> Optional[str]:
        if Path(name).is_dir():
            path = Path(name) / filename
            return str(path) if path.exists() else None
        try:
            return hf_hub_download(name, filename)
        except EntryNotFoundError:
            return None

    index = fetch("model.safetensors.index.json")
    if index is None:
        single = fetch("model.safetensors")
        if single is None:
            return None
        from safetensors import safe_open

        with safe_open(single, framework="pt") as handle:
            return {single: list(handle.keys())}
    files: Dict[str, List[str]] = {}
    for key, filename in json.loads(Path(index).read_text())["weight_map"].items():
        files.setdefault(filename, []).append(key)
    return {fetch(filename): keys for filename, keys in files.items()}


def model_key(key: str, wanted: Iterable[str], prefix: str = "") -> Optional[str]:
    """The model's name for a checkpoint key, or None if the model has no such tensor

    Two renames and no more: a multimodal checkpoint's nested language model,
    and a base-model checkpoint saved without the head class's prefix
    (GPT-2's `h.0...` for `transformer.h.0...`), which `from_pretrained`
    also adds.
    """
    candidates = [key]
    if key.startswith(NESTED):
        candidates.append(f"model.{key[len(NESTED):]}")
    if prefix:
        candidates.append(f"{prefix}.{key}")
    return next((candidate for candidate in candidates if candidate in wanted), None)


def ignored(model, key: str) -> bool:
    """Whether the model's own class declares this checkpoint key unused (a vision tower, an MTP head)"""
    patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    return any(re.search(pattern, key) for pattern in patterns)


def drop_cache(path: str) -> None:
    """Tell the kernel this file's pages are done with, so the cache does not hold a second model"""
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    except (OSError, AttributeError):
        pass


def tied(model) -> List[str]:
    """Model keys that are the same tensor as another, and so are never loaded on their own

    Transformers 4 lists them and transformers 5 maps each to its source; the
    keys are what both agree on.
    """
    if not getattr(model.config.get_text_config(), "tie_word_embeddings", False):
        return []
    return list(getattr(model, "_tied_weights_keys", None) or [])


def fill(model, files: Dict[str, List[str]]) -> Dict[str, int]:
    """Copy every tensor the model wants out of `files`, one file at a time; returns counts for the log"""
    targets = dict(model.state_dict(keep_vars=True))
    skip = set(tied(model))
    wanted = set(targets) - skip
    loaded: Dict[str, str] = {}
    unused: List[str] = []
    from safetensors import safe_open

    prefix = getattr(model, "base_model_prefix", "")
    device = next(model.parameters()).device
    for path, keys in files.items():
        mapped = {key: model_key(key, wanted, prefix) for key in keys}
        if not any(mapped.values()):
            unused.extend(keys)
            continue
        # Opened *on the device*: read on the CPU, every tensor passes through host
        # memory the allocator keeps, and filling the 1.7B that way peaked at a whole
        # second copy (+3.2 GB for 3.2 GB of weights, malloc_trim or not). Read to the
        # device, the host side stays flat (+0.0 GB).
        with safe_open(path, framework="pt", device=str(device)) as handle, torch.no_grad():
            for key, name in mapped.items():
                if name is None:
                    unused.append(key)
                    continue
                if name in loaded:
                    raise LoadError(f"'{name}' is in the checkpoint twice: '{loaded[name]}' and '{key}'")
                tensor = handle.get_tensor(key)
                target = targets[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise LoadError(f"'{key}' is {tuple(tensor.shape)} and '{name}' is {tuple(target.shape)}")
                target.copy_(tensor.to(target.dtype))
                loaded[name] = key
                del tensor
        drop_cache(path)
    missing = sorted(wanted - set(loaded))
    if missing:
        raise LoadError(f"{len(missing)} model tensors are not in the checkpoint, e.g. {missing[:5]}; "
                        "they would stay randomly initialized")
    if skip:
        model.tie_weights()
    return {"loaded": len(loaded), "files": len(files), "tied": len(skip), "unused": len(unused),
            "unused_undeclared": sum(not ignored(model, key) for key in unused)}


def stream(name: str, dtype: torch.dtype, device: str):
    """The checkpoint `name` built on `device` and filled file by file, or None if it is not safetensors"""
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    files = checkpoint_files(name)
    if files is None:
        return None
    config = AutoConfig.from_pretrained(name)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    fill(model, files)
    # from_pretrained reads this and from_config does not; without it the stop tokens differ
    # and so does every generation
    with contextlib.suppress(OSError):
        model.generation_config = GenerationConfig.from_pretrained(name)
    return model.eval()
