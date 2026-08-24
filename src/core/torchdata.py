from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from .config import Position
from .dataset import DatasetError, LabeledPrompts
from .prompts import load_labeled, scan

"""
The torch side of a dataset: map-style and streaming Datasets over prompt
sets, and the same thing over the activations a capture produced.

One rule shapes all of it: a prompt dataset yields text, never token ids.
Tokenization belongs to the backend that owns the model -- an adapter knows
its tokenizer, its padding side and its pad token, and a DataLoader that
tokenizes on its own is a second opinion about all three. The classic version
of that bug is a set tokenized once, cached, then captured through a model
whose vocabulary moved; nothing errors, and the activations are of a
different sentence than the one in the file.

So collate_prompts hands the adapter a list of strings and a tensor of labels,
which is exactly what adapter.capture already takes. What the DataLoader buys
is everything around it: batches that never hold the whole file, workers that
read and parse the next batch while the GPU is busy on this one, and per-batch
padding instead of padding every prompt in the corpus to the longest one in
it.

ActivationDataset is the other half. A capture is the expensive step and its
output is small, so it is worth keeping: [n, layers, d_model] floats plus the
model, layers and position they came from. Those three facts travel with the
tensor because a directory of .pt files whose provenance lives in their
filenames is a directory of activations you will eventually mix up.

A common pipe could be: PromptDataset | prompt_loader | capture_dataset | train_probe
"""

Source = Union[LabeledPrompts, "PromptDataset", IterableDataset, DataLoader]

class PromptDataset(Dataset):
    """A LabeledPrompts as a map-style torch Dataset of (text, label)

    Thin on purpose: the dataset object keeps the name, the label names and
    the groups, and this exposes the rows to a DataLoader without taking
    ownership of any of that.
    """

    def __init__(self, data: LabeledPrompts):
        self.data = data

    @classmethod
    def from_path(cls, path: str, **kwargs) -> "PromptDataset":
        """Load a .prompts or .jsonl file and wrap it"""
        return cls(load_labeled(path, **kwargs))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.data.texts[index], self.data.labels[index]

    def __repr__(self) -> str:
        return f"PromptDataset({self.data.name!r}, n={len(self)}, balance={self.data.balance:.2f})"

class StreamingPrompts(IterableDataset):
    """A .prompts file read lazily, one example at a time, sharded across workers

    For a file too big to hold, or one being captured once and never split.
    Two things it cannot do, both because it never sees the whole file:
    shuffle, and keep a group together. Split before streaming, not after --
    load_prompts is the one to use for anything that gets split.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise DatasetError(f"no dataset at {self.path}")
        # no header is read until iteration starts, so the filename is the name
        self.name = self.path.stem

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        worker = get_worker_info()
        stride = 1 if worker is None else worker.num_workers
        offset = 0 if worker is None else worker.id
        with self.path.open() as handle:
            for index, example in enumerate(scan(handle, source=str(self.path))):
                if index % stride == offset:
                    yield example.text, example.label

    def __repr__(self) -> str:
        return f"StreamingPrompts({str(self.path)!r})"

def collate_prompts(batch: Sequence[Tuple[str, int]]) -> Tuple[List[str], torch.Tensor]:
    """Batch (text, label) pairs into (list of text, label tensor)

    The text stays text. This is the whole reason the collate is written out
    rather than left to the default one, which would try to stack strings.
    """
    texts = [text for text, _ in batch]
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return texts, labels

def prompt_loader(
    data: Source,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    """A DataLoader over prompts that yields (texts, labels)

    shuffle defaults to False because the common use is capture, where the
    order of the rows is the order of the activations and shuffling it only
    makes the two harder to line up. Pass a seed with shuffle to keep the
    permutation reproducible.
    """
    if isinstance(data, LabeledPrompts):
        data = PromptDataset(data)
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    if isinstance(data, IterableDataset):
        if shuffle:
            raise DatasetError("a streaming dataset cannot be shuffled; it never holds more than one example")
        return DataLoader(data, batch_size=batch_size, collate_fn=collate_prompts, num_workers=num_workers)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_prompts,
        generator=generator,
    )

@dataclass
class ActivationDataset(Dataset):
    """Captured activations with their labels, and the provenance to use them honestly

    activations is [n, layers, d_model], holding the layers named in `layers`
    in that order. Keeping the layer axis rather than one dataset per layer is
    what makes a depth sweep one forward pass instead of one per depth.
    """
    activations: torch.Tensor
    labels: List[int]
    layers: List[int]
    model_id: str
    position: str = "last"
    name: str = "unnamed"
    groups: Optional[List[int]] = None

    def __post_init__(self):
        if self.activations.dim() != 3:
            raise DatasetError(
                f"expected [n, layers, d_model] activations, got shape {tuple(self.activations.shape)}"
            )
        if self.activations.shape[0] != len(self.labels):
            raise DatasetError(f"{self.activations.shape[0]} activations but {len(self.labels)} labels")
        if self.activations.shape[1] != len(self.layers):
            raise DatasetError(
                f"activations carry {self.activations.shape[1]} layers but {len(self.layers)} are named: {self.layers}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.activations[index], self.labels[index]

    @property
    def d_model(self) -> int:
        return int(self.activations.shape[2])

    @property
    def n_bytes(self) -> int:
        """What holding this capture costs, which is what decides whether to keep it"""
        return self.activations.numel() * self.activations.element_size()

    def at(self, layer: int) -> "ActivationDataset":
        """The same dataset keeping one layer, addressed by its absolute index

        The index is the model's, not a position in this tensor: a capture of
        layers [0, 4, 8] answers at(8), not at(2). Asking for a layer that was
        not captured is an error naming the ones that were, because the
        alternative is silently probing the wrong depth.
        """
        if layer not in self.layers:
            raise DatasetError(f"layer {layer} was not captured; this dataset holds layers {self.layers}")
        position = self.layers.index(layer)
        return ActivationDataset(
            activations=self.activations[:, position : position + 1],
            labels=list(self.labels),
            layers=[layer],
            model_id=self.model_id,
            position=self.position,
            name=self.name,
            groups=None if self.groups is None else list(self.groups),
        )

    def tensors(self) -> Tuple[torch.Tensor, List[int]]:
        """The pair train_probe and evaluate take, straight out"""
        return self.activations, self.labels

    def save(self, path: str) -> None:
        """Write the capture and its provenance to one file"""
        torch.save(
            {
                "activations": self.activations,
                "labels": self.labels,
                "layers": self.layers,
                "model_id": self.model_id,
                "position": self.position,
                "name": self.name,
                "groups": self.groups,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str) -> "ActivationDataset":
        """Read a capture back, refusing anything that is not one"""
        file = Path(path)
        if not file.exists():
            raise DatasetError(f"no capture at {file}")
        payload = torch.load(file, weights_only=False)
        missing = {"activations", "labels", "layers", "model_id"} - set(payload)
        if missing:
            raise DatasetError(f"{file} is missing capture fields {sorted(missing)}")
        return cls(**payload)

def activation_loader(
    data: ActivationDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """A DataLoader over captured activations, for fitting in minibatches

    Shuffled by default, which is the opposite of prompt_loader and for the
    opposite reason: nothing downstream has to line these rows up against a
    file any more, and a minibatch fit on rows still in label order is a fit
    on batches that are all one class.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, generator=generator)

def _underlying(data: Source) -> Optional[LabeledPrompts]:
    """The LabeledPrompts behind whatever was passed, if there is one"""
    if isinstance(data, LabeledPrompts):
        return data
    if isinstance(data, DataLoader):
        return _underlying(data.dataset)
    return getattr(data, "data", None)

def _name(data: Source) -> str:
    """What to call the capture that comes out of this source"""
    source = _underlying(data)
    if source is not None:
        return source.name
    if isinstance(data, DataLoader):
        return _name(data.dataset)
    return getattr(data, "name", "unnamed")

def capture_dataset(
    adapter,
    data: Source,
    layers: Optional[Sequence[int]] = None,
    position: Position = Position.LAST,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> ActivationDataset:
    """Run an adapter over a prompt dataset in batches and keep what comes back

    adapter.capture batches internally already, so this is not about speed on
    a small set. It is about never holding the whole set at once: prompts
    arrive batch_size at a time, each batch is padded to its own longest
    prompt rather than to the longest prompt in the corpus, and only the
    activations accumulate.

    Position.ALL is refused here on purpose. It keeps the sequence axis, whose
    length is whatever the longest prompt in each batch happened to be, so the
    batches would not concatenate into one tensor -- call adapter.capture
    directly for that, on prompts you have chosen.
    """
    position = Position(position)
    if position is Position.ALL:
        raise DatasetError(
            "capture_dataset keeps [n, layers, d_model], and Position.ALL adds a sequence axis "
            "whose length differs per batch; call adapter.capture directly for whole sequences"
        )
    resolved = list(layers) if layers is not None else [adapter.layer()]
    source = _underlying(data)
    ordered = not isinstance(data, DataLoader)
    loader = data if isinstance(data, DataLoader) else prompt_loader(
        data, batch_size=batch_size or adapter.cfg.batch_size, num_workers=num_workers
    )

    chunks: List[torch.Tensor] = []
    labels: List[int] = []
    for texts, batch_labels in loader:
        chunks.append(adapter.capture(texts, layers=resolved, position=position))
        labels.extend(int(label) for label in batch_labels)
    if not chunks:
        raise DatasetError("capture needs at least one prompt, and the loader yielded none")

    return ActivationDataset(
        activations=torch.cat(chunks, dim=0),
        labels=labels,
        layers=resolved,
        model_id=adapter.cfg.id,
        position=position.value,
        name=_name(data),
        # groups only survive when this function fixed the order; a caller's
        # DataLoader may have shuffled, and misaligned groups are worse than none
        groups=source.groups if ordered and source is not None else None,
    )
