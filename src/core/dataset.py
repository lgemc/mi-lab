import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

"""
A dataset here is the smallest thing a probe can be trained on: prompts and a
binary label. Nothing more is needed, and anything more would tie the object
to one task.

Two fields exist beyond that pair, and both are there to stop a result being
wrong rather than to make it prettier. label_names says what 0 and 1 mean, so
a report reads 'truthful vs deceptive' instead of asking the reader to
remember which way round the file was written. groups says which rows are the
same item -- a minimal pair, a set of paraphrases -- and split() keeps a group
whole, because a contrast pair straddling the split puts two sentences that
differ by one word on both sides of it and the AUC that comes back is
measuring the word.

The synthetic generator exists so that the whole probe path -- capture, train,
evaluate, sweep -- is runnable and testable without a download. It is a toy,
and results on it are worth exactly as much as a toy is: use it to check that
the machinery works, then point load_jsonl at something real.

A common pipe could be: load_jsonl | split | capture | train_probe
"""

class DatasetError(ValueError):
    """Raised when a dataset is unusable: missing fields, non-binary labels, or a split that leaves a class empty"""

@dataclass(frozen=True)
class LabeledPrompts:
    """Prompts with a binary label, and the name of where they came from"""
    texts: List[str]
    labels: List[int]
    name: str = "unnamed"
    label_names: Tuple[str, str] = ("negative", "positive")
    groups: Optional[List[int]] = None

    def __post_init__(self):
        if len(self.texts) != len(self.labels):
            raise DatasetError(f"{len(self.texts)} texts but {len(self.labels)} labels")
        if not self.texts:
            raise DatasetError("a dataset needs at least one example")
        unknown = set(self.labels) - {0, 1}
        if unknown:
            raise DatasetError(f"labels must be 0 or 1, found {sorted(unknown)}")
        if len(self.label_names) != 2 or len(set(self.label_names)) != 2:
            raise DatasetError(f"label_names must be two distinct names for 0 and 1, got {list(self.label_names)}")
        if self.groups is not None and len(self.groups) != len(self.texts):
            raise DatasetError(f"{len(self.texts)} texts but {len(self.groups)} group ids")

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def positives(self) -> int:
        return sum(self.labels)

    @property
    def units(self) -> List[List[int]]:
        """Row indices bundled into the units a split has to keep whole

        Without groups every row is its own unit and this is the identity,
        which is why an ungrouped dataset splits exactly as it always did.
        """
        if self.groups is None:
            return [[index] for index in range(len(self))]
        units: Dict[int, List[int]] = {}
        for index, group in enumerate(self.groups):
            units.setdefault(group, []).append(index)
        return list(units.values())

    @property
    def duplicates(self) -> List[str]:
        """Texts that appear more than once, which are a leak waiting for a split

        A duplicate across the split turns the reported AUC into a memory
        test, and duplicates are what you get when a template set is sampled
        with replacement or two exports are concatenated.
        """
        seen: Dict[str, int] = {}
        for text in self.texts:
            seen[text] = seen.get(text, 0) + 1
        return [text for text, count in seen.items() if count > 1]

    def rename(self, name: str) -> "LabeledPrompts":
        """A copy under a different name, for when the source file is not the story"""
        return replace(self, name=name)

    @property
    def balance(self) -> float:
        """Fraction of the dataset that is positive; 0.5 is balanced

        Worth printing next to every AUC: a probe on a 95/5 split can look
        excellent while having learned the base rate and nothing else.
        """
        return self.positives / len(self)

    def subset(self, indices: Sequence[int]) -> "LabeledPrompts":
        """Take the given rows, keeping the dataset's name, label names and groups

        Group ids are carried over rather than renumbered, so two subsets of
        one dataset can still be checked for a group that crossed between
        them.
        """
        return LabeledPrompts(
            texts=[self.texts[index] for index in indices],
            labels=[self.labels[index] for index in indices],
            name=self.name,
            label_names=self.label_names,
            groups=None if self.groups is None else [self.groups[index] for index in indices],
        )

    def split(self, test_frac: float = 0.3, seed: int = 0) -> Tuple["LabeledPrompts", "LabeledPrompts"]:
        """Shuffle and split, keeping both classes present on both sides and every group whole

        Stratified rather than plain-random: on a small dataset an unlucky
        shuffle can put every positive on one side, and an AUC computed there
        is undefined rather than merely bad.

        The unit of the split is the group, not the row, and groups are
        stratified by the set of labels they carry -- so single positives,
        single negatives and contrast pairs are each cut at test_frac
        separately. A pair whose two halves differ by one word is the whole
        point of a pair and would be the worst possible thing to leak.
        """
        if not 0.0 < test_frac < 1.0:
            raise DatasetError(f"test_frac must be strictly between 0 and 1, got {test_frac}")
        rng = random.Random(seed)
        strata: Dict[Tuple[int, ...], List[List[int]]] = {}
        for unit in self.units:
            signature = tuple(sorted({self.labels[index] for index in unit}))
            strata.setdefault(signature, []).append(unit)

        train_indices: List[int] = []
        test_indices: List[int] = []
        for signature in sorted(strata):
            units = strata[signature]
            rng.shuffle(units)
            cut = round(test_frac * len(units))
            cut = min(max(cut, 1), len(units) - 1) if len(units) > 1 else 0
            for unit in units[:cut]:
                test_indices.extend(unit)
            for unit in units[cut:]:
                train_indices.extend(unit)
        if not train_indices or not test_indices:
            raise DatasetError(f"'{self.name}' is too small to split into train and test")
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
        return self.subset(train_indices), self.subset(test_indices)

def load_jsonl(
    path: str,
    text_field: str = "text",
    label_field: str = "label",
    limit: Optional[int] = None,
) -> LabeledPrompts:
    """Read one JSON object per line into a dataset

    Field names are arguments because every dataset on disk calls these two
    columns something different, and renaming a file's contents to suit a
    library is how a corpus gets silently forked.
    """
    source = Path(path)
    if not source.exists():
        raise DatasetError(f"no dataset at {source}")

    texts: List[str] = []
    labels: List[int] = []
    for number, line in enumerate(source.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"{source}:{number} is not valid JSON") from error
        if text_field not in row or label_field not in row:
            raise DatasetError(
                f"{source}:{number} has keys {sorted(row)}, expected '{text_field}' and '{label_field}'"
            )
        texts.append(str(row[text_field]))
        labels.append(int(row[label_field]))
        if limit is not None and len(texts) >= limit:
            break
    return LabeledPrompts(texts=texts, labels=labels, name=source.stem)

def save_jsonl(data: LabeledPrompts, path: str, text_field: str = "text", label_field: str = "label") -> None:
    """Write a dataset back out as one JSON object per line

    The counterpart of load_jsonl, and the format to hand to something that is
    not this framework. Groups and label names have nowhere to live here,
    which is one of the reasons prompts.dumps exists.
    """
    lines = [
        json.dumps({text_field: text, label_field: label})
        for text, label in zip(data.texts, data.labels, strict=True)
    ]
    Path(path).write_text("\n".join(lines) + "\n")

_POSITIVE = [
    "I loved every minute of {}",
    "{} was wonderful, easily the best this year",
    "A delightful {}, warm and generous throughout",
    "We had a great time at {}, would go again",
    "{} exceeded every expectation I had",
    "Everyone I spoke to adored {}",
    "There is nothing I would change about {}",
    "{} is the reason I still bother going out",
    "Hours later I was still thinking about {}",
    "I would recommend {} without hesitation",
]
_NEGATIVE = [
    "I hated every minute of {}",
    "{} was dreadful, easily the worst this year",
    "A tedious {}, cold and mean-spirited throughout",
    "We had an awful time at {}, never again",
    "{} fell short of every expectation I had",
    "Everyone I spoke to loathed {}",
    "There is nothing I would keep about {}",
    "{} is the reason I stopped bothering to go out",
    "Hours later I was still annoyed about {}",
    "I would warn anyone away from {} without hesitation",
]
_SUBJECTS = [
    "the film", "the concert", "the restaurant", "the museum", "the play",
    "the hotel", "the exhibition", "the lecture", "the festival", "the tour",
    "the recital", "the gallery", "the workshop", "the matinee",
]

def synthetic(n: int = 100, seed: int = 0) -> LabeledPrompts:
    """A balanced toy sentiment set built from templates

    Sentences are distinct by construction. Sampling templates with
    replacement instead looks fine and quietly puts the same sentence in
    train and test, at which point the reported AUC is measuring how well
    the probe memorized a duplicate.

    The two halves share their subjects and sentence shapes, so a probe that
    separates them has to be using the sentiment rather than the topic. Note
    that the sentiment word is never the last token -- reading this set at the
    last position asks the model to have carried it, which is a real question
    and part of what makes the toy set worth anything.
    """
    if n < 2:
        raise DatasetError(f"a synthetic dataset needs at least 2 examples, got {n}")
    per_class = [n // 2, n - n // 2]
    capacity = len(_SUBJECTS) * len(_POSITIVE)
    if max(per_class) > capacity:
        raise DatasetError(
            f"cannot build {n} distinct examples: {capacity} per class are available, so n is capped at {2 * capacity}"
        )

    rng = random.Random(seed)
    texts: List[str] = []
    labels: List[int] = []
    for label, templates in ((0, _NEGATIVE), (1, _POSITIVE)):
        combinations = [(template, subject) for template in templates for subject in _SUBJECTS]
        for template, subject in rng.sample(combinations, per_class[label]):
            texts.append(template.format(subject))
            labels.append(label)

    order = list(range(len(texts)))
    rng.shuffle(order)
    dataset = LabeledPrompts(texts=texts, labels=labels, name=f"synthetic-{n}")
    return dataset.subset(order)
