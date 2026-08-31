import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import DatasetError, LabeledPrompts
from .prompts import load_prompts

r"""
Spanish-to-English translation as data, in the two shapes the translation
circuit study needs and with the two shapes kept deliberately separate.

The first shape is *sentences*: parallel es/en pairs (FLORES, WMT dev) stored
in the repo's plain-text prompt format, one pair per group -- `- ` the Spanish
line, `+ ` its indented English twin -- so a split can never cut a pair in half
and the file reads as the bitext it is. `load_pairs` turns such a file back
into (spanish, english) tuples; `translation_prompt` wraps a source sentence in
whichever prompt form Phase 0 settled on. Sentences are what generation and
COMET/BLEU are scored on, and what the neuron scan feeds the model.

The second shape is a *CircuitTask*, registered in data/tasks.py like every
other task. Natural sentences can never satisfy patching's alignment
constraint -- two FLORES lines tokenize to two lengths -- so the registered
task is word-level: one frame naming a Spanish word, completed by its English
translation, corrupted by swapping the Spanish word for another. That is the
IOI shape (single-token answers, one frame, one corruption) applied to
translation, and it is the task component-level circuit work runs on while the
sentence pairs remain the task the *quality* numbers come from.

A common pipe could be: load_pairs | translation_prompt | generate | score
"""

# One frame, ending where the English word has to go. The trailing space stays
# out of the prompt: the answer token carries it, the way IOI's names do.
WORD_FRAME = "The Spanish word{source} means{answer}"

# Pairs whose two sides are likely one token each with a leading space on a
# multilingual BPE. A starting point, not a promise: build_translation filters
# both sides against the tokenizer of the model in hand.
WORD_PAIRS = (
    ("gato", "cat"), ("perro", "dog"), ("casa", "house"), ("libro", "book"),
    ("agua", "water"), ("sol", "sun"), ("luna", "moon"), ("rojo", "red"),
    ("verde", "green"), ("leche", "milk"), ("pan", "bread"), ("noche", "night"),
    ("hombre", "man"), ("mujer", "woman"), ("tiempo", "time"), ("mano", "hand"),
    ("cabeza", "head"), ("puerta", "door"), ("mesa", "table"), ("calle", "street"),
    ("ciudad", "city"), ("rey", "king"), ("reina", "queen"), ("guerra", "war"),
    ("paz", "peace"), ("fuego", "fire"), ("nieve", "snow"), ("lluvia", "rain"),
    ("viento", "wind"), ("flor", "flower"), ("caballo", "horse"), ("sangre", "blood"),
    ("oro", "gold"), ("plata", "silver"), ("madre", "mother"), ("padre", "father"),
    ("mar", "sea"), ("cielo", "sky"), ("carta", "letter"), ("dinero", "money"),
)

INSTRUCTION_FORM = (
    "Translate the following Spanish sentence into English.\n"
    "Spanish: {source}\n"
    "English:"
)

# The WMT-style form: no instruction at all, k worked examples and an
# unfinished frame. Which of the two forms actually elicits English from the
# model is a Phase 0 question (Wu et al. 2601.11019: instruction-form and
# few-shot recruit different task-initiation features), so both are kept.
FEW_SHOT_HEADER = "Spanish: {source}\nEnglish: {target}\n\n"
FEW_SHOT_FORM = "Spanish: {source}\nEnglish:"

def pairs_to_prompts(pairs: Sequence[Tuple[str, str]], name: str) -> LabeledPrompts:
    """Lay parallel sentences out as a grouped prompt set: label 0 Spanish, label 1 English

    One group per pair is the invariant the format exists for -- split() keeps
    a group whole, so no downstream cut can put a sentence and its translation
    on opposite sides.
    """
    if not pairs:
        raise DatasetError(f"'{name}' has no sentence pairs")
    texts: List[str] = []
    labels: List[int] = []
    groups: List[int] = []
    for index, (spanish, english) in enumerate(pairs):
        if not spanish.strip() or not english.strip():
            raise DatasetError(f"'{name}' pair {index} has an empty side")
        texts.extend((spanish, english))
        labels.extend((0, 1))
        groups.extend((index, index))
    return LabeledPrompts(texts=texts, labels=labels, name=name, groups=groups, label_names=("spanish", "english"))

def load_pairs(path: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Read a bitext prompt set back into (spanish, english) tuples

    Refuses a group that is not exactly one Spanish line and one English line:
    a bitext with a dangling half is one whose scores silently compare a
    sentence against the wrong reference.
    """
    data = load_prompts(path, limit=None if limit is None else 2 * limit)
    if data.groups is None:
        raise DatasetError(f"{path} has no groups, so it does not pair sentences with translations")
    by_group: Dict[int, List[Tuple[int, str]]] = {}
    for text, label, group in zip(data.texts, data.labels, data.groups, strict=True):
        by_group.setdefault(group, []).append((label, text))
    pairs = []
    for group in sorted(by_group):
        members = by_group[group]
        if sorted(label for label, _ in members) != [0, 1]:
            raise DatasetError(
                f"{path} group {group} holds {len(members)} lines with labels "
                f"{sorted(label for label, _ in members)}, not one Spanish and one English"
            )
        spanish = next(text for label, text in members if label == 0)
        english = next(text for label, text in members if label == 1)
        pairs.append((spanish, english))
    return pairs if limit is None else pairs[:limit]

def translation_prompt(source: str, form: str = "instruction", shots: Sequence[Tuple[str, str]] = ()) -> str:
    """Wrap a Spanish sentence in the prompt form the model will be asked to continue"""
    if form == "instruction":
        return INSTRUCTION_FORM.format(source=source)
    if form == "few_shot":
        if not shots:
            raise DatasetError("the few-shot form needs at least one worked (spanish, english) example")
        header = "".join(FEW_SHOT_HEADER.format(source=spanish, target=english) for spanish, english in shots)
        return header + FEW_SHOT_FORM.format(source=source)
    raise DatasetError(f"unknown prompt form '{form}'; known forms are ['few_shot', 'instruction']")

def default_pairs_path(name: str) -> Path:
    """Where a converted bitext lands: downloaded data stays under data/external"""
    return Path(__file__).resolve().parents[2] / "data" / "external" / "translation" / f"{name}.prompts"

def build_translation(adapter, size: int = 16, seed: int = 0, **options):
    """The word-level translation task, one IOI-shaped frame over the pair pool

    Clean names a Spanish word and ends where its English translation goes;
    the corruption swaps the Spanish word and nothing else, so the logit
    difference between the two answers is exactly the translation signal.
    Both sides of every pair are filtered against the tokenizer in hand,
    leading space included, because the frame supplies no space of its own.
    """
    from .tasks import TaskError, TaskExample, TemplateTask, require_alignment, single_tokens

    kept = [
        (spanish, english)
        for spanish, english in WORD_PAIRS
        if single_tokens(adapter, (f" {spanish}",)) and single_tokens(adapter, (f" {english}",))
    ]
    if len(kept) < 4:
        raise TaskError(
            f"'translation' needs at least 4 word pairs kept whole and only {len(kept)} of "
            f"{len(WORD_PAIRS)} survived '{adapter.cfg.id}''s tokenizer; widen WORD_PAIRS or "
            "pick a model whose vocabulary keeps common Spanish words whole"
        )
    rng = random.Random(seed)
    examples = []
    for _ in range(size):
        (spanish, english), (other_spanish, other_english) = rng.sample(kept, 2)
        examples.append(TaskExample(
            clean=WORD_FRAME.format(source=f" {spanish}", answer=""),
            corrupted=WORD_FRAME.format(source=f" {other_spanish}", answer=""),
            answer=f" {english}",
            distractor=f" {other_english}",
            variant="swap-source",
        ))
    return require_alignment(adapter, TemplateTask(
        examples=examples, name="translation", frame=WORD_FRAME, corruption="swap-source",
        description="complete a Spanish word's English translation",
    ))
