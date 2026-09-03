import random
from dataclasses import dataclass
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

This task needs a model that can actually translate, and that rules out the
one every other experiment here debugs on. GPT-2 small keeps 179 of MUSE's
es-en pairs whole in its BPE, so the tokenizer is not the obstacle -- but it
picks the right English word for 2 of those 179, against a chance rate of 0.6%.
A 124M English-only model has no Spanish-to-English behaviour, and a circuit
cannot be found for a behaviour that is not there. Qwen3-1.7B scores 420 of 971
on the same test. So the repo's "run it on gpt2-small first" rule does not
apply to this task, and the substitute is to debug the *method* on gpt2-small
with `ioi` and spend the large model only on translation.

The sentence shape has one more convention, `eval_split`, which is the way a
pairs file becomes an evaluation: the last `SHOTS` pairs are the worked
examples every prompt shares and the first `size` are what gets scored. The
shots come off the *end* so that growing `size` never turns a scored sentence
into a shot, and the same split names the counterfactual reference prompts a
mean ablation averages over -- the eval prompt with its translation logic
removed and nothing else changed.

A common pipe could be: load_pairs | eval_split | generate | clean_completion | score
"""

# One frame, ending where the English word has to go. The trailing space stays
# out of the prompt: the answer token carries it, the way IOI's names do.
#
# The glossary shape, not the sentence. "The Spanish word costo means" was the
# frame through every run under results/qwen3-1.7b-sweep/ up to kl-t005, and
# at that position the full model's next token is ` "` on 274 of 300 pool
# words -- it continues `means "cost" and ...` -- so the answer sat one token
# past the position every loss trained and every ranking scored. A mask that
# opened a quote was faithful, and every collapsed circuit emitted ` " " "`.
# Ending in `English:` the answer *is* the next token on 74% of the pool with
# the same leading-space token as before; `means "` scores the same but
# changes the answer token and drops 44 pairs, and two worked shots add
# nothing. (scratchpad frames.py, seven frames, 300 pairs, 2026-09-03.)
WORD_FRAME = "Spanish:{source}\nEnglish:{answer}"

# The offline fallback, and only that. Forty pairs written down by hand, of
# which Qwen3-1.7B keeps twenty-five whole, which a 75/25 split turns into
# eighteen training prompts -- and a whole-model sheaf put 28M open gates
# against those eighteen and memorized them (train 0.917, held-out 0.125).
# `scripts.build_translation_pool` derives a real pool from MUSE and writes it
# beside the bitexts; `load_word_pairs` prefers it whenever it exists. This
# list stays so that the task is buildable with no download, which is what the
# tests rely on.
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

# The reference distribution a mean ablation is allowed to average over.
# Wang et al. (2211.00593 3) are explicit that the knockout mean has to come
# from a distribution that strips the task information and leaves everything
# else standing: p_ABC keeps the p_IOI templates and their grammatical roles
# and varies only the names, "because using p_IOI would not remove enough
# information helpful for the task". Zhang et al. (2502.11806 3) build their
# counterfactual X- on the same principle -- preserve the grammatical
# structure, replace only the words carrying the translation logic
# ("English: cloud - Nothing: _").
#
# So the counterfactual form is the few-shot form with the translation logic
# taken out and nothing else touched: same skeleton, same shot count, the same
# real Spanish source sentence, and the same shot target sentences token for
# token -- only the language labels are neutralised and the shot targets are
# rotated off their own sources, so the worked examples no longer demonstrate
# a translation relation. Averaging over raw English text instead would remove
# the prompt format and the in-context task along with the translation, and a
# component knocked out toward that mean is being told to forget how to
# continue a prompt at all.
COUNTERFACTUAL_HEADER = "Text: {source}\nNothing: {target}\n\n"
COUNTERFACTUAL_FORM = "Text: {source}\nNothing:"

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
    if form == "counterfactual":
        if len(shots) < 2:
            raise DatasetError(
                "the counterfactual form needs at least two worked examples: it removes the translation "
                "relation by rotating each shot's target onto a different shot's source, which needs "
                "somewhere to rotate to"
            )
        targets = [english for _, english in shots]
        rotated = targets[1:] + targets[:1]
        header = "".join(
            COUNTERFACTUAL_HEADER.format(source=spanish, target=target)
            for (spanish, _), target in zip(shots, rotated, strict=True)
        )
        return header + COUNTERFACTUAL_FORM.format(source=source)
    raise DatasetError(
        f"unknown prompt form '{form}'; known forms are ['counterfactual', 'few_shot', 'instruction']"
    )

# Worked examples per few-shot prompt. Phase 0 settled on three: the smallest
# count at which the model stopped treating the frame as an instruction to
# ignore, and small enough that the prompt stays one screen.
SHOTS = 3

def clean_completion(text: str) -> str:
    """The first line of the completion is the translation; the rest is drift"""
    return text.strip().split("\n")[0].strip()

@dataclass(frozen=True)
class EvalSplit:
    """A pairs file cut into what is prompted, what it is scored against, and what every prompt shares"""
    shots: Tuple[Tuple[str, str], ...]
    pairs: Tuple[Tuple[str, str], ...]
    form: str = "few_shot"

    @property
    def sources(self) -> List[str]:
        return [spanish for spanish, _ in self.pairs]

    @property
    def references(self) -> List[str]:
        return [english for _, english in self.pairs]

    @property
    def prompts(self) -> List[str]:
        return [translation_prompt(spanish, form=self.form, shots=self.shots) for spanish in self.sources]

    def __len__(self) -> int:
        return len(self.pairs)

def eval_split(pairs: Sequence[Tuple[str, str]], size: Optional[int] = None, shots: int = SHOTS,
               form: str = "few_shot") -> EvalSplit:
    """The last `shots` pairs as worked examples, the first `size` as the scored set

    A pair cannot be both: the split refuses a `size` that reaches into the
    shots, because a sentence the model was shown translated is not a sentence
    it translated.
    """
    pairs = list(pairs)
    if shots < 1 or shots >= len(pairs):
        raise DatasetError(f"{shots} shots out of {len(pairs)} pairs leaves nothing to score")
    scored = pairs[:size] if size is not None else pairs[:-shots]
    if len(scored) > len(pairs) - shots:
        raise DatasetError(
            f"{len(scored)} scored sentences and {shots} shots overlap in a file of {len(pairs)} pairs; "
            f"the most this file can score is {len(pairs) - shots}"
        )
    return EvalSplit(shots=tuple(pairs[-shots:]), pairs=tuple(scored), form=form)

def counterfactual_prompts(pairs: Sequence[Tuple[str, str]], shots: int = SHOTS) -> List[str]:
    """The distribution a mean ablation averages over: every non-shot source in the counterfactual form

    The mean has to strip the task and preserve everything else -- the same
    skeleton, the same shots, the same Spanish sources -- which is what the
    counterfactual form is. An earlier version averaged over raw English
    reference sentences, which differ from the eval prompts in format and
    in-context task as well as in translation, and ablating toward it removed
    the model's ability to continue the prompt at all.
    """
    pairs = list(pairs)
    worked = pairs[-shots:]
    return [translation_prompt(spanish, form="counterfactual", shots=worked) for spanish, _ in pairs[:-shots]]

def pool_path(config: str) -> Path:
    """Where the derived word pool for one model lands

    Per model, because the filter that produces it is the model's own
    tokenizer and the model's own answers -- a pool built for Qwen is not a
    pool for GPT-2, and one file named for both would silently be read as
    either.
    """
    return (Path(__file__).resolve().parents[2] / "data" / "external" / "translation"
            / f"es-en-words-{config}.tsv")

def load_word_pairs(config: str) -> Tuple[Tuple[str, str], ...]:
    """The derived pool if it has been built, otherwise the built-in fallback

    Silent about which it used, because `build_translation` reports the count
    it kept and the caller can see eighteen against four hundred. What is not
    silent is the failure: a pool file that exists and cannot be parsed raises
    rather than falling back, since falling back to twenty-five pairs while
    the caller believes it has four hundred is the failure this whole change
    exists to remove.
    """
    path = pool_path(config)
    if not path.exists():
        return WORD_PAIRS
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise DatasetError(
                f"{path}:{number} is not `spanish<TAB>english`: {line!r}. Rebuild it with "
                f"`uv run python -m scripts.build_translation_pool {config}`."
            )
        pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise DatasetError(f"{path} holds no pairs; rebuild or delete it to use the built-in list")
    return tuple(pairs)

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

    available = load_word_pairs(adapter.cfg.id)
    kept = [
        (spanish, english)
        for spanish, english in available
        if single_tokens(adapter, (f" {spanish}",)) and single_tokens(adapter, (f" {english}",))
    ]
    if len(kept) < 4:
        raise TaskError(
            f"'translation' needs at least 4 word pairs kept whole and only {len(kept)} of "
            f"{len(available)} survived '{adapter.cfg.id}''s tokenizer. Build the derived pool "
            f"with `uv run python -m scripts.build_translation_pool {adapter.cfg.id}`, or pick a "
            "model whose vocabulary keeps common Spanish words whole"
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
