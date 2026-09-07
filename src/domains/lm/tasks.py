import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ...data.tasks import (
    CircuitTask,
    TaskError,
    register_frame,
    register_options,
    register_task,
)
from .data.ioi import CORRUPTIONS, FRAMES, IOIDataset, build_ioi
from .data.translation import build_translation
from .readout import LogitDifference, logit_difference_readout

"""
The five tasks a circuit is found on here, and the one shape four of them share.

`data/tasks.py` holds what a task *is* and the registry of them; this holds the
tasks. The split is the second axis: a frame is a sentence in a language, a
pool is a list of words the tokenizer has been asked to approve, and an answer
is a token id, so every line below is about one modality and none of the code
that measures needs to be.

TemplateTask is the shape: one frame, one corruption, one token length, filled
from pools the model's own tokenizer has approved. The three constraints IOI
enforces are not IOI's constraints, they are patching's -- a position that is a
name in one prompt and a preposition in the next makes every grid an average
over two questions -- so `require_alignment` is applied to every task here.

The four cheap ones are cheap on purpose: all of them run on GPT-2 small on a
laptop, and each is a task the literature already asked a circuit question of.

A common pipe could be: build_task | patch_heads | discover | specificity
"""

@dataclass(frozen=True)
class TaskExample:
    """One clean prompt, its corrupted twin, and the two answers between which the task is scored

    `answer` and `distractor` are the exact strings the model sees, leading
    space and all: " are" mid-sentence is a different token from "are" at the
    start of one, and a task that dropped the space would measure a token the
    model was never going to emit.
    """
    clean: str
    corrupted: str
    answer: str
    distractor: str
    variant: str = ""

@dataclass
class TemplateTask:
    """A batch of prompts from one frame, sharing one corruption and one token length"""
    examples: List[TaskExample]
    name: str
    modality: str = "text"
    frame: str = ""
    corruption: str = ""
    description: str = ""
    marks: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def clean(self) -> List[str]:
        return [example.clean for example in self.examples]

    @property
    def corrupted(self) -> List[str]:
        return [example.corrupted for example in self.examples]

    @property
    def variants(self) -> Dict[str, int]:
        """How many examples fall in each variant of the frame

        The control the task rests on, reported rather than assumed. A frame
        with two orderings and every example in one of them measures the
        ordering, not the task.
        """
        counts: Dict[str, int] = {}
        for example in self.examples:
            counts[example.variant] = counts.get(example.variant, 0) + 1
        return counts

    def answers(self, adapter) -> Tuple[List[int], List[int]]:
        """Token ids of the right and wrong answer, one pair per example

        Kept beside `readout` rather than folded into it because the sheaf
        training loop scores against the whole vocabulary and needs the ids
        themselves. It is not part of CircuitTask: a token id is the one thing
        in this file that could not mean anything to another modality.
        """
        answer = [adapter.single_token(example.answer) for example in self.examples]
        distractor = [adapter.single_token(example.distractor) for example in self.examples]
        return answer, distractor

    def readout(self, adapter) -> LogitDifference:
        """The logit difference between this task's two answers, on the model in hand"""
        if not self.examples:
            raise TaskError(f"'{self.name}' is empty, so there is nothing to score")
        answer, distractor = self.answers(adapter)
        return logit_difference_readout(adapter, answer, distractor)

    def labels(self, adapter) -> List[str]:
        """The first prompt as token strings, for labelling a position axis"""
        if not self.examples:
            raise TaskError(f"'{self.name}' is empty, so it has no positions to label")
        return adapter.tokens(self.examples[0].clean)

    def landmarks(self, adapter) -> Dict[str, int]:
        """The positions worth reading a patching grid at, plus the end of the prompt

        END is computed rather than stored because it is the one landmark
        every task has and the one a frame cannot state without knowing how
        its fills tokenized.
        """
        return {**self.marks, "END": len(self.labels(adapter)) - 1}

    def subset(self, indices: Sequence[int]) -> "TemplateTask":
        """The same task over a chosen few of its examples"""
        chosen = list(indices)
        outside = [index for index in chosen if not 0 <= index < len(self.examples)]
        if outside:
            raise TaskError(f"examples {outside} are outside the {len(self.examples)} '{self.name}' holds")
        return TemplateTask(
            examples=[self.examples[index] for index in chosen],
            name=self.name, modality=self.modality, frame=self.frame, corruption=self.corruption,
            description=self.description, marks=dict(self.marks),
        )

def require_alignment(adapter, task: CircuitTask) -> CircuitTask:
    """Reject a task whose prompts do not line up position for position

    Patching reads position i of one run into position i of another. If the
    prompts differ in length, position i is a name in one and a preposition in
    the next, and the grid that comes out is an average over two questions
    with nothing to say that it is.
    """
    prompts = list(task.clean) + list(task.corrupted)
    if not prompts:
        raise TaskError(f"'{task.name}' has no prompts")
    lengths = {len(adapter.tokens(text)) for text in prompts}
    if len(lengths) > 1:
        offender = min(prompts, key=lambda text: (len(adapter.tokens(text)), text))
        raise TaskError(
            f"'{task.name}' tokenizes to {sorted(lengths)} lengths on '{adapter.cfg.id}', not one "
            f"(for instance '{offender}'); patching needs every prompt to line up position for position"
        )
    return task

def single_tokens(adapter, pieces: Sequence[str]) -> List[str]:
    """Keep only the strings this model's tokenizer encodes as exactly one token

    Called with the string as it appears in the prompt, leading space
    included. A pool filtered on the bare word and then used with a space is
    a pool that was never checked.
    """
    kept = []
    for piece in pieces:
        try:
            adapter.single_token(piece)
        except ValueError:
            continue
        kept.append(piece)
    return kept

def _require_pool(adapter, name: str, kept: Sequence[str], asked: Sequence[str], needed: int) -> List[str]:
    """Fail with the pool that survived rather than with an IndexError three frames later"""
    if len(kept) < needed:
        raise TaskError(
            f"'{name}' needs {needed} single-token fills and only {len(kept)} of {len(asked)} survived "
            f"'{adapter.cfg.id}''s tokenizer; widen the pool or pick a model whose vocabulary keeps them whole"
        )
    return list(kept)

# ------------------------------------------------------------------- the tasks

# what `ioi.corruption` and `ioi.frame` may be, declared beside the task rather
# than imported by experiment/spec.py: a spec that names an impossible corruption
# fails before the model loads, and the kernel does not learn what a corruption is
register_options("ioi", corruption=sorted(CORRUPTIONS), frame=list(range(len(FRAMES))))

@register_task("ioi", "which of two names the sentence has not used yet (Wang et al., 2023)")
def _ioi(adapter, size: int = 16, seed: int = 0, **options) -> IOIDataset:
    """The task the circuit literature is built on, built by data/ioi.py"""
    return build_ioi(adapter, size=size, seed=seed, **options)

@register_task("translation", "the English for a Spanish word (translation-circuit design, Phase 0)")
def _translation(adapter, size: int = 16, seed: int = 0, **options) -> TemplateTask:
    """The word-level ES->EN task, built by data/translation.py; sentence-level
    quality lives in the bitext loaders there rather than in this registry,
    because natural sentences can never satisfy patching's alignment constraint."""
    return build_translation(adapter, size=size, seed=seed, **options)

GREATER_THAN_FRAME = "The{noun} lasted from the year 17{start} to the year 17"
GREATER_THAN_NOUNS = (
    " war", " siege", " journey", " voyage", " reign", " storm", " drought", " conflict", " revolt", " project",
)

@register_task("greater_than", "a year later than the one just named (Hanna et al., 2023)")
def _greater_than(adapter, size: int = 16, seed: int = 0, low: int = 20, high: int = 80, **options) -> TemplateTask:
    """Continue a date range, where any year before the start is wrong

    The corruption is the one the task was introduced with: set the start year
    to 01, which leaves a perfectly well-formed sentence in which *every*
    two-digit year is a valid continuation. So the clean run has to prefer a
    late year over an early one and the corrupted run has no reason to, which
    is exactly the span patching gets to recover.
    """
    if not 1 <= low < high <= 98:
        raise TaskError(f"the start year has to leave room above and below it, got low={low} high={high}")
    rng = random.Random(seed)
    asked = [f"{value:02d}" for value in range(1, 99)]
    years = _require_pool(adapter, "greater_than", single_tokens(adapter, asked), asked, needed=20)
    nouns = _require_pool(
        adapter, "greater_than", single_tokens(adapter, GREATER_THAN_NOUNS), GREATER_THAN_NOUNS, needed=1
    )
    available = {int(year): year for year in years}
    if 1 not in available:
        raise TaskError(
            f"'01' is not a single token on '{adapter.cfg.id}', so the corruption cannot be written; "
            "this task needs a tokenizer that keeps two-digit years whole"
        )
    starts = [
        value for value in range(low, high)
        if value in available
        and any(other > value for other in available)
        and any(other < value for other in available)
    ]
    if not starts:
        raise TaskError(
            f"no start year between {low} and {high} has both a later and an earlier single-token year "
            f"on '{adapter.cfg.id}'; widen the range"
        )

    examples = []
    for _ in range(size):
        start = rng.choice(starts)
        # one noun per example, not one per prompt: the corruption is the start year and
        # nothing else, and a clean/corrupted pair differing in two things measures neither
        noun = rng.choice(nouns)
        later = [value for value in available if value > start]
        earlier = [value for value in available if value < start]
        examples.append(TaskExample(
            clean=GREATER_THAN_FRAME.format(noun=noun, start=available[start]),
            corrupted=GREATER_THAN_FRAME.format(noun=noun, start=available[1]),
            answer=available[rng.choice(later)],
            distractor=available[rng.choice(earlier)],
            variant="later" if start >= (low + high) // 2 else "earlier",
        ))
    return require_alignment(adapter, TemplateTask(
        examples=examples, name="greater-than", frame=GREATER_THAN_FRAME, corruption="year-01",
        description="prefer a year after the start of the range to one before it",
    ))

INDUCTION_WORDS = (
    " apple", " table", " river", " cloud", " stone", " window", " garden", " silver", " forest", " candle",
    " mountain", " letter", " bridge", " ocean", " pencil", " summer", " tiger", " marble", " valley", " lantern",
)

@register_task("induction", "copy what followed this token last time (Olsson et al., 2022)")
def _induction(adapter, size: int = 16, seed: int = 0, length: int = 5, **options) -> TemplateTask:
    """Repeat a token from earlier in a random list and predict what followed it

    The corruption removes the repetition and nothing else: the final token
    becomes a word the list has not used, so the prefix that made the answer
    predictable is gone while the list, its length and every other position
    stay exactly as they were.
    """
    if length < 3:
        raise TaskError(f"an induction list needs at least three words to have a distractor, got length={length}")
    pool = _require_pool(
        adapter, "induction", single_tokens(adapter, INDUCTION_WORDS), INDUCTION_WORDS, needed=length + 2
    )
    rng = random.Random(seed)

    examples = []
    for _ in range(size):
        drawn = rng.sample(pool, length + 1)
        words, unseen = drawn[:length], drawn[length]
        prefix = "Words:" + "".join(words)
        examples.append(TaskExample(
            clean=prefix + words[0],
            corrupted=prefix + unseen,
            answer=words[1],
            # a word from the same list, so the difference is what followed the repeat
            # rather than the model's taste in nouns
            distractor=words[length - 1],
            variant="repeat",
        ))
    return require_alignment(adapter, TemplateTask(
        examples=examples, name="induction", frame="Words: w1 .. wn w1", corruption="no-repeat",
        description="after a repeated token, predict what followed it the first time",
    ))

AGREEMENT_FRAME = "The{subject} near the{attractor}"
AGREEMENT_NOUNS = (
    (" key", " keys"), (" door", " doors"), (" book", " books"), (" light", " lights"), (" road", " roads"),
    (" painting", " paintings"), (" window", " windows"), (" letter", " letters"), (" bridge", " bridges"),
    (" garden", " gardens"),
)

@register_task("agreement", "which verb form the subject takes, across an attractor")
def _agreement(adapter, size: int = 16, seed: int = 0, **options) -> TemplateTask:
    """Choose the verb that agrees with the subject rather than with the noun beside it

    The attractor always carries the opposite number to the subject, so a
    model reading the nearest noun gets it wrong every time. The corruption
    flips the subject's number, which flips the correct verb -- the same shape
    as the IOI 'swap' corruption, and it opens the widest span two verbs can.
    """
    pairs = [
        pair for pair in AGREEMENT_NOUNS
        if len(single_tokens(adapter, pair)) == 2
    ]
    _require_pool(adapter, "agreement", pairs, AGREEMENT_NOUNS, needed=2)
    verbs = _require_pool(adapter, "agreement", single_tokens(adapter, (" is", " are")), (" is", " are"), needed=2)
    singular, plural = verbs[0], verbs[1]
    rng = random.Random(seed)

    examples = []
    for index in range(size):
        subject, attractor = rng.sample(pairs, 2)
        # the two numbers alternate rather than being sampled, so a task of any even
        # size is exactly balanced and one of any size is off by at most one example
        number, opposite = (1, 0) if index % 2 == 0 else (0, 1)
        subject_plural = number == 1
        examples.append(TaskExample(
            clean=AGREEMENT_FRAME.format(subject=subject[number], attractor=attractor[opposite]),
            corrupted=AGREEMENT_FRAME.format(subject=subject[opposite], attractor=attractor[opposite]),
            answer=plural if subject_plural else singular,
            distractor=singular if subject_plural else plural,
            variant="plural" if subject_plural else "singular",
        ))
    return require_alignment(adapter, TemplateTask(
        examples=examples, name="agreement", frame=AGREEMENT_FRAME, corruption="flip-number",
        description="agree with the subject, not with the noun nearest the verb",
    ))


@register_frame("translation")
def _translation_frame(source: str) -> str:
    """`Spanish: perro\nEnglish:` -- the frame every translation circuit was pruned under

    Imported from data/translation.py rather than restated, because a server
    framing a word differently from the run that learned the circuit is asking
    the circuit a question it was never trained on, and the answer would look
    like a bad circuit.
    """
    from .data.translation import WORD_FRAME

    return WORD_FRAME.format(source=f" {source.strip()}", answer="")
