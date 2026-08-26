import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..core.metrics import logit_difference

"""
Indirect Object Identification is the task the circuit literature is built on
(Wang et al., 2023). A sentence names two people, mentions one of them again,
and the sentence ends where the *other* one has to go: "Then, John and Mary
went to the store. Mary gave a ring to" wants "John". Everything a circuit
study measures on it is one number, logit(John) - logit(Mary), and every claim
about a head is a claim about how that number moves.

Three constraints make the task readable, and all three are enforced here
rather than assumed:

- Every name is one token. Two tokens and the answer is spread over several
  logits, so the difference stops measuring the thing it is named after.
- Every prompt in a dataset comes from one frame and has the same token
  length. Patching swaps position i of one run into position i of another; if
  position 7 is a name in one prompt and a preposition in the next, the
  heatmap that comes out is an average over two different questions.
- The two name orders are balanced. In ABBA the answer is the first name and
  in BABA it is the second, so a model that always says the first name scores
  half. Without the balance, "the circuit" can be position and nothing else.

The corrupted counterpart is where the causal claim comes from. 'abc' replaces
the repeated name with an unrelated third one, which leaves a well-formed
sentence and destroys only the duplicate-name signal the circuit runs on.
'swap' exchanges the two names in the opening clause, which flips which answer
is correct -- a harder corruption that moves the logit difference further, at
the cost of changing two things at once.

A common pipe could be: build_ioi | evaluate | direct_logit_attribution
"""

class IOIError(ValueError):
    """Raised when an IOI dataset cannot be built honestly: names that split, prompts of unequal length"""

FRAMES = (
    "Then, {first} and {second} went to the {place}. {third} gave a {object} to",
    "Then, {first} and {second} had a lot of fun at the {place}. {third} gave a {object} to",
)

# Names GPT-2's BPE is likely to keep whole. The list is a starting point, not
# a promise: build_ioi filters it against the tokenizer of the model in hand,
# because a name that is one token on GPT-2 can be three somewhere else.
NAMES = (
    "John", "Mary", "Tom", "Sarah", "Mike", "Anna", "Paul", "Kate", "Alice", "Bob",
    "Frank", "Henry", "Jack", "Sam", "Jim", "Lee", "Alex", "Emma", "Dan", "Rose",
)
PLACES = ("store", "school", "park", "station", "market")
OBJECTS = ("ring", "book", "drink", "snack", "ball")

ORDERS = ("ABBA", "BABA")
CORRUPTIONS = ("abc", "swap")

@dataclass(frozen=True)
class IOIExample:
    """One clean prompt, its corrupted twin, and the two names the answer is between"""
    clean: str
    corrupted: str
    io: str
    subject: str
    order: str

@dataclass
class IOIDataset:
    """A batch of IOI prompts sharing one frame, one corruption and one token length"""
    examples: List[IOIExample]
    frame: str
    corruption: str
    name: str = "ioi"

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def clean(self) -> List[str]:
        return [example.clean for example in self.examples]

    @property
    def corrupted(self) -> List[str]:
        return [example.corrupted for example in self.examples]

    @property
    def balance(self) -> float:
        """Share of examples in ABBA order, which should sit at a half"""
        if not self.examples:
            return 0.0
        return sum(example.order == "ABBA" for example in self.examples) / len(self.examples)

    def answers(self, adapter) -> Tuple[List[int], List[int]]:
        """Token ids of the indirect object and the subject, one pair per example

        Both are read from the model in hand rather than cached on the
        dataset: the ids are the tokenizer's opinion, and a dataset that
        carried them would be silently wrong the moment it met another model.
        """
        io = [adapter.single_token(" " + example.io) for example in self.examples]
        subject = [adapter.single_token(" " + example.subject) for example in self.examples]
        return io, subject

    def token_labels(self, adapter) -> List[str]:
        """The first prompt as token strings, for labelling a position axis"""
        if not self.examples:
            raise IOIError("an empty dataset has no positions to label")
        return adapter.tokens(self.examples[0].clean)

    def landmarks(self, adapter) -> Dict[str, int]:
        """Where the three name mentions and the end of the sentence sit, by position

        These are the positions worth reading a patching heatmap at. They are
        the same for every example because the frame and the name lengths are,
        which is the whole reason a dataset is allowed only one frame.
        """
        if not self.examples:
            raise IOIError("an empty dataset has no landmarks")
        example = self.examples[0]
        tokens = adapter.tokens(example.clean)
        # the names are compared as the strings the model sees, leading space and all:
        # " John" mid-sentence is a different token from "John" at the start of one
        wanted = (" " + example.io, " " + example.subject)
        mentions = [index for index, token in enumerate(tokens) if token in wanted]
        if len(mentions) != 3:
            raise IOIError(
                f"expected three name mentions in '{example.clean}', found {len(mentions)}; "
                "the frame must name the two people once each and one of them again"
            )
        first, second, third = mentions
        io_position, subject_position = (first, second) if example.order == "ABBA" else (second, first)
        return {"IO": io_position, "S1": subject_position, "S2": third, "END": len(tokens) - 1}

def single_token_names(adapter, names: Sequence[str] = NAMES) -> List[str]:
    """Keep only the names this model's tokenizer encodes as one token

    Leading space included, because that is how a name appears mid-sentence
    and it is a different token from the same name at the start of one.
    """
    kept = []
    for name in names:
        try:
            adapter.single_token(" " + name)
        except ValueError:
            continue
        kept.append(name)
    return kept

def _sentence(
    frame: str, io: str, subject: str, order: str, place: str, obj: str, repeated: Optional[str] = None
) -> str:
    """Fill a frame: two names in the opening clause, then one of them again

    order says which name comes first, and `repeated` says which one the
    second sentence mentions -- normally the subject, and something else when
    the point is to break exactly that.
    """
    first, second = (io, subject) if order == "ABBA" else (subject, io)
    return frame.format(first=first, second=second, third=repeated or subject, place=place, object=obj)

def build_ioi(
    adapter,
    size: int = 32,
    seed: int = 0,
    frame: int = 0,
    corruption: str = "abc",
    names: Optional[Sequence[str]] = None,
) -> IOIDataset:
    """Build a balanced batch of clean/corrupted IOI pairs for the model in hand

    The two name orders alternate rather than being sampled, so a dataset of
    any even size is exactly balanced and a dataset of any size is off by at
    most one example. Sampling them would leave the balance to chance, and the
    balance is the control the whole task rests on.
    """
    if corruption not in CORRUPTIONS:
        raise IOIError(f"unknown corruption '{corruption}'; known corruptions are {sorted(CORRUPTIONS)}")
    if not 0 <= frame < len(FRAMES):
        raise IOIError(f"frame {frame} is outside the {len(FRAMES)} frames this module ships")
    if size < 1:
        raise IOIError(f"an IOI dataset needs at least one example, got {size}")

    pool = single_token_names(adapter, names or NAMES)
    if len(pool) < 3:
        raise IOIError(
            f"only {len(pool)} of {len(names or NAMES)} names are single tokens on '{adapter.cfg.id}'; "
            "IOI needs at least three (two in the sentence and one to corrupt it with)"
        )

    template = FRAMES[frame]
    rng = random.Random(seed)
    examples = []
    for index in range(size):
        io, subject, distractor = rng.sample(pool, 3)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        order = ORDERS[index % len(ORDERS)]

        clean = _sentence(template, io, subject, order, place, obj)
        if corruption == "abc":
            corrupted = _sentence(template, io, subject, order, place, obj, repeated=distractor)
        else:
            # exchange the two roles, keeping the order slot: the repeated name becomes the
            # one that was the answer, so the correct answer becomes the other one
            corrupted = _sentence(template, subject, io, order, place, obj)
        examples.append(IOIExample(clean=clean, corrupted=corrupted, io=io, subject=subject, order=order))

    dataset = IOIDataset(examples=examples, frame=template, corruption=corruption, name=f"ioi-{corruption}")
    _require_one_length(adapter, dataset)
    return dataset

def _require_one_length(adapter, dataset: IOIDataset) -> None:
    """Reject a dataset whose prompts do not line up position for position

    Patching reads position i of one run into position i of another. If the
    prompts differ in length, position i is a name in one and a preposition in
    the next, and the resulting heatmap is an average over two questions with
    no warning that it is.
    """
    lengths = {len(adapter.tokens(text)) for text in dataset.clean + dataset.corrupted}
    if len(lengths) > 1:
        offender = min(
            dataset.clean + dataset.corrupted, key=lambda text: (len(adapter.tokens(text)), text)
        )
        raise IOIError(
            f"prompts tokenize to {sorted(lengths)} tokens on '{adapter.cfg.id}', not one length "
            f"(for instance '{offender}'); patching needs every prompt to line up position for position"
        )

@dataclass(frozen=True)
class IOIReport:
    """How well a model does the task, before any circuit question is asked"""
    n: int
    accuracy: float
    clean: float
    corrupted: float
    differences: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    @property
    def span(self) -> float:
        """How far the corruption moved the behaviour, which is what patching gets to recover"""
        return self.clean - self.corrupted

    def __str__(self) -> str:
        return (
            f"{self.n} prompts  accuracy {self.accuracy:.1%}  "
            f"logit diff {self.clean:+.3f} clean / {self.corrupted:+.3f} corrupted (span {self.span:+.3f})"
        )

def evaluate(adapter, dataset: IOIDataset) -> IOIReport:
    """Run both halves of the dataset and report the behaviour the circuit has to explain

    Accuracy is quoted on the clean prompts because that is the claim about
    the model; the span between the clean and corrupted logit differences is
    quoted because that is what every later number is a fraction of. A span
    near zero means the corruption did not corrupt anything, and no amount of
    patching on top of it will mean much.
    """
    if not len(dataset):
        raise IOIError("an empty dataset has nothing to evaluate")
    io, subject = dataset.answers(adapter)
    clean = logit_difference(adapter.logits(dataset.clean), io, subject)
    corrupted = logit_difference(adapter.logits(dataset.corrupted), io, subject)
    return IOIReport(
        n=len(dataset),
        accuracy=float((clean > 0).to(torch.float64).mean()),
        clean=float(clean.mean()),
        corrupted=float(corrupted.mean()),
        differences=clean.float(),
    )
