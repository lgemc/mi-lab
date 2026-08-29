import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .dataset import DatasetError, LabeledPrompts, load_jsonl

r"""
The plain-text prompt set: what a dataset looks like when a human has to
read it, review it and edit it.

Interpretability data is small, hand-made and argued over. A few hundred
prompts decide what a probe learns, and the interesting question about a
dataset -- are the two classes matched, or does one of them just talk about
different things? -- is answered by reading it, not by loading it. JSONL
answers that question badly: every prompt is wrapped in punctuation, a diff
of a fixed typo shows the whole line, and a review of 200 examples is a
review of 200 lines of syntax.

So the format is one example per line, with the label as the first character:

    # comments and blank lines are ignored anywhere
    name: sentiment-templates          the header, before the first example
    labels: negative, positive         what 0 and 1 mean, in that order

    + I loved every minute of the film     an example of label 1
    - I hated every minute of the film     an example of label 0
      - The bridge is lovely               indented: same group as the line above

Three decisions in there are worth stating.

**The label is a sigil, not a word.** `+` and `-` are one column wide, so the
prompts line up and an unbalanced or badly matched class is visible by
squinting at the left edge of the file.

**Indentation means "same item".** A minimal pair is two prompts differing by
one word, and the two halves must never end up on opposite sides of a split;
an indented line joins the group above it and dataset.split keeps a group
whole. This is the only structure the format has, and it exists because the
alternative is a silent leak.

**Whitespace is written down.** A prompt ending in a space tokenizes
differently from one that does not, and that difference is invisible in an
editor, so trailing whitespace is stripped and `\s` means a space that is
meant to be there. `\n`, `\t` and `\\` are the other escapes; anything
else after a backslash is an error rather than a guess.

A common pipe could be: load_prompts | split | capture | train_probe
"""

SUFFIXES = (".prompts", ".txt")
COMMENT = "#"
MARKERS = {"-": 0, "+": 1}
HEADER_KEYS = ("name", "labels", "notes", "source")
ESCAPES = {"n": "\n", "t": "\t", "s": " ", "\\": "\\"}

_HEADER = re.compile(r"^([a-z][a-z_]*):(.*)$")
_EXAMPLE = re.compile(r"^(\s*)([-+])(\s+)(.*)$")
_BARE_MARKER = re.compile(r"^\s*[-+]\s*$")

@dataclass(frozen=True)
class Example:
    """One example line: its text, its label, and the group it was written into"""
    text: str
    label: int
    group: int

def _unescape(text: str, source: str, number: int) -> str:
    r"""Turn the escapes a prompt set may contain into the characters they name

    An unknown escape is an error rather than a literal backslash. In a file
    of prompts a stray \q is a typo far more often than it is intent, and a
    prompt whose whitespace is not what its author thought is exactly the bug
    that surfaces three days later as an unexplained drop in AUC.
    """
    if "\\" not in text:
        return text
    out: List[str] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            out.append(text[index])
            index += 1
            continue
        if index + 1 == len(text):
            raise DatasetError(f"{source}:{number} ends in a lone backslash; a literal one is written '\\\\'")
        escape = text[index + 1]
        if escape not in ESCAPES:
            raise DatasetError(
                f"{source}:{number} has unknown escape '\\{escape}'; "
                f"known escapes are {sorted(ESCAPES)} and a literal backslash is '\\\\'"
            )
        out.append(ESCAPES[escape])
        index += 2
    return "".join(out)

def _escape(text: str) -> str:
    r"""Write a prompt so that reading it back gives the same string

    Only whitespace at the ends is escaped as \s: whitespace in the middle of
    a sentence survives the round trip on its own and escaping it would make
    the file harder to read, which is the entire point of the format.
    """
    escaped = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    if escaped.startswith(" "):
        escaped = "\\s" + escaped[1:]
    if escaped.endswith(" "):
        escaped = escaped[:-1] + "\\s"
    return escaped

def scan(lines: Iterable[str], source: str = "<text>", meta: Optional[Dict[str, str]] = None) -> Iterator[Example]:
    """Parse a prompt set line by line, yielding one Example at a time

    A generator rather than a list, so a file bigger than memory can be
    streamed straight into a capture. Header values are written into `meta`
    as they are read, which is how a streaming reader sees them without a
    second pass over the file.

    Every error names `source:line`. A dataset error you have to bisect a file
    to locate is a dataset error you will fix by deleting examples.
    """
    meta = {} if meta is None else meta
    group = -1
    started = False
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith(COMMENT):
            continue

        example = _EXAMPLE.match(line)
        if example:
            indent, marker, _, text = example.groups()
            if indent and not started:
                raise DatasetError(
                    f"{source}:{number} is indented, which means it joins the example above it, "
                    "but it is the first example in the file"
                )
            if not indent:
                group += 1
            started = True
            unescaped = _unescape(text, source, number)
            if not unescaped:
                raise DatasetError(f"{source}:{number} has a label but no prompt")
            yield Example(text=unescaped, label=MARKERS[marker], group=group)
            continue

        header = _HEADER.match(line)
        if header:
            key, value = header.group(1), header.group(2).strip()
            if started:
                raise DatasetError(
                    f"{source}:{number} sets '{key}' after the first example; "
                    "the header describes the whole file and belongs at the top of it"
                )
            if key not in HEADER_KEYS:
                raise DatasetError(
                    f"{source}:{number} sets unknown header '{key}';"
                    f" known headers are {list(HEADER_KEYS)}"
                )
            if key in meta:
                # the second one would win silently, and the file would say two things
                raise DatasetError(f"{source}:{number} sets '{key}' twice; it was already set to {meta[key]!r}")
            meta[key] = value
            continue

        if _BARE_MARKER.match(line):
            raise DatasetError(f"{source}:{number} has a label but no prompt")
        if line.lstrip()[0] in MARKERS:
            raise DatasetError(f"{source}:{number} needs a space between the label and the prompt: {line!r}")
        raise DatasetError(
            f"{source}:{number} is not a comment, a 'key: value' header, or an example "
            f"starting with '+ ' or '- ': {line!r}"
        )

def _label_names(meta: Dict[str, str], source: str) -> Optional[tuple]:
    """Read the `labels:` header, which names label 0 and then label 1"""
    if "labels" not in meta:
        return None
    names = [name.strip() for name in meta["labels"].split(",")]
    if len(names) != 2 or not all(names):
        raise DatasetError(
            f"{source}: 'labels' names label 0 then label 1, so it takes exactly two "
            f"comma-separated names; got {meta['labels']!r}"
        )
    return tuple(names)

def parse(
    text: str,
    name: str = "unnamed",
    limit: Optional[int] = None,
    source: Optional[str] = None,
) -> LabeledPrompts:
    """Read a whole prompt set into a dataset

    `limit` is a floor rounded up to the end of a group rather than a hard
    cut: half a contrast pair is worse than one pair too many.
    """
    source = source or name
    if limit is not None and limit < 1:
        raise DatasetError(f"limit must be at least 1, got {limit}")
    meta: Dict[str, str] = {}
    examples: List[Example] = []
    for example in scan(text.splitlines(), source=source, meta=meta):
        if limit is not None and len(examples) >= limit and (not examples or example.group != examples[-1].group):
            break
        examples.append(example)
    if not examples:
        raise DatasetError(f"{source} has a header but no examples")

    grouped = len({example.group for example in examples}) != len(examples)
    dataset = LabeledPrompts(
        texts=[example.text for example in examples],
        labels=[example.label for example in examples],
        name=meta.get("name") or name,
        # a file where every group holds a single example is exactly an ungrouped one
        groups=[example.group for example in examples] if grouped else None,
    )
    names = _label_names(meta, source)
    return dataset if names is None else replace(dataset, label_names=names)

def load_prompts(path: str, limit: Optional[int] = None) -> LabeledPrompts:
    """Read a prompt set off disk, named by its `name:` header or else by its filename"""
    file = Path(path)
    if not file.exists():
        raise DatasetError(f"no dataset at {file}")
    return parse(file.read_text(), name=file.stem, limit=limit, source=str(file))

def dumps(data: LabeledPrompts) -> str:
    """Write a dataset back out as a prompt set

    What survives a round trip is what the dataset object holds: the name, the
    label names, the prompts and their groups. `notes:` and `source:` are for
    whoever reads the file, so a file rewritten from an object loses its prose
    -- which is the usual reason to keep hand-written sets hand-written and to
    let this function write converted ones.
    """
    marker = {value: key for key, value in MARKERS.items()}
    lines = [f"name: {data.name}", f"labels: {data.label_names[0]}, {data.label_names[1]}", ""]
    previous: Optional[int] = None
    for index, (text, label) in enumerate(zip(data.texts, data.labels, strict=True)):
        group = None if data.groups is None else data.groups[index]
        indent = "  " if group is not None and group == previous else ""
        if not indent and previous is not None and group is not None:
            lines.append("")
        lines.append(f"{indent}{marker[label]} {_escape(text)}")
        previous = group
    return "\n".join(lines) + "\n"

def save_prompts(data: LabeledPrompts, path: str) -> None:
    """Write a dataset to a .prompts file"""
    Path(path).write_text(dumps(data))

def load_labeled(
    path: str,
    text_field: str = "text",
    label_field: str = "label",
    limit: Optional[int] = None,
) -> LabeledPrompts:
    """Load a dataset by what the file is, so one --data flag takes either format

    The suffix decides, because the alternative is a flag naming the format
    and a second flag naming the file, which is two ways to say one thing and
    one of them will eventually be wrong.
    """
    suffix = Path(path).suffix.lower()
    if suffix in SUFFIXES:
        return load_prompts(path, limit=limit)
    if suffix == ".jsonl":
        return load_jsonl(path, text_field=text_field, label_field=label_field, limit=limit)
    raise DatasetError(
        f"cannot tell what kind of dataset '{path}' is from its suffix; "
        f"known suffixes are {sorted((*SUFFIXES, '.jsonl'))}"
    )
