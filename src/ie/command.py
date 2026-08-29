from dataclasses import dataclass
from typing import Dict, Optional

"""
What the user typed after the colon, and which view it means.

Ported from the shape k9s uses: a short alias resolves to a resource, an
optional /filter narrows the rows, and a handful of commands are handled
before any resource lookup happens. Keeping the parse in one place is what
lets the same string come from a keystroke, a startup argument or a drill-down
without three spellings of the same syntax.

Aliases are the reason a terminal UI is faster than a CLI: `:ar` is the whole
interaction, and it has to stay one keystroke away from `:artifacts` without a
second table saying which is which.

A common pipe could be: keystroke | parse | registry | PageStack.push
"""

# alias -> resource, the same two-letter shape k9s uses. A resource is always
# reachable by its full name too, so nothing here is the only way in.
ALIASES: Dict[str, str] = {
    "mo": "models", "m": "models",
    "ru": "runs", "r": "runs",
    "ar": "artifacts", "a": "artifacts",
    "la": "layers", "l": "layers",
    "to": "tokens", "t": "tokens",
    "ac": "activations",
    "no": "nodes", "n": "nodes",
    "te": "tensors",
    "h": "help", "?": "help",
    "q": "quit", "quit": "quit", "exit": "quit",
}

# commands that take the rest of the line as an argument rather than a filter
SETTERS = ("model", "prompt", "root")

@dataclass(frozen=True)
class Command:
    """One parsed instruction: which resource, narrowed how, or which setter"""
    resource: str
    filter: str = ""
    argument: str = ""

    @property
    def is_setter(self) -> bool:
        return self.resource in SETTERS

def parse(line: str) -> Optional[Command]:
    """Read a command line into the resource it names and the filter it carries

    Returns None for an empty line rather than raising, because pressing colon
    and then escape is a thing people do constantly and it is not an error.

    A setter takes the rest of the line verbatim -- a prompt has spaces and
    slashes in it, and splitting it on either would make the one command that
    needs a sentence the one command that cannot take one.
    """
    text = line.strip().lstrip(":").strip()
    if not text:
        return None

    head, _, rest = text.partition(" ")
    name = ALIASES.get(head.lower(), head.lower())
    if name in SETTERS:
        return Command(resource=name, argument=rest.strip())

    # /filter may be attached to the resource or given as the next word
    narrowed = ""
    if "/" in text:
        head_part, _, narrowed = text.partition("/")
        head = head_part.strip().split(" ")[0] if head_part.strip() else head
        name = ALIASES.get(head.lower(), head.lower())
    elif rest.strip():
        narrowed = rest.strip()
    return Command(resource=name, filter=narrowed.strip())
