from enum import Enum
from typing import Dict, Tuple

from .errors import ArtifactError

"""
The closed sets, in one place because they are what validate() checks against.

Everything else in this package describes a shape. This module is the names a
card is allowed to use, and each set is closed on purpose: an unrecognized
entry is a claim the reader would otherwise take on trust.

These are declared here rather than imported from `core` even where `core` has
the same names. The envelope pulls in nothing but json, safetensors and torch,
so a reader without this repository can still open a card -- and the format's
vocabulary is a published contract with a version attached. If a value is added
to the repository's own enum, the format should not silently start emitting a
name older readers cannot parse; it should need a deliberate version bump. The
copy is the version boundary.

Every enum subclasses `str` so that a member is its own wire format: json.dumps
writes "head_out", asdict leaves it alone, and comparison against a plain
string from a loaded card is true. `StrEnum` would be tidier and is not used on
purpose -- it changes what `format()` produces, and this repository already
made that decision once (see the UP042 ignore in pyproject.toml).

The trap that comes with `(str, Enum)`: `f"{Component.HEAD_OUT}"` is
"Component.HEAD_OUT", not "head_out". Interpolate `.value`, or let json do it.

A common pipe could be: one_of | Artifact | validate
"""

class Kind(str, Enum):
    """What an artifact is, which decides what it has to carry

    Adding one is a version bump and an entry in REQUIRED: a kind whose
    required tensors nobody wrote down is a kind that validates vacuously.
    """
    CIRCUIT = "circuit"
    PROBE = "probe"
    STEERING_VECTOR = "steering_vector"
    ACTIVATION_MAP = "activation_map"


class Component(str, Enum):
    """Where in a block a measurement was read from or written into

    This is a *site*, not a part of a graph -- see NodeComponent, which is the
    other vocabulary and is deliberately disjoint from this one. RESIDUAL is
    the stream leaving a block, HEAD_OUT the input to the attention output
    projection, MLP_OUT the feed-forward write, ATTENTION the pattern itself.
    """
    RESIDUAL = "residual"
    HEAD_OUT = "head_out"
    MLP_OUT = "mlp_out"
    ATTENTION = "attention"


class NodeComponent(str, Enum):
    """What kind of thing a node in a circuit graph is

    Disjoint from Component on purpose, and the reason both fields are still
    called `component` on the wire: a site says where a value was read from,
    a node says what part of the model it stands for. A node claiming
    "head_out" or a site claiming "head" is a category error, and before these
    were separate types nothing caught it.
    """
    HEAD = "head"
    MLP = "mlp"


class Position(str, Enum):
    """Which token position(s) the measurement covers

    LAST is the model's decision state, MEAN averages over real (non-padding)
    tokens, ALL keeps the whole sequence -- which is what a position map needs
    and what makes a payload seq_len times larger.
    """
    LAST = "last"
    MEAN = "mean"
    ALL = "all"


class Assumption(str, Enum):
    """A structural assumption that picks one solution out of an equivalence class

    Steering directions are generically non-identifiable: an infinite class of
    geometrically distinct directions produces identical behaviour, and the
    bound on the components in the null space is infinite, so more data cannot
    narrow it. These are the assumptions that break the gauge. An artifact
    names which it imposed, and an empty list -- the common case -- is the
    honest answer said out loud.
    """
    INDEPENDENCE = "independence"
    SPARSITY = "sparsity"
    MULTI_ENVIRONMENT = "multi_environment"
    CROSS_LAYER = "cross_layer"


# What each kind has to carry to be that kind. An artifact missing these is not
# an incomplete artifact, it is a different thing wearing the label.
REQUIRED: Dict[Kind, Tuple[str, ...]] = {
    Kind.CIRCUIT: ("head_attribution", "head_effects"),
    Kind.PROBE: ("weight", "bias", "mean", "std"),
    Kind.STEERING_VECTOR: ("vector",),
    Kind.ACTIVATION_MAP: ("values",),
}

# Kinds whose numbers are a fraction of a corruption's span, and so cannot be
# read at all without it.
NEEDS_SPAN: Tuple[Kind, ...] = (Kind.CIRCUIT,)

def names(enum) -> list:
    """Every value an enum accepts, sorted, for a message that has to list them"""
    return sorted(member.value for member in enum)

def one_of(enum, value, field: str):
    """Coerce a card's string to its member, or say what the field accepts

    Enum's own ValueError names the class and the bad value and stops there.
    Every message in this package says what to do instead, so this one lists
    the vocabulary rather than leaving the reader to find it in the source.

    A member passed in is returned unchanged, so this is safe to call on a
    field that has already been coerced.
    """
    try:
        return enum(value)
    except ValueError as error:
        raise ArtifactError(
            f"{field} is {value!r}, which is not one of {names(enum)}"
        ) from error
