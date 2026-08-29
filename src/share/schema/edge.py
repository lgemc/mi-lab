from dataclasses import dataclass

"""
A measured dependency between two nodes.

Nothing in this repository measures one. The type exists anyway so that a
circuit can say it found none, which is a different claim from a card that
never mentioned wiring: an artifact omitting the field reads as one whose
connections nobody thought to record.

A common pipe could be: path_patching | Edge | Artifact.edges
"""

@dataclass(frozen=True)
class Edge:
    """A measured dependency between two nodes, as node ids and what was measured

    Nothing in this repository measures one yet, so circuits it emits carry an
    empty list. That is the honest state of the art rather than a gap in the
    schema: an artifact says it found no edges, instead of leaving a reader to
    guess whether the tool looked.

    `kind` says what was measured to establish the dependency, and is prose
    rather than an enum precisely because the field has not agreed on the
    answer -- closing the set now would freeze one lab's method into the
    format. The value this repository writes is "unmeasured", the default and
    a sentinel. A tool that does measure edges would write what it ran:
    "path_patching", "attribution_graph", "transcoder_attribution".

    `weight` is on whatever scale `kind` implies, which is a gap: an edge has
    no `units` field yet. Until something here measures one, adding it would
    be guessing at the shape.
    """
    source: str
    target: str
    kind: str = "unmeasured"
    weight: float = 0.0
