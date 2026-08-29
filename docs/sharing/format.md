# The card, and the gates

## What an artifact holds

Everything on the card exists because leaving it out is a way a number gets misread
somewhere else. `Payload` is the one to look at twice: values, axes, units and tick labels
are one object, so there is no call that stores a tensor without saying what it means.

```mermaid
classDiagram
    class Artifact {
        +kind : str
        +id : str
        +method : str
        +metrics : dict
        +notes : str
        +validate()
        +save(path)
        +load(path)$
    }
    class ModelRef {
        +id : str
        +hf_name : str
        +revision : str
        +n_layers : int
        +d_model : int
        +n_heads : int
    }
    class Site {
        +layers : list~int~
        +fracs : list~float~
        +component : str
        +position : str
        +at(layers, n_layers)$
    }
    class Span {
        +metric : str
        +clean : float
        +corrupted : float
        +span : float
    }
    class Node {
        +id : str
        +component : str
        +layer : int
        +head : int
        +role : str
        +in_circuit : bool
        +scores : dict
    }
    class Edge {
        +source : str
        +target : str
        +kind : str
        +weight : float
    }
    class Payload {
        +values : Tensor
        +axes : list~str~
        +units : str
        +labels : dict
    }
    class Metric {
        +value : float
        +definition : str
        +units : str
    }
    class Controls {
        +cross_task : list~Control~
        +random_baseline : list~Control~
        +empty : bool
    }
    class Control {
        +name : str
        +metric : str
        +value : float
    }

    Artifact --> "1" ModelRef : model
    Artifact --> "1" Site : site
    Artifact --> "0..1" Span : span
    Artifact --> "*" Node : nodes
    Artifact --> "*" Edge : edges
    Artifact --> "*" Payload : tensors
    Artifact --> "*" Metric : metrics
    Artifact --> "1" Controls : controls
    Controls --> "*" Control
```

`Edge` is in the schema and every circuit this repository emits carries an empty list of
them. That is the honest state of the art rather than a gap: an artifact that says it found
no edges is different from one that left a reader guessing whether the tool looked.

`Metric` is the same idea as `Payload`, applied to a number rather than a grid. A value, what
it is in, and what was computed to get it, in one object — because `faithfulness` names a
logit-difference recovery here and a normalized KL reproduction elsewhere, and both land near
0.9.

`Controls` is written even when both its lists are empty, for the reason `Edge` is. A circuit
that was never ablated against another task and one that was must not be byte-identical, and
`Controls.empty` is the question a reader asks first.

`Node.scores` holds both halves of the circuit study — `attribution` and `causal` — because
they disagree and the disagreement is the finding. One summary score per component throws
away the result that the second measurement exists to produce.

## The kind decides what has to be there

```mermaid
flowchart LR
    K["kind"] --> C["circuit"]
    K --> P["probe"]
    K --> S["steering_vector"]
    K --> A["activation_map"]

    C --> CT["head_attribution<br/>head_effects<br/>+ span, required"]
    P --> PT["weight, bias<br/>mean, std"]
    S --> ST["vector"]
    A --> AT["values"]
```

An artifact missing these is not an incomplete artifact of that kind. It is a different
thing wearing the label, and `validate()` says so rather than letting a reader find out by
getting a `KeyError` in the middle of applying it.

## What refuses to leave the machine

Validation runs at `save`, not only at `load`, so a wrong artifact never ships. The solid
line is the way through; each dashed branch is the plausible-looking wrong number that gate
exists to stop.

```mermaid
flowchart LR
    S["Artifact.save"] --> K{"kind and<br/>version"}
    K --> L{"a fraction<br/>per layer"}
    L --> T{"the kind's<br/>tensors"}
    T --> SP{"span, if it<br/>reports recovery"}
    SP --> DF{"a definition<br/>per metric"}
    DF --> ID{"known structural<br/>assumptions"}
    ID --> AX{"an axis name<br/>per dimension"}
    AX --> LB{"labels as long<br/>as their axis"}
    LB --> SH{"widths against<br/>model and site"}
    SH --> W["write artifact.json<br/>+ tensors.safetensors"]

    K --- R1["guessing at the difference<br/>is worse than stopping"]
    L --- R2["bare indices cannot be<br/>placed in another model"]
    T --- R3["a different thing<br/>wearing the label"]
    SP --- R4["a percentage of<br/>an unstated whole"]
    DF --- R4b["two artifacts reporting<br/>0.9 for different things"]
    ID --- R4c["a constraint that does not<br/>break the equivalence class"]
    AX --- R5["a bare grid is one the<br/>next reader transposes"]
    LB --- R6["labels out of step<br/>name the wrong column"]
    SH --- R7["a subset of layers<br/>read as layer indices"]

    linkStyle 10,11,12,13,14,15,16,17,18 stroke-dasharray:4 4
```

The header is duplicated into the safetensors metadata, so a `tensors.safetensors` that got
separated from its card still says what it is.

## What ships empty on purpose

Three fields are written with nothing in them by every artifact this repository emits. None of
them is a placeholder — each is the recorded absence of a check, which is a different claim
from silence.

```mermaid
flowchart LR
    A["edges: []"] --> A1["the tool looked for wiring<br/>and measured none"]
    B["controls.cross_task: []"] --> B1["never ablated against another task,<br/>so this is a within-task claim"]
    C["identifiability: []"] --> C1["no structural assumption imposed,<br/>so this is a direction, not the direction"]

    A1 --> W["artifact check<br/>warns on each"]
    B1 --> W
    C1 --> W
```

An artifact that ran the check and one that never considered it would otherwise be the same
bytes, and the reader would have no way to tell which they were holding.

## Why the site is a fraction as well as an index

Layer 8 of a twelve-layer model and layer 42 of a sixty-four-layer one are the same place.
Only the fraction survives a model swap, and deriving one from the other needs a depth the
reader may not have resolved — so both are written down.

```mermaid
flowchart LR
    A["small model<br/>layer 8 of 12"] --> F["frac 0.667<br/>two thirds through"]
    B["large model<br/>layer 42 of 64"] --> F
    F --> U["a claim another lab<br/>can place"]

    C["layer 8"] --> N["placed nowhere<br/>without the depth"]
```

`Site.at` requires `n_layers` rather than defaulting it away, which is why packing a probe
made this repository teach `LinearProbe` to carry its own `n_layers`. The format asked a
question the code could not answer, and the code changed.
