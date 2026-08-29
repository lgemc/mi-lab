# Sharing, drawn

Diagrams for the `.mia` envelope and the converters that fill it. The prose is
[`../artifact-format.md`](../artifact-format.md) and the field-by-field specification is
[`../rfcs/0001-mia-format.md`](../rfcs/0001-mia-format.md); these pages are the same thing as
pictures, for reading before either rather than instead of them.

- [format.md](format.md) — what a card holds, and what `validate()` refuses to let out
- [landscape.md](landscape.md) — what already exists for this elsewhere, and where it stops

## The round trip

`share/schema.py` and `share/storage.py` are the envelope and know nothing about this
repository. `share/converters/` is where the two sides meet, which is what keeps
`storage.load` from importing transformers.

```mermaid
flowchart LR
    subgraph measured["what this lab measured"]
        A["Attribution<br/>direct logit attribution"]
        E["HeadEffects<br/>activation patching"]
        R["CircuitReport<br/>greedy search, verify"]
        P["LinearProbe"]
        V["direction + strength sweep"]
        M["captured activations"]
    end

    subgraph converters["share/converters/"]
        FC["circuit.from_circuit"]
        FP["probe.from_probe"]
        FS["steering.from_steering"]
        FA["activations.from_activations"]
    end

    OUT["Artifact<br/>share/schema.py"]

    subgraph dir["name.mia — a directory"]
        CARD["artifact.json<br/>stdlib JSON, no torch"]
        TENS["tensors.safetensors<br/>memory-mappable"]
    end

    IN["Artifact"]
    BACK["LinearProbe<br/>scores activations again"]

    A --> FC
    E --> FC
    R --> FC
    P --> FP
    V --> FS
    M --> FA

    FC --> OUT
    FP --> OUT
    FS --> OUT
    FA --> OUT

    OUT -->|"validate, storage.save"| CARD
    OUT --> TENS
    CARD -->|"storage.load"| IN
    TENS --> IN
    IN -->|"probe.to_probe"| BACK
```

Four doors in and one door back is the honest shape today. Every converter is lossy on
purpose — it writes down what another lab needs to *use* the result and not the intermediate
state that only means something inside this process — but `probe` is the kind that has been
carried all the way back to a working object, and that round trip is the test of the format.
A shared artifact that does not come back as something you can call has shared nothing.

## Where the boundary sits

The import direction is the whole design. `schema.py` and `storage.py` sit under `converters/`
and never learn what an `IOIDataset` is, so a reader with neither transformers nor this
repository installed can still open a card.

```mermaid
flowchart TD
    CLI["cli, experiment/runner"] --> LOAD
    CLI --> CONV
    LOAD["share/loaders.py<br/>one door, either form"] --> CONV
    CONV["share/converters/<br/>knows both sides"] --> ART
    CONV --> METH["methods: circuits, probing"]
    CONV --> DATA["data/ioi.py"]
    CONV --> CORE["core/config.py"]
    ART["share/schema.py + storage.py<br/>the envelope"] --> STD["json, safetensors, torch"]
    ART --> PROV["share/provenance.py<br/>git, torch version"]

    READER["another lab's tool"] --> ART
```

`another lab's tool` reaching `schema.py` without passing through anything else is the
property being protected. Add a converter for a new experiment kind as a module under
`converters/`; adding a special case inside `schema.py` moves that line.
