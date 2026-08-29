import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file, save_file

from .schema import FORMAT, VERSION, Artifact, ArtifactError, Edge, ModelRef, Node, Payload, Site, Span

"""
An artifact on disk, and back off it again.

An artifact is a directory: `artifact.json` is the card and every number lives
in `tensors.safetensors` beside it. The card is stdlib-readable JSON, so
deciding whether an artifact is worth downloading never requires torch, and
the tensors are memory-mappable, so a large one need not be read whole.

Reading and writing live here rather than on the Artifact itself so that the
schema stays a description of a result and not of a filesystem. What the two
halves owe each other is the round trip: what `save` writes, `load` has to
give back as the same object, and `from_manifest` refuses a card whose tensor
names disagree with the file rather than handing back a partial one.

Validation runs on the way out as well as on the way in, so an artifact that
is wrong never leaves the machine that made it.

A common pipe could be: from_circuit | save | load | to_probe
"""

MANIFEST = "artifact.json"
TENSORS = "tensors.safetensors"
SUFFIX = ".mia"

def to_manifest(artifact: Artifact) -> Dict[str, Any]:
    """The card, as the plain data that gets written to artifact.json"""
    return {
        "format": FORMAT,
        "version": artifact.version,
        "kind": artifact.kind,
        "id": artifact.id,
        "created_at": artifact.created_at,
        "model": asdict(artifact.model),
        "site": asdict(artifact.site),
        "task": artifact.task,
        "measurement": {
            "method": artifact.method,
            "span": asdict(artifact.span) if artifact.span is not None else None,
            "metrics": artifact.metrics,
        },
        "graph": {
            "nodes": [asdict(node) for node in artifact.nodes],
            "edges": [asdict(edge) for edge in artifact.edges],
        },
        "tensors": {name: payload.describe() for name, payload in artifact.tensors.items()},
        "provenance": artifact.provenance,
        "notes": artifact.notes,
    }

def from_manifest(manifest: Dict[str, Any], tensors: Dict[str, torch.Tensor]) -> Artifact:
    """Rebuild an artifact from its card and the tensors that were stored beside it"""
    unknown = set(manifest) - {
        "format", "version", "kind", "id", "created_at", "model", "site",
        "task", "measurement", "graph", "tensors", "provenance", "notes",
    }
    if unknown:
        raise ArtifactError(
            f"artifact.json has unknown keys {sorted(unknown)}; a key this reader does not know is a "
            "claim it would silently drop"
        )
    if manifest.get("format") != FORMAT:
        raise ArtifactError(f"not a {FORMAT} artifact: format is {manifest.get('format')!r}")

    measurement = manifest.get("measurement") or {}
    graph = manifest.get("graph") or {}
    described = manifest.get("tensors") or {}

    stored, described_names = set(tensors), set(described)
    if stored != described_names:
        raise ArtifactError(
            f"the card and the tensor file disagree: card-only {sorted(described_names - stored)}, "
            f"file-only {sorted(stored - described_names)}"
        )

    span = measurement.get("span")
    return Artifact(
        kind=manifest["kind"],
        id=manifest["id"],
        model=ModelRef(**manifest["model"]),
        site=Site(**manifest["site"]),
        task=manifest.get("task") or {},
        method=measurement.get("method", "unspecified"),
        metrics=measurement.get("metrics") or {},
        span=Span(**span) if span else None,
        nodes=[Node(**node) for node in graph.get("nodes", [])],
        edges=[Edge(**edge) for edge in graph.get("edges", [])],
        tensors={
            name: Payload(
                values=tensors[name],
                axes=described[name]["axes"],
                units=described[name]["units"],
                labels=described[name].get("labels") or {},
            )
            for name in described
        },
        provenance=manifest.get("provenance") or {},
        notes=manifest.get("notes", ""),
        created_at=manifest.get("created_at", ""),
        version=manifest.get("version", VERSION),
    )

def save(artifact: Artifact, path: str) -> str:
    """Write the artifact as a directory: the card, and the tensors beside it

    Validation happens here rather than at read time as well, so an
    artifact that is wrong never leaves the machine that made it.
    """
    artifact.validate()
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / MANIFEST).write_text(json.dumps(to_manifest(artifact), indent=2, sort_keys=True))
    # the header is duplicated into the tensor file's own metadata so a
    # tensors.safetensors that got separated from its card still says what it is
    save_file(
        {name: payload.values.contiguous() for name, payload in artifact.tensors.items()},
        target / TENSORS,
        metadata={"format": FORMAT, "version": artifact.version, "kind": artifact.kind, "id": artifact.id},
    )
    return str(target)

def load(path: str) -> Artifact:
    """Read an artifact back from its directory, checking the card against the tensors"""
    source = Path(path)
    card, payload = source / MANIFEST, source / TENSORS
    if not card.exists():
        raise ArtifactError(f"no {MANIFEST} in {source}; a {FORMAT} artifact is a directory containing one")
    if not payload.exists():
        raise ArtifactError(f"{source} has a card but no {TENSORS}; its numbers are missing")
    try:
        manifest = json.loads(card.read_text())
    except json.JSONDecodeError as error:
        raise ArtifactError(f"{card} is not valid JSON") from error
    try:
        return from_manifest(manifest, load_file(payload)).validate()
    except TypeError as error:
        raise ArtifactError(f"{card} does not describe a {FORMAT} v{VERSION} artifact: {error}") from error

def find_artifacts(root: str) -> List[Artifact]:
    """Every artifact under a directory, skipping what does not read as one

    A directory that fails to load is skipped rather than fatal, the same way
    a half-written run does not make a listing unusable.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found = []
    for directory in sorted(base.rglob(f"*{SUFFIX}")):
        try:
            found.append(load(str(directory)))
        except ArtifactError:
            continue
    return found
