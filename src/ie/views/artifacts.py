from pathlib import Path
from typing import List

from ...share import storage
from ..view import Detail, Hint, Row, View

"""
The results that are meant to leave the machine, read without loading one.

An artifact card is JSON and nothing here opens a checkpoint, which is what
makes this the view to run over a directory of downloads: kind, model, site
and what the thing claims are all answerable before deciding to trust it.

The warnings column is `artifact check` inlined, because the whole reason
those checks exist is that nobody runs them as a separate step. A circuit with
no cross-task ablation and a probe with no recorded commit look fine until
something says so on the same line as the result.

A common pipe could be: :artifacts | enter | :nodes
"""

class Artifacts(View):
    """Every .mia under the session root, and what a reader could not trust about it"""

    title = "artifacts"
    columns = ("kind", "id", "model", "site", "size", "flags")
    hints = (Hint("enter", "card"), Hint("n", "nodes"), Hint("t", "tensors"))

    def rows(self) -> List[Row]:
        found = []
        for artifact in self._found():
            site = artifact.site
            where = f"{site.component.value} x{len(site.layers)}L" if site.layers else "-"
            found.append(Row(
                key=artifact.id,
                cells=(
                    artifact.kind.value, artifact.id, artifact.model.id, where,
                    f"{artifact.n_bytes / 1024:.1f}K", " ".join(_flags(artifact)),
                ),
                payload=artifact,
            ))
        return found

    def _found(self):
        """Artifacts under the root, plus any sitting in the working directory"""
        seen = storage.find_artifacts(self.session.root)
        here = [path for path in Path().glob(f"*{storage.SUFFIX}") if path.is_dir()]
        for path in here:
            try:
                seen.append(storage.load(str(path)))
            except Exception:
                continue
        return seen

    def _empty_note(self) -> str:
        if self.filter:
            return super()._empty_note()
        return (
            f"no {storage.SUFFIX} artifacts under '{self.session.root}' or here -- "
            "an ioi_circuit run writes one, or pack a probe with `artifact pack`"
        )

    def on_enter_row(self, row: Row) -> None:
        artifact = row.payload
        self.session.artifact = artifact
        self.explorer.push(Detail(self.explorer, self.session, f"card {artifact.id}", _card(artifact)))

    def detail(self, row: Row):
        return None if row.payload is None else _card(row.payload)

def _flags(artifact) -> List[str]:
    """The `artifact check` warnings, as short marks that fit in a column"""
    marks = []
    if artifact.provenance.get("git_dirty"):
        marks.append("dirty")
    if not artifact.provenance.get("git_commit"):
        marks.append("no-commit")
    if artifact.span is not None and abs(artifact.span.span) < 1e-3:
        marks.append("flat-span")
    if artifact.kind == "circuit" and not artifact.edges:
        marks.append("no-edges")
    if artifact.kind == "circuit" and not artifact.controls.cross_task:
        marks.append("within-task")
    if artifact.kind == "steering_vector" and not artifact.identifiability:
        marks.append("unidentified")
    if artifact.model.n_layers is None or artifact.model.d_model is None:
        marks.append("no-sizes")
    return marks or ["-"]

def _card(artifact) -> dict:
    """The card as a flat record, in the order the format writes it"""
    record = {
        "kind": artifact.kind.value, "id": artifact.id, "version": artifact.version,
        "created_at": artifact.created_at, "model": artifact.model.id,
        "hf_name": artifact.model.hf_name, "sizes": (
            f"{artifact.model.n_layers}L x {artifact.model.d_model}d x {artifact.model.n_heads}h"
        ),
        "site.component": artifact.site.component.value, "site.position": artifact.site.position.value,
        "site.layers": artifact.site.layers, "site.fracs": artifact.site.fracs,
        "method": artifact.method,
    }
    if artifact.span is not None:
        record["span"] = (
            f"{artifact.span.metric}  clean {artifact.span.clean:+.4f}  "
            f"corrupted {artifact.span.corrupted:+.4f}  span {artifact.span.span:+.4f}"
        )
    for name, metric in artifact.metrics.items():
        record[f"metric.{name}"] = f"{metric.value:+.6g} [{metric.units}] -- {metric.definition}"
    record["identifiability"] = artifact.identifiability or "[] none imposed"
    record["controls.cross_task"] = artifact.controls.cross_task or "[] never run"
    record["controls.random_baseline"] = artifact.controls.random_baseline or "[] never run"
    record["nodes"] = f"{len(artifact.nodes)} ({len(artifact.circuit_heads)} in circuit)"
    record["edges"] = artifact.edges or "[] none measured"
    record["tensors"] = list(artifact.tensors)
    for name, value in artifact.task.items():
        record[f"task.{name}"] = value
    for name, value in artifact.provenance.items():
        record[f"provenance.{name}"] = value
    if artifact.notes:
        record["notes"] = artifact.notes
    return record
