from typing import List

from ..view import Detail, Hint, Row, View

"""
The heads a circuit study kept, with what both halves of the study said.

The two score columns are the whole reason this view is not a single sorted
list: attribution is the direct path and is exact, patching is causal and
expensive, and where they disagree the disagreement is the result. A head with
a strongly negative attribution and a positive causal effect is not an error
bar -- on GPT-2 small the negative name movers write hard against the correct
answer and patching says the model needs them.

Sorted by the search's own step order rather than by either score, because
that is the order the circuit was built in and re-sorting hides which head was
found first.

A common pipe could be: :artifacts | enter | :nodes
"""

class Nodes(View):
    """The circuit's components, and every measurement made of each"""

    title = "nodes"
    columns = ("step", "node", "role", "layer", "head", "attribution", "causal", "minimality", "cumulative")
    hints = (Hint("enter", "node record"),)

    def rows(self) -> List[Row]:
        artifact = self._artifact()
        found = []
        for node in sorted(artifact.nodes, key=lambda item: item.scores.get("step", 0)):
            scores = node.scores
            found.append(Row(
                key=node.id,
                cells=(
                    _num(scores.get("step"), "{:.0f}"), node.id, node.role or "-",
                    node.layer, node.head if node.head is not None else "-",
                    _num(scores.get("attribution"), "{:+.3f}"),
                    _num(scores.get("causal"), "{:+.3f}"),
                    _num(scores.get("minimality"), "{:+.3f}"),
                    _num(scores.get("cumulative_recovery"), "{:.3f}"),
                ),
                payload=node,
            ))
        return found

    def _artifact(self):
        artifact = self.session.artifact
        if artifact is None:
            raise ValueError("no artifact selected -- open one from `:artifacts` first")
        if not artifact.nodes:
            raise ValueError(f"'{artifact.id}' is a {artifact.kind.value} and carries no graph nodes")
        return artifact

    def on_enter_row(self, row: Row) -> None:
        node = row.payload
        record = {
            "id": node.id, "component": node.component.value, "layer": node.layer,
            "head": node.head, "role": node.role or "-", "in_circuit": node.in_circuit,
        }
        record.update({f"score.{name}": value for name, value in node.scores.items()})
        record["reading"] = (
            "attribution is the direct path only and exact; causal is patching, through every path. "
            "Where they disagree, the disagreement is the finding."
        )
        self.explorer.push(Detail(self.explorer, self.session, f"node {node.id}", record))

def _num(value, spec: str) -> str:
    return "-" if value is None else spec.format(value)
