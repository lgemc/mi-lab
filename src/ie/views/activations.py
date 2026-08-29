from typing import List

from ...core.config import Position
from ..view import Hint, Row, View

"""
What the residual stream is carrying at every layer, for the current prompt.

One row per layer, measured at the last position, because that is the one the
model predicts from and the one a probe is fit at. The norm column is the
number that says where a model puts its mass -- it grows with depth on every
transformer anyone has looked at, and a capture that does not is a capture
that went wrong.

Given a layer as an argument this narrows to that layer and reports per
position instead, which is the drill-down from the layer list: the same
quantity, cut the other way.

A common pipe could be: :layers | enter | :tokens
"""

class Activations(View):
    """Residual stream norms, per layer or per position within one layer"""

    title = "activations"
    columns = ("layer", "depth", "norm", "mean", "std", "max |x|")
    needs_model = True
    hints = (Hint("enter", "positions in this layer"),)

    def __init__(self, app_ref, session, argument: str = ""):
        super().__init__(app_ref, session, argument=argument)
        # named at construction rather than on mount, because the page stack compares
        # titles when it is pushed -- and a mount happens after that, so a child that
        # renamed itself later silently replaced the view it was drilled from
        if self.argument:
            self.title = f"activations L{self.argument}"
            self.columns = ("position", "token", "norm", "mean", "std", "max |x|")

    def rows(self) -> List[Row]:
        adapter = self.session.adapter()
        cfg = adapter.cfg
        if self.argument:
            return self._by_position(adapter, int(self.argument))

        # every layer, named explicitly: capture defaults to the config's probe layer
        # alone, so asking for "the activations" without saying so gets exactly one row
        layers = list(range(cfg.n_layers))
        captured = self.session.cached(
            "activations",
            lambda: adapter.capture([self.session.prompt], layers=layers, position=Position.LAST),
        )
        values = captured[0]
        return [
            Row(
                key=str(layer),
                cells=(layer, f"{layer / cfg.n_layers:.3f}", *_stats(values[index])),
                payload=layer,
            )
            for index, layer in enumerate(layers)
        ]

    def _by_position(self, adapter, layer: int) -> List[Row]:
        """One layer, cut per token position rather than per layer"""
        from ...model.adapter import require_circuits

        captured = self.session.cached(
            f"activations.all.{layer}",
            lambda: adapter.capture([self.session.prompt], layers=[layer], position=Position.ALL),
        )
        values = captured[0, 0]
        try:
            pieces = self.session.cached("tokens", lambda: require_circuits(adapter).tokens(self.session.prompt))
        except Exception:
            pieces = ["?"] * values.shape[0]
        return [
            Row(
                key=str(index),
                cells=(index, pieces[index] if index < len(pieces) else "?", *_stats(values[index])),
            )
            for index in range(values.shape[0])
        ]

    def on_enter_row(self, row: Row) -> None:
        if self.argument or row.payload is None:
            self.explorer.flash("already at position level")
            return
        self.explorer.push(Activations(self.explorer, self.session, argument=str(row.payload)))

def _stats(vector) -> tuple:
    """The four numbers that say what a residual-stream vector looks like"""
    return (
        f"{float(vector.norm()):.2f}",
        f"{float(vector.mean()):+.4f}",
        f"{float(vector.std()):.4f}",
        f"{float(vector.abs().max()):.2f}",
    )
