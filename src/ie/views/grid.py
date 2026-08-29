from typing import List

from ..view import Detail, Hint, Row, View

"""
The numbers themselves, laid out the way the payload's axes say they are.

Everything else in the explorer is a list of things. This is the one view that
shows a measurement, and it exists because a circuit study's result is a
layer-by-head grid -- naming its shape and units in a table of metadata tells
you the grid is there without ever showing it to you.

The columns come from the payload's own axis labels when it has them, which is
the promise the format makes: a heatmap whose columns are named tokens can be
redrawn by a tool that never ran the model, and this is that tool. Without
labels the ticks are indices, which is honest rather than invented.

The `strongest` column is per row rather than per grid, because the question
asked of a head grid is almost always "which head in this layer" and finding
that by eye across twelve columns of signed floats is how you misread one.

A common pipe could be: :artifacts | enter | :tensors | enter
"""

class Grid(View):
    """One payload's values, as a table rather than as a shape"""

    hints = (Hint("enter", "this row, tick by tick"),)

    def __init__(self, app_ref, session, name: str, payload, site=None):
        super().__init__(app_ref, session)
        # the site, when there is one, is what turns a row index into a layer number.
        # A grid measured over a subset of layers has rows that are not layer indices,
        # which is the misreading the format's own shape check exists to prevent --
        # showing them as 0..n here would reintroduce it in the one view that shows
        # the numbers.
        self.site = site
        # not `self.name`: Widget.name is a read-only property and shadowing it raises
        self.tensor = name
        self.payload = payload
        self.title = f"grid {name}"
        self.columns = self._columns()

    def _columns(self) -> tuple:
        """The header row, from the axis labels when the payload carries them"""
        axes, shape = self.payload.axes, self.payload.values.shape
        if len(shape) == 0:
            return ("value", self.payload.units)
        if len(shape) == 1:
            return (axes[0] if axes else "index", self.payload.units)
        down, across = axes[0], axes[1]
        return (f"{down} \\ {across}", *self._ticks(across, shape[1]), "strongest")

    def _ticks(self, axis: str, width: int) -> List[str]:
        """Ticks along an axis: its labels, else the site's layers, else indices"""
        labels = self.payload.labels.get(axis)
        if labels:
            return [_short(label) for label in labels]
        if axis == "layer" and self.site is not None and len(self.site.layers) == width:
            return [f"L{layer}" for layer in self.site.layers]
        initial = axis[0] if axis else "c"
        return [f"{initial}{index}" for index in range(width)]

    def rows(self) -> List[Row]:
        values = self.payload.values
        if values.dim() == 0:
            return [Row(key="value", cells=("value", f"{float(values):+.6g}"))]
        if values.dim() == 1:
            ticks = self._ticks(self.payload.axes[0] if self.payload.axes else "", values.shape[0])
            return [
                Row(key=str(index), cells=(ticks[index], f"{float(values[index]):+.4f}"),
                    payload=(index, float(values[index])))
                for index in range(values.shape[0])
            ]
        if values.dim() > 2:
            raise ValueError(
                f"'{self.tensor}' has {values.dim()} axes {self.payload.axes}; this view lays out two, "
                "so slice it before looking at it"
            )

        down_ticks = self._ticks(self.payload.axes[0], values.shape[0])
        across_ticks = self._ticks(self.payload.axes[1], values.shape[1])
        found = []
        for index in range(values.shape[0]):
            line = values[index]
            peak = int(line.abs().argmax())
            found.append(Row(
                key=str(index),
                cells=(
                    down_ticks[index],
                    *[f"{float(cell):+.3f}" for cell in line],
                    f"{across_ticks[peak]} {float(line[peak]):+.3f}",
                ),
                payload=(index, line),
            ))
        return found

    def on_enter_row(self, row: Row) -> None:
        """One row of the grid as a record, which is readable when twelve columns are not"""
        if row.payload is None or self.payload.values.dim() < 2:
            self.explorer.flash("already at a single value")
            return
        index, line = row.payload
        across = self.payload.axes[1]
        ticks = self._ticks(across, line.shape[0])
        record = {
            f"{self.payload.axes[0]}": self._ticks(self.payload.axes[0], self.payload.values.shape[0])[index],
            "units": self.payload.units,
        }
        order = sorted(range(line.shape[0]), key=lambda where: -abs(float(line[where])))
        for rank, where in enumerate(order, start=1):
            record[f"{rank}. {across} {ticks[where]}"] = float(line[where])
        self.explorer.push(Detail(self.explorer, self.session, f"{self.title} row {index}", record))

def _short(label: str, width: int = 8) -> str:
    """A tick label narrow enough to be a column header

    Token strings are the common case and they carry their leading space, which
    matters for reading them and wastes a column of width -- so it is shown as a
    visible marker rather than dropped.
    """
    text = label.replace(" ", "_") if label.startswith(" ") else label
    return text if len(text) <= width else text[: width - 1] + "~"
