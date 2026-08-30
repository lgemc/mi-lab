from typing import ClassVar, Dict, List

from rich.text import Text

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

`v` swaps the numbers for a heat map. Reading 144 signed floats and finding
the shape in them is work a colour scale does in one glance, and the shape is
the finding here: two heads writing hard against the answer is the result the
whole circuit argument turns on. The numbers stay one keypress away, because a
block tells you where to look and never what the value was.

A common pipe could be: :artifacts | enter | :tensors | enter | v
"""

# Keyed by what a sign *means*, not by the colour it happens to be -- the same rule
# src/viz/style.py states, and the same two meanings its palette gives "positive" and
# "negative". They are restated rather than imported because that module pulls in
# matplotlib and seaborn at import time, and this explorer promises to open on a run
# without either.
HEAT = {"towards": (0x2A, 0x9D, 0x8F), "against": (0xE7, 0x6F, 0x51)}

# Magnitude as shading as well as colour, so the grid still reads on a monochrome
# terminal and for a reader who cannot separate the two hues.
BLOCKS = ("·", "░", "▒", "▓", "█")

# the dimmest a non-zero value is drawn: a small effect has to stay visibly different
# from no effect, which is the distinction this whole view exists to show
FLOOR = 0.4

def _shade(meaning: str, share: float) -> str:
    """A hex colour for one signed magnitude, dim for small and full for the peak"""
    weight = FLOOR + (1.0 - FLOOR) * min(1.0, share)
    red, green, blue = (int(channel * weight) for channel in HEAT[meaning])
    return f"#{red:02x}{green:02x}{blue:02x}"

def _block(value: float, limit: float) -> Text:
    """One cell of the heat map: sign as colour, magnitude as shading"""
    share = 0.0 if limit <= 0 else abs(value) / limit
    if share < 0.02:
        # not "blank": an unmeasured cell and a measured zero must not look alike
        return Text(" ·", style="#5a5f6a")
    level = min(len(BLOCKS) - 1, 1 + int(share * (len(BLOCKS) - 1)))
    meaning = "towards" if value > 0 else "against"
    return Text(BLOCKS[level] * 2, style=_shade(meaning, share))

class Grid(View):
    """One payload's values, as a table rather than as a shape"""

    hints = (Hint("enter", "this row, tick by tick"), Hint("v", "numbers / heat map"))
    keys: ClassVar[Dict[str, str]] = {"v": "toggle_heat"}

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
        # numbers first, heat on request: the title is what the page stack compares, so
        # a mode must not change it, and a view that opened in a mode you did not ask
        # for is one you have to undo before you can read it
        self.heat = False
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

    def toggle_heat(self) -> None:
        """`v`: swap signed floats for a colour scale, and back"""
        if self.payload.values.dim() == 0:
            self.explorer.flash("a single value has no shape to draw")
            return
        self.heat = not self.heat
        self.reload()

    def _limit(self) -> float:
        """The magnitude the scale saturates at, symmetric so zero stays the middle

        One limit for the whole grid, never per row: a per-row scale draws the
        strongest head of a layer that does nothing the same as the strongest
        head overall, which is the misreading a heat map is supposed to remove.
        """
        values = self.payload.values
        return max(float(values.abs().max()), 1e-9)

    def note(self) -> str:
        if not self.heat:
            return ""
        limit = self._limit()
        return (f"{BLOCKS[-1] * 2} towards  {BLOCKS[-1] * 2} against  "
                f"·  under 2%   scale ±{limit:.4g} {self.payload.units}   <v> numbers")

    def rows(self) -> List[Row]:
        values = self.payload.values
        if values.dim() == 0:
            return [Row(key="value", cells=("value", f"{float(values):+.6g}"))]
        if values.dim() == 1:
            ticks = self._ticks(self.payload.axes[0] if self.payload.axes else "", values.shape[0])
            limit = self._limit()
            return [
                Row(key=str(index), cells=(ticks[index], self._value(float(values[index]), limit)),
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
        limit = self._limit()
        found = []
        for index in range(values.shape[0]):
            line = values[index]
            peak = int(line.abs().argmax())
            cells = ([_block(float(cell), limit) for cell in line] if self.heat
                     else [f"{float(cell):+.3f}" for cell in line])
            found.append(Row(
                key=str(index),
                cells=(
                    down_ticks[index],
                    *cells,
                    # the peak keeps its number in both modes: a block says where to
                    # look and a reader still has to be able to say how much
                    f"{across_ticks[peak]} {float(line[peak]):+.3f}",
                ),
                payload=(index, line),
            ))
        return found

    def _value(self, value: float, limit: float):
        """One 1-D cell: the number, or the block and the number beside it

        A one-axis payload has a column to spare, so heat mode keeps the value
        here rather than making the reader toggle back for it.
        """
        if not self.heat:
            return f"{value:+.4f}"
        cell = _block(value, limit)
        cell.append(f"  {value:+.4f}", style="")
        return cell

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
