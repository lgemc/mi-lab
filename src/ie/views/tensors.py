from typing import List

from ..view import Hint, Row, View

"""
What an artifact actually carries, with the axes that say what it means.

A tensor without its axes is one the next reader transposes, so this view puts
the axis names and the unit on the same line as the shape rather than making
them something you go and look up. The `labels` column says whether the ticks
along an axis are named -- which is the difference between a heatmap somebody
else can redraw and a grid of numbers.

A common pipe could be: :artifacts | enter | :tensors
"""

class Tensors(View):
    """The payloads under an artifact, and what their axes mean"""

    title = "tensors"
    columns = ("name", "shape", "axes", "units", "dtype", "labels")
    hints = (Hint("enter", "the numbers"), Hint("y", "shape and axes"))

    def rows(self) -> List[Row]:
        artifact = self.session.artifact
        if artifact is None:
            raise ValueError("no artifact selected -- open one from `:artifacts` first")
        found = []
        for name, payload in artifact.tensors.items():
            described = payload.describe()
            shape = " x ".join(
                f"{axis}={size}" for axis, size in zip(payload.axes, payload.values.shape, strict=True)
            ) or "scalar"
            labelled = ", ".join(f"{axis}({len(names)})" for axis, names in payload.labels.items()) or "-"
            found.append(Row(
                key=name,
                cells=(name, shape, ", ".join(payload.axes) or "-", payload.units,
                       described["dtype"], labelled),
                payload=(name, payload),
            ))
        return found

    def on_enter_row(self, row: Row) -> None:
        """Show the values, not another description of them

        Enter used to open a record of the shape and the dtype, which is the
        thing you already read off this table. The numbers are what an artifact
        is for, and `y` still gives the metadata.
        """
        from .grid import Grid

        name, payload = row.payload
        site = self.session.artifact.site if self.session.artifact is not None else None
        self.explorer.push(Grid(self.explorer, self.session, name, payload, site=site))

    def detail(self, row: Row):
        name, payload = row.payload
        values = payload.values
        record = {
            "name": name, "shape": list(values.shape), "dtype": str(values.dtype).removeprefix("torch."),
            "axes": payload.axes, "units": payload.units,
        }
        for axis, names in payload.labels.items():
            record[f"labels.{axis}"] = names
        if values.numel():
            record["min"] = float(values.min())
            record["max"] = float(values.max())
            record["mean"] = float(values.float().mean())
            record["strongest"] = _strongest(payload)
        return record

def _strongest(payload) -> str:
    """Where the largest magnitude sits, named by the axes rather than by index

    A grid's argmax as a flat integer is the least useful form of the most
    useful fact about it.
    """
    values = payload.values
    if not values.numel() or not payload.axes:
        return "-"
    flat = int(values.abs().argmax())
    where = []
    for axis, size in zip(reversed(payload.axes), reversed(values.shape), strict=True):
        index = flat % size
        flat //= size
        ticks = payload.labels.get(axis)
        where.append(f"{axis}={ticks[index]!r}" if ticks else f"{axis}={index}")
    return "  ".join(reversed(where)) + f"  value {float(values.flatten()[int(values.abs().argmax())]):+.4f}"
