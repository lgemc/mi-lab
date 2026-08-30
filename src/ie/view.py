from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import DataTable, Static

"""
What every resource view is, so that adding one is a class rather than a
screen.

A view answers three questions and nothing else: what its columns are, what
its rows are right now, and what pressing Enter on a row means. Everything
else -- the table, the filter, the empty state, the error line -- is here once,
because a view that draws its own table is a view that will disagree with the
next one about what a selected row looks like.

This is k9s's ResourceViewer split down to what a single-process explorer
needs: no watchers, no informers, just a `rows()` a view recomputes when it is
shown or refreshed.

`needs_model` is the load-bearing flag. A view that sets it says it cannot
answer without the weights, and the app loads the checkpoint before showing
it; a view that leaves it False must never touch `session.adapter()`, or the
explorer stops opening instantly on things that need no model at all.

A common pipe could be: registry | View | rows | on_enter
"""

@dataclass
class Row:
    """One line in a resource table, and the object it stands for

    `payload` is what a drill-down receives. Keeping it beside the cells rather
    than re-finding the object by its key is what stops a view from having to
    parse its own rendering back into data.
    """
    key: str
    cells: Sequence[str]
    payload: Any = None
    tone: str = ""

@dataclass
class Hint:
    """One key this view answers to, for the footer"""
    key: str
    description: str

class View(Vertical):
    """A table of one kind of thing, with a drill-down and a filter"""

    title = "view"
    columns: Sequence[str] = ()
    needs_model = False
    hints: Sequence[Hint] = ()

    # key -> method on this view. Declared rather than bound on the app, because a
    # key that means something here and nothing elsewhere belongs to the view that
    # answers it -- and because the footer hints were advertising keys that no code
    # implemented, which is worse than not offering them.
    keys: ClassVar[Dict[str, str]] = {}

    DEFAULT_CSS = """
    View { height: 1fr; }
    View > DataTable { height: 1fr; }
    View > .ie-note { color: $text-muted; padding: 0 1; height: auto; }
    """

    def __init__(self, app_ref, session, argument: str = ""):
        super().__init__()
        self.explorer = app_ref
        self.session = session
        self.argument = argument
        self.filter = ""
        self._error = ""
        self._rows: List[Row] = []
        self._note = Static("", classes="ie-note")
        self._table = DataTable(cursor_type="row", zebra_stripes=True)

    # ------------------------------------------------------------------ shape

    def rows(self) -> List[Row]:
        """Everything this view currently has to show. Recomputed, never cached here."""
        raise NotImplementedError

    def on_enter_row(self, row: Row) -> None:
        """What drilling into a row means. Default: nothing to drill into."""
        self.explorer.flash(f"{row.key} has nothing under it")

    def detail(self, row: Row) -> Optional[Dict[str, Any]]:
        """The record behind a row, shown by `y`. None means there is none."""
        return None

    # ------------------------------------------------------------------ mount

    def compose(self):
        yield self._note
        yield self._table

    def on_mount(self) -> None:
        # a view is mounted once and hidden when covered, but a remount is cheap to
        # survive and a DataTable refuses a column key it already has
        if not self._table.columns:
            # the position is what makes a column key unique, not the header text -- the
            # same rule the rows follow below, and for a stronger reason: a grid indexed
            # by token position has one column per position, and the IOI task repeats a
            # name on purpose, so ' Sam' is two different columns. A DataTable refuses a
            # duplicate key by raising, which took the whole app down on the one grid the
            # circuit study exists to show. Text, so a token holding '[' is not markup.
            for index, column in enumerate(self.columns):
                self._table.add_column(Text(cell_text(column)), key=f"{index}:{column}")
        self.reload()

    # ----------------------------------------------------------------- filling

    def reload(self) -> None:
        """Recompute the rows and redraw, keeping the cursor where it can be kept

        An explorer that jumps to the top on every refresh is one you cannot
        read a long table in.
        """
        cursor = self._table.cursor_row
        try:
            self._rows = list(self.rows())
            self._error = ""
        except Exception as error:  # a view must report a failure, never take the app down
            self._rows = []
            self._error = f"[!] {error}"

        shown = [row for row in self._rows if self._matches(row)]
        self._table.clear()
        for index, row in enumerate(shown):
            # the position is what makes the table key unique, not the row's own key: an
            # artifact id is derived from its dataset and model, so two runs of one
            # experiment produce the same id, and a DataTable refuses a duplicate key by
            # raising. Selection is by cursor position anyway, so nothing here needs the
            # row's identity -- and no view should be able to take the app down by having
            # two of something.
            # Text, not str: a bare string is parsed as console markup, and the cells that
            # matter most here are exactly the bracketed ones -- `[!]` on a row that would
            # not read, and every list value a Detail renders as `[a, b]`. Those came out
            # blank, so a run that shipped an artifact looked like a run that shipped none
            self._table.add_row(*[_cell(cell) for cell in row.cells], key=f"{index}:{row.key}")
        self.set_note(self._error or (self.note() if shown else self._empty_note()))
        if shown and cursor is not None:
            self._table.move_cursor(row=min(cursor, len(shown) - 1))
        self.explorer.refresh_chrome()

    def note(self) -> str:
        """A line kept above a table that has rows -- a legend, a scale, a caveat

        Empty for most views: a note that is always there is one nobody reads.
        """
        return ""

    def _empty_note(self) -> str:
        """What to say when there is nothing, which is a finding and not a blank screen"""
        if self.filter:
            return f"nothing matches '{self.filter}' among {len(self._rows)} rows"
        return "nothing here"

    def _matches(self, row: Row) -> bool:
        if not self.filter:
            return True
        needle = self.filter.lower()
        return any(needle in cell_text(cell).lower() for cell in row.cells) or needle in row.key.lower()

    def set_filter(self, text: str) -> None:
        self.filter = text.strip()
        self.reload()

    def set_note(self, text: str) -> None:
        self._note.update(Text(text))
        self._note.display = bool(text)

    # ---------------------------------------------------------------- selection

    @property
    def visible(self) -> List[Row]:
        return [row for row in self._rows if self._matches(row)]

    def drill(self) -> None:
        """Act on the selected row, whatever asked -- a keypress or the table itself

        Enter arrives here as a DataTable.RowSelected message rather than as a
        key binding, because the focused DataTable binds enter to its own
        select_cursor and consumes it. An app-level binding for enter looks
        right, shows up in the footer, and never fires.
        """
        row = self.selected()
        if row is None:
            self.explorer.flash("nothing selected")
            return
        try:
            self.on_enter_row(row)
        except Exception as error:  # same promise as reload: report, never take the app down
            self.explorer.flash(f"[!] {error}")

    def on_key(self, event) -> None:
        """Dispatch this view's own keys, which bubble up from the focused table"""
        handler = self.keys.get(event.key)
        if handler is None:
            return
        event.stop()
        try:
            getattr(self, handler)()
        except Exception as error:  # a view key must report a failure, never take the app down
            self.explorer.flash(f"[!] {error}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on the table, which is the only way enter actually reaches a view"""
        event.stop()
        self.drill()

    def selected(self) -> Optional[Row]:
        """The row under the cursor, or None when the table is empty"""
        shown = self.visible
        index = self._table.cursor_row
        if index is None or not shown or index >= len(shown):
            return None
        return shown[index]

    @property
    def count(self) -> str:
        """`shown/total`, which is the number a filter makes worth printing"""
        shown, total = len(self.visible), len(self._rows)
        return f"{shown}/{total}" if shown != total else str(total)

    # ---------------------------------------------------------------- lifecycle

    def on_show(self) -> None:
        """Called when this view reaches the top of the stack"""
        self._table.focus()

    def on_hide(self) -> None:
        """Called when another view covers this one"""

class Detail(View):
    """A two-column record view: what `y` shows, and what a leaf drills into

    Everything in this explorer that is not a list is this. A record is
    key/value pairs in the order the producer wrote them, because reordering
    somebody's card into alphabetical order loses the grouping they chose.
    """

    title = "detail"
    columns = ("field", "value")

    def __init__(self, app_ref, session, title: str, record: Dict[str, Any], hints=(),
                 actions: Optional[Dict[str, Callable[[], None]]] = None):
        super().__init__(app_ref, session)
        self.title = title
        self.record = record
        self.hints = hints
        # field name -> what Enter on that field means. A record is inert by default,
        # but a producer that knows one of its fields names something openable says so
        # here -- which keeps Detail from having to learn what a run or a card is.
        self.actions: Dict[str, Callable[[], None]] = dict(actions or {})

    def rows(self) -> List[Row]:
        return [Row(key=str(name), cells=(str(name), _render(value)), payload=value)
                for name, value in self.record.items()]

    def on_enter_row(self, row: Row) -> None:
        """Open what this field names, or say it is a leaf like every other field"""
        action = self.actions.get(row.key)
        if action is None:
            super().on_enter_row(row)
            return
        action()

def _cell(value: Any) -> Text:
    """One cell as a renderable, keeping a style the view already chose

    A view that means something by colour builds its own Text -- the heat map's
    blocks carry magnitude and sign and nothing else. Everything else gets the
    plain wrapping that stops `[a, b]` being read as console markup.
    """
    return value if isinstance(value, Text) else Text(cell_text(value))

def cell_text(value: Any) -> str:
    """One table cell as a string, unwrapping the enums the schema uses

    A (str, Enum) formats as "Component.HEAD_OUT" rather than "head_out", which
    is the documented trap in share/schema/vocabulary.py. Every cell goes
    through here so no view has to remember it -- the first draft of this file
    did not, and the artifacts table shipped rows reading "Kind.CIRCUIT".
    """
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)

def _render(value: Any) -> str:
    """One cell for a value that may be a dict, a list, or a number

    Nested structure is flattened to a single line rather than truncated to
    nothing: a reader scanning a card wants to see that `landmarks` has four
    entries even when they do not all fit.
    """
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return f"{value:,.6g}"
    if isinstance(value, dict):
        return "  ".join(f"{name}={_render(item)}" for name, item in value.items()) or "{}"
    if isinstance(value, (list, tuple)):
        rendered = ", ".join(_render(item) for item in value)
        return f"[{rendered}]" if len(rendered) < 160 else f"[{len(value)} items] {rendered[:150]}..."
    return str(value)
