from typing import ClassVar, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Input, Static

from . import registry
from .command import ALIASES, parse
from .session import Session
from .stack import PageStack
from .view import Detail, View

"""
The explorer itself: one screen, a stack of views inside it, and a colon.

The shape is k9s's. A persistent header saying what the session is pointed at,
a breadcrumb saying how you got to the view you are looking at, a table in the
middle, and a command line that only appears when you ask for it. Escape pops
the stack rather than quitting, which is the single interaction that makes a
terminal explorer feel like one.

One screen rather than a stack of Textual screens, because the header and the
command line have to survive a drill-down: they are what tell you which model
these numbers are about, and a view that could cover them could show a table
of activations from a checkpoint that is no longer loaded.

The checkpoint is loaded here rather than inside a view, so that exactly one
place knows how to say "this will take a moment" and exactly one place catches
the failure when the weights are not reachable offline.

Run with: python -m src.ie
"""

class Chrome(Static):
    """The header: what the session is pointed at, and where you are in it"""

class Explorer(App):
    """Navigate a model, the runs it produced and the artifacts they shipped"""

    TITLE = "ie"
    SUB_TITLE = "interpretability explorer"

    CSS = """
    Screen { layout: vertical; }
    #chrome { height: auto; padding: 0 1; background: $panel; color: $text; }
    #chrome .ie-dim { color: $text-muted; }
    #body { height: 1fr; }
    #command { display: none; dock: bottom; }
    #command.ie-open { display: block; }
    #flash { height: auto; padding: 0 1; color: $warning; }
    """

    BINDINGS: ClassVar[List[Binding]] = [
        Binding("colon", "command", "command", key_display=":"),
        Binding("slash", "filter", "filter", key_display="/"),
        Binding("escape", "back", "back"),
        Binding("enter", "drill", "select"),
        Binding("y", "record", "record"),
        Binding("r", "reload", "refresh"),
        Binding("question_mark", "help", "help", key_display="?"),
        Binding("ctrl+c", "quit", "quit"),
    ]

    def __init__(self, session: Optional[Session] = None, start: str = registry.ROOT):
        super().__init__()
        self.session = session or Session()
        self.stack = PageStack(on_change=self.refresh_chrome)
        self.start_resource = start
        self.last_flash = ""
        self._chrome = Chrome(id="chrome")
        self._flash = Static("", id="flash")
        self._command = Input(placeholder="resource, or /filter", id="command")
        self._body = Container(id="body")

    # -------------------------------------------------------------- composition

    def compose(self) -> ComposeResult:
        yield self._chrome
        yield self._flash
        yield self._body
        yield Horizontal(self._command)
        yield Footer()

    def on_mount(self) -> None:
        self.run_command(self.start_resource)
        if len(self.stack) == 0:
            self.push(Detail(self, self.session, "help", _help_record()))

    # -------------------------------------------------------------------- stack

    def push(self, view: View) -> None:
        """Show a view, loading a checkpoint first when it says it needs one

        A covered view is hidden rather than unmounted, so going back reveals
        the table you left -- same cursor, same filter -- instead of rebuilding
        it. Anything the stack drops is unmounted for real by `_show_top`.
        """
        if view.needs_model and not self.session.loaded:
            self.flash(f"loading {self.session.model_id} ... first use pays for the checkpoint")
            try:
                self.session.adapter()
            except Exception as error:
                self.flash(f"[!] cannot load {self.session.model_id}: {error}")
                return
            self.flash(f"{self.session.model_id} loaded")
        dropped = set(self.stack.pages)
        self.stack.push(view)
        kept = set(self.stack.pages)
        for gone in dropped - kept:
            gone.remove()
        for covered in self._body.children:
            covered.display = False
        self._body.mount(view)
        # focus after the mount lands, not during the push: on_show focuses the table,
        # and focusing a widget that is not mounted yet quietly does nothing -- which
        # is a keyboard that ignores you on the view you just opened
        self.call_after_refresh(view.on_show)

    def _show_top(self) -> None:
        """Unmount whatever left the stack and reveal what it was covering"""
        kept = set(self.stack.pages)
        for mounted in list(self._body.children):
            if mounted not in kept:
                mounted.remove()
        top = self.current()
        for mounted in self._body.children:
            mounted.display = mounted is top
        if top is not None:
            top.on_show()

    def current(self) -> Optional[View]:
        return self.stack.top()

    # ------------------------------------------------------------------ commands

    def run_command(self, line: str) -> None:
        """Do whatever a command line says, or say why it cannot be done"""
        command = parse(line)
        if command is None:
            return
        if command.resource == "quit":
            self.exit()
            return
        if command.resource == "help":
            self.push(Detail(self, self.session, "help", _help_record()))
            return
        if command.is_setter:
            self._apply_setter(command.resource, command.argument)
            return

        view = registry.build(command.resource, self, self.session)
        if view is None:
            known = ", ".join(sorted(registry.VIEWS))
            self.flash(f"no resource '{command.resource}'; known are {known}  (`:help` lists the aliases)")
            return
        self.push(view)
        if command.filter:
            view.set_filter(command.filter)

    def _apply_setter(self, name: str, argument: str) -> None:
        """`:model`, `:prompt` and `:root` change the session every view reads"""
        if not argument:
            current = {"model": self.session.model_id, "prompt": self.session.prompt, "root": self.session.root}
            self.flash(f"{name} is {current[name]!r} -- give it a value to change it")
            return
        if name == "model":
            self.session.set_model(argument)
            self.flash(f"model is now {argument}; nothing loaded yet")
        elif name == "prompt":
            self.session.set_prompt(argument)
            self.flash(f"prompt set, {len(argument)} characters")
        else:
            self.session.root = argument
            self.flash(f"root is now {argument}")
        view = self.current()
        if view is not None:
            view.reload()
        self.refresh_chrome()

    # -------------------------------------------------------------------- actions

    def action_command(self) -> None:
        self._open_input(":", "resource, e.g. runs, artifacts, layers, or `model gpt2-small`")

    def action_filter(self) -> None:
        self._open_input("/", "filter the rows shown")

    def _open_input(self, prefix: str, placeholder: str) -> None:
        self._command.value = prefix
        self._command.placeholder = placeholder
        self._command.add_class("ie-open")
        self._command.focus()
        self._command.cursor_position = len(prefix)

    def _close_input(self) -> None:
        self._command.remove_class("ie-open")
        self._command.value = ""
        view = self.current()
        if view is not None:
            view.on_show()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value
        self._close_input()
        if text.startswith("/"):
            view = self.current()
            if view is not None:
                view.set_filter(text[1:])
            return
        self.run_command(text)

    def action_back(self) -> None:
        """Escape: clear a filter if there is one, else pop the stack"""
        if self._command.has_class("ie-open"):
            self._close_input()
            return
        view = self.current()
        if view is not None and view.filter:
            view.set_filter("")
            self.flash("filter cleared")
            return
        if self.stack.pop() is None:
            self.flash("at the root -- `:q` or ctrl+c to leave")
            return
        self._show_top()

    def action_drill(self) -> None:
        """Enter, for the case where the table does not have focus

        The table almost always does, and then it handles enter itself and
        this never runs -- see View.drill. Both roads end in the same method.
        """
        view = self.current()
        if view is not None:
            view.drill()

    def action_record(self) -> None:
        """`y`: the record behind the selected row, which is k9s's YAML view"""
        view = self.current()
        row = view.selected() if view else None
        if view is None or row is None:
            self.flash("nothing selected")
            return
        record = view.detail(row)
        if record is None:
            self.flash(f"{row.key} has no record behind it")
            return
        self.push(Detail(self, self.session, f"{view.title}/{row.key}", record))

    def action_reload(self) -> None:
        view = self.current()
        if view is not None:
            view.reload()
            self.flash(f"reloaded {view.title}")

    def action_help(self) -> None:
        self.push(Detail(self, self.session, "help", _help_record()))

    # --------------------------------------------------------------------- chrome

    def flash(self, message: str) -> None:
        """One transient line, for the thing that just happened or just failed

        Kept on the app as well as on the widget: what was last said is part of
        what the explorer did, and reading it back off a Static is reaching
        into a rendering to recover a fact.
        """
        self.last_flash = message
        self._flash.update(message)

    def refresh_chrome(self) -> None:
        """Redraw the header: the session context, then where you are in it"""
        view = self.current()
        where = self.stack.breadcrumb() or "-"
        count = f"  [{view.count}]" if view is not None else ""
        hints = "  ".join(f"<{hint.key}> {hint.description}" for hint in (view.hints if view else ()))
        filtered = f"  /{view.filter}" if view is not None and view.filter else ""
        self._chrome.update(
            f"model   {self.session.describe()}\n"
            f"prompt  {_shorten(self.session.prompt)}\n"
            f"root    {self.session.root}\n"
            f"view    {where}{count}{filtered}   {hints}"
        )

def _shorten(text: str, width: int = 96) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."

def _help_record() -> dict:
    """What the explorer answers to, built from the registry rather than restated"""
    record = {
        "usage": "press : for a command, / to filter, enter to drill in, escape to go back",
        "": "--- resources ---",
    }
    for name, aliases, needs_model in registry.catalogue():
        cost = "loads the checkpoint on first use" if needs_model else "no model needed"
        record[f":{name}"] = f"aliases {aliases or '-'}   ({cost})"
    record["  "] = "--- session ---"
    record[":model <id>"] = "point at another config, e.g. `:model gpt2-small`; drops what was loaded"
    record[":prompt <text>"] = "the text the token and activation views work over"
    record[":root <dir>"] = "where to look for runs and artifacts (default: outputs)"
    record["   "] = "--- keys ---"
    record["enter"] = "drill into the selected row"
    record["escape"] = "clear the filter, else go back one view"
    record["y"] = "the record behind the selected row"
    record["r"] = "recompute this view"
    record[":q / ctrl+c"] = "leave"
    record["    "] = "--- filters ---"
    record["`:runs failed`"] = "a resource and a filter in one line"
    record["`:artifacts /gpt2`"] = "the slash form, same thing"
    record["aliases"] = ", ".join(f"{alias}={target}" for alias, target in sorted(ALIASES.items()))
    return record
