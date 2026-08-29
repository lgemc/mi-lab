from typing import Callable, List, Optional

"""
Where the explorer has been, and how Escape gets back there.

k9s calls this the PageStack, and it is the piece that makes a terminal
explorer navigable rather than a set of screens with no relationship: pushing
a view remembers the one underneath, Escape pops back to it, and the trail of
titles is the breadcrumb that says how you got here.

Views are told when they are covered and uncovered, so a view that is not on
top can stop working. Nothing here captures anything expensive yet, but the
lifecycle is the hook that keeps a future live-updating view from polling a
model while the user is reading something else.

A common pipe could be: parse | push | pop | breadcrumb
"""

class PageStack:
    """The view history: push to drill in, pop to go back, top is what is shown"""

    def __init__(self, on_change: Optional[Callable[[], None]] = None):
        self._pages: List = []
        self._on_change = on_change

    def __len__(self) -> int:
        return len(self._pages)

    @property
    def pages(self) -> List:
        return list(self._pages)

    def top(self):
        """The view being shown, or None before anything has been pushed"""
        return self._pages[-1] if self._pages else None

    def push(self, page) -> None:
        """Show a view, remembering whatever it covers

        Pushing the same page twice in a row replaces rather than stacks, so
        that re-running `:runs` from the runs view does not build a pile of
        identical pages Escape has to walk back down.

        Sameness is the title, not the class. Two views of the same class are
        routinely different pages -- `activations` and `activations L9` are the
        layer list and one layer's positions -- and comparing types made
        drilling in silently replace the view you were drilling from.
        """
        covered = self.top()
        if covered is not None:
            covered.on_hide()
            if covered.title == page.title:
                self._pages.pop()
        self._pages.append(page)
        page.on_show()
        self._changed()

    def pop(self):
        """Go back one view, refusing to pop the last one

        The root view is what the explorer is; popping it would leave an empty
        screen with no way to say what to do next.
        """
        if len(self._pages) <= 1:
            return None
        page = self._pages.pop()
        page.on_hide()
        revealed = self.top()
        if revealed is not None:
            revealed.on_show()
        self._changed()
        return page

    def clear_to_root(self):
        """Drop everything above the root, which is what a fresh command does"""
        while len(self._pages) > 1:
            self._pages.pop().on_hide()
        if self._pages:
            self._pages[0].on_show()
        self._changed()

    def breadcrumb(self) -> str:
        """The trail of titles, which is the only thing that says how you got here"""
        return " > ".join(page.title for page in self._pages)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
