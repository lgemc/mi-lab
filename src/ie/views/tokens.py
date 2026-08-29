from typing import List

from ...model.adapter import require_circuits
from ..view import Hint, Row, View

"""
How the model actually cuts the current prompt up.

Tokenization is the step where a prompt stops being the sentence you wrote,
and almost every confusing interpretability result starts here: a leading
space is part of the token, a name that looks single-token is two, and a
position index in a heatmap means nothing until you can see which string it
points at. Showing the strings with their indices is what makes every other
position-indexed view readable.

The repeated-token column exists because the IOI task is built on one name
appearing twice, and finding those positions by eye in a wrapped line is how
you mislabel a landmark.

A common pipe could be: :prompt <text> | :tokens | :activations
"""

class Tokens(View):
    """The current prompt as the model sees it"""

    title = "tokens"
    columns = ("index", "token", "repr", "id", "note")
    needs_model = True
    hints = (Hint("enter", "nothing under a token"),)

    def rows(self) -> List[Row]:
        adapter = require_circuits(self.session.adapter())
        prompt = self.session.prompt
        pieces = self.session.cached("tokens", lambda: adapter.tokens(prompt))
        seen = {}
        for index, piece in enumerate(pieces):
            seen.setdefault(piece, []).append(index)

        found = []
        for index, piece in enumerate(pieces):
            where = seen[piece]
            note = f"repeats at {where}" if len(where) > 1 else ""
            if index == len(pieces) - 1:
                note = (note + "  " if note else "") + "END (what the model predicts from)"
            try:
                token_id = adapter.single_token(piece)
            except Exception:
                token_id = "-"
            found.append(Row(key=str(index), cells=(index, piece, repr(piece), token_id, note)))
        return found

    def _empty_note(self) -> str:
        return "no prompt -- set one with `:prompt <text>`"
