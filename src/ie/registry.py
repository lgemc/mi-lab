from typing import Dict, List, Tuple

from .views.activations import Activations
from .views.artifacts import Artifacts
from .views.layers import Layers
from .views.models import Models
from .views.nodes import Nodes
from .views.runs import Runs
from .views.tensors import Tensors
from .views.tokens import Tokens

"""
Which resource name builds which view.

k9s calls this customViewers, and the reason it is a table rather than a
conditional is that adding a resource should be an entry here and a module
under views/ -- never an edit to the command parser or the app. A view that
cannot be reached without touching three files is a view nobody adds.

The order is the order `:help` lists them in, which is roughly the order you
would explore in: what exists, what ran, what shipped, then into the model.

A common pipe could be: parse | build | PageStack.push
"""

VIEWS: Dict[str, type] = {
    "models": Models,
    "runs": Runs,
    "artifacts": Artifacts,
    "layers": Layers,
    "tokens": Tokens,
    "activations": Activations,
    "nodes": Nodes,
    "tensors": Tensors,
}

ROOT = "runs"

def build(resource: str, app_ref, session, argument: str = ""):
    """The view a resource name means, or None if nothing answers to it"""
    factory = VIEWS.get(resource)
    return None if factory is None else factory(app_ref, session, argument=argument)

def catalogue() -> List[Tuple[str, str, bool]]:
    """Every resource, its aliases and whether it needs a checkpoint, for `:help`"""
    from .command import ALIASES

    rows = []
    for name, factory in VIEWS.items():
        aliases = sorted(alias for alias, target in ALIASES.items() if target == name)
        rows.append((name, ", ".join(aliases), bool(factory.needs_model)))
    return rows
