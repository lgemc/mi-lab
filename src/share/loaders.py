from pathlib import Path

from ..methods.probing import LinearProbe
from . import storage
from .converters.probe import to_probe

"""
Opening a result without first knowing which form it arrived in.

Anything that only wants to *apply* a probe should not have to know whether it
was handed the .pt this repository writes or a shared .mia artifact. That is
the whole promise of a shared format, and the cheapest place to keep it is one
door rather than a branch in every caller.

A common pipe could be: open_probe | score | steer
"""

def open_probe(path: str) -> LinearProbe:
    """Read a probe from either the .pt this repo writes or a shared .mia artifact

    Anything that only wants to *apply* a probe should not have to know which
    of the two it was handed. That is the whole promise of a shared format,
    and the cheapest place to keep it is here rather than in every caller.
    """
    source = Path(path)
    if source.is_dir():
        return to_probe(storage.load(str(source)))
    return LinearProbe.load(path)
