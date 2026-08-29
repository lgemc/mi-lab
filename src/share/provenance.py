import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch

"""
Who made an artifact, from which code, and whether that code was committed.

It sits apart from the schema because it is the one part of a card that is
read off this machine rather than measured: it shells out to git. An artifact
whose provenance is confidently wrong is worse than one with none, which is
why `dirty` is recorded beside the commit rather than the commit being
recorded alone.

A common pipe could be: stamp | Artifact | save
"""

def stamp(format_version: str, **extra: Any) -> Dict[str, Any]:
    """The provenance every artifact gets: which tool, which commit, and whether it was clean

    The format version is passed in rather than read from the schema, because
    the schema stamps every artifact it builds and a module cannot import the
    one importing it.

    `dirty` is the field that keeps the commit honest. A hash recorded from a
    tree with uncommitted edits names code that never existed, and an artifact
    whose provenance is confidently wrong is worse than one with none.
    """
    described = _git("describe", "--always", "--dirty")
    record: Dict[str, Any] = {
        "tool": "mi-lab",
        "format_version": format_version,
        "git_commit": (described or "").removesuffix("-dirty") or None,
        "git_dirty": bool(described and described.endswith("-dirty")),
        "torch": torch.__version__,
    }
    record.update(extra)
    return record

def _git(*args: str) -> Optional[str]:
    """Ask git something about this checkout, or None if there is no answer to be had"""
    try:
        result = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
