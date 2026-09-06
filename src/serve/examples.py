"""Held-out prompts to hand a served model, so nobody has to invent one.

A generation server with an empty text box asks its first visitor to write a
ToolAlpaca prompt from memory -- the tool documentation, the parameter schema
and the instruction, in the exact shape the model was fine-tuned on. They will
not, so they will type something else, and read the answer to a question the
model was never trained on as the model being bad.

So the prompts ship. One file per dataset used for fine-tuning, exported by
`Self-Distillation/export_heldout.py` from that dataset's **eval** split, which
training never reads. That split is the whole point: an example the model was
trained on tells you it memorised, not that it learned, and a demo built on one
is a demo that always works.

The files are data under the mount-or-image, read at request time and not at
start, so adding a dataset is writing a file -- the same rule the circuits
follow next door. A malformed file is reported by name rather than dropped: a
dataset that is silently absent looks exactly like one nobody exported.

A common pipe could be: eval split | export | mount | GET /examples | POST /infer
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

#: Where the exported files live inside the image; the Deployment can move it.
DEFAULT_ROOT = Path("data/sdft_examples")


class ExamplesError(ValueError):
    """An examples root that cannot be read, said with the way out"""


class Examples:
    """Every held-out example set under a root, re-read whenever asked"""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT

    def load(self) -> Dict[str, object]:
        """The datasets under the root, and the files that could not be read

        Returns the same shape whether or not the root exists. A server with no
        examples mounted is a normal state -- the circuit stacks have none --
        and it must not be an error, only an empty list.
        """
        datasets: List[dict] = []
        problems: List[str] = []
        if not self.root.is_dir():
            return {"root": str(self.root), "datasets": datasets, "problems": problems}

        for path in sorted(self.root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                problems.append(f"{path.name}: {type(error).__name__}: {error}")
                continue
            examples = payload.get("examples")
            if not isinstance(examples, list):
                problems.append(f"{path.name}: no 'examples' list")
                continue
            datasets.append({
                "dataset": payload.get("dataset", path.stem),
                "split": payload.get("split"),
                # Carried through rather than restated here: the exporter knows
                # what its gold column means and this module does not.
                "note": payload.get("note"),
                "split_size": payload.get("split_size"),
                "count": len(examples),
                "examples": examples,
            })
        return {"root": str(self.root), "datasets": datasets, "problems": problems}
