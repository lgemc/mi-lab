"""
The decoder language model as one subject of study: its backend, its tasks, its
readout, its data, and what is about language in particular.

A domain is the second axis. The packages above this one are ordered by *what
kind of thing* a module is -- a config, a measurement, an artifact -- and that
order is one-way. This axis is orthogonal to it and answers a different
question: which subject a module is about. The kernel may not import a domain
and a domain may not import another, so the whole of `methods/` measures this
model the same way it would measure any other differentiable function with
addressable sites, and knowing which token id the answer is at lives here.

Importing this package is what fills the registries -- the backend under its
config key, the five tasks under their names, the one-input frames a server
builds prompts with, and the experiment kinds the runner dispatches on. Every
import below is for that side effect and nothing else, which is why they are
the whole body of this file.

Nothing above imports it. `src/plugins.py` names it as a string and imports it
on demand, which is the one door the kernel has to a domain and the reason the
contract in `.importlinter` can be mechanical rather than a paragraph.
"""

from . import backend as _backend  # noqa: F401  registers the transformers backend
from . import experiments as _experiments  # noqa: F401  registers ioi_circuit
from . import tasks as _tasks  # noqa: F401  registers the five tasks and their frames
