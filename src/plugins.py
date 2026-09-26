import importlib
from typing import Tuple

"""
Where the kernel finds the domains, without importing one.

The second axis has one rule that is load-bearing and one that is always
locally convenient to break: **the kernel may not import a domain**. Every
registry in this repository is filled by importing the module that registers
into it, though -- `BACKENDS` is empty until a backend module runs, `TASKS`
until a task module does -- and `model/adapter.py` solved that by importing
its backends at the bottom of the file, which is now an import from the kernel
into a domain and exactly what the contract forbids.

So the edge is a string. This module names each domain as text and imports it
on demand; `.importlinter` sees no dependency because there is none to see,
and what the kernel actually depends on is a *list of names*, which is the
same thing `configs/*.yaml` already does when it names a backend.

Two properties make this honest rather than a way around the check:

- It is one place. A domain added anywhere else is a domain nothing loads.
- The import happens at *call* time, never at import time. A registry lookup
  loads the domains and then looks; by then the kernel module holding the
  registry is fully imported, so the domain importing it back is not a cycle.
  That is what the bottom-of-the-file import was buying, bought differently.

A common pipe could be: load_domains | BACKENDS | load_adapter
"""

#: Every domain in this repository, as an import path. A domain is one subject
#: of study: a model family, its tasks, and how a number comes out of it.
DOMAINS: Tuple[str, ...] = ("src.domains.lm",)

_loaded = False


def load_domains() -> None:
    """Import every domain once, so the registries they fill are not empty

    Idempotent and cheap after the first call. Callers are the registry lookups
    themselves -- `load_adapter`, `build_task`, `run_experiment` -- rather than
    every entry point, because a library caller who imported `run_experiment`
    directly should not have to remember an import the CLI happens to do.
    """
    global _loaded
    if _loaded:
        return
    # set before importing, not after: a domain that registers into a registry
    # whose lookup calls back in here would otherwise recurse forever
    _loaded = True
    for name in DOMAINS:
        importlib.import_module(name)


def domain_names() -> Tuple[str, ...]:
    """The domains this repository ships, for a message that has to list them"""
    return DOMAINS
