from typing import Any, Dict, Tuple

from .definitions import describe
from .schema.errors import ArtifactError
from .schema.version import VERSION

"""
Reading a card written by an older version of this format.

The format refuses a version it does not implement rather than guessing at the
difference, which is the right default and leaves a hole: an artifact written
last week is unreadable by the reader you have today, and the only honest way
out is a migration that says what it changed.

v0.1 -> v0.2 is the one that exists. It moves each metric from a bare float to
a value with the definition that produced it, and adds the two blocks whose
whole point is that they are written even when empty. The definitions are
recovered from the table this repository writes rather than invented; for a
name the table does not know, the definition says the definition was never
recorded, which is a true statement and the one the v0.2 gate is asking for.

What a migration must not do is make an old artifact look like it was measured
under the new rules. An upgraded card carries `upgraded_from` in its
provenance, so a reader can tell a circuit that recorded no cross-task control
from one that never had the field.

A common pipe could be: load | upgrade | save
"""

UPGRADES = ("0.1",)

def needs_upgrade(manifest: Dict[str, Any]) -> bool:
    """Whether this card is an older version this module knows how to move forward"""
    return str(manifest.get("version", "")) in UPGRADES

def explain(manifest: Dict[str, Any]) -> str:
    """Why a card will not load, in terms of what to do about it"""
    version = manifest.get("version", "unknown")
    if needs_upgrade(manifest):
        return (
            f"this is a {version} card and this reader is v{VERSION}; upgrade it with "
            "`python -m src.cli artifact upgrade <path>` (the original is kept unless --in-place)"
        )
    return (
        f"this is a v{version} card and this reader is v{VERSION}, and no migration from it exists; "
        "read it with a matching version rather than guessing at the difference"
    )

def upgrade(manifest: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    """Move a card forward to the current version, and say what that changed

    Returns the new card and one line per change, because a migration that
    edits somebody's result silently is one they cannot check.
    """
    version = str(manifest.get("version", ""))
    if version == VERSION:
        return dict(manifest), []
    if version not in UPGRADES:
        raise ArtifactError(explain(manifest))

    card = dict(manifest)
    changes = []
    measurement = dict(card.get("measurement") or {})

    metrics = {}
    for name, value in (measurement.get("metrics") or {}).items():
        if isinstance(value, dict):
            metrics[name] = value
            continue
        definition, units = describe(name, str(measurement.get("method", "")))
        metrics[name] = {"value": float(value), "definition": definition, "units": units}
        recovered = "recovered" if not definition.startswith("not recorded") else "marked unrecorded"
        changes.append(f"metric '{name}': definition {recovered}, units '{units}'")
    measurement["metrics"] = metrics

    if "identifiability" not in measurement:
        measurement["identifiability"] = []
        changes.append("measurement.identifiability: added as [] -- no assumption was recorded")
    card["measurement"] = measurement

    if "controls" not in card:
        card["controls"] = {"cross_task": [], "random_baseline": []}
        changes.append("controls: added as empty -- no control was recorded, which is now sayable")

    card["version"] = VERSION
    provenance = dict(card.get("provenance") or {})
    provenance["upgraded_from"] = version
    card["provenance"] = provenance
    changes.append(f"version: {version} -> {VERSION}, recorded as provenance.upgraded_from")
    return card, changes
