from dataclasses import dataclass, field
from typing import List, Sequence

from .errors import ArtifactError
from .vocabulary import Component, Position, one_of

"""
Where in a model something was measured, in both spellings.

Layer 8 of a twelve-layer model and layer 42 of a sixty-four-layer one are the
same place, and only the fraction survives a model swap. The index is what a
hook needs; the fraction is what another lab can place. Deriving one from the
other needs the model's depth, which the reader may not have resolved, so both
are written down.

A common pipe could be: cfg.layers | Site.at | Artifact
"""

@dataclass(frozen=True)
class Site:
    """Where in the model this was measured

    layers are absolute indices because that is what a hook needs, and fracs
    are the same layers as depth fractions because that is what transfers to a
    model of another size. Both are written down: deriving one from the other
    needs n_layers, and an artifact should be readable without resolving the
    checkpoint.
    """
    layers: List[int] = field(default_factory=list)
    fracs: List[float] = field(default_factory=list)
    component: Component = Component.RESIDUAL
    position: Position = Position.LAST

    @classmethod
    def at(
        cls,
        layers: Sequence[int],
        n_layers: int,
        component: Component | str = Component.RESIDUAL,
        position: Position | str = Position.LAST,
    ) -> "Site":
        """Build a site from absolute layers, stamping in the fractions they sit at

        The model's depth is required rather than optional. Without it the
        fraction cannot be computed, and a site carrying only indices is one
        the receiving lab cannot place -- which is the failure this whole
        field lives with today, not a corner case to paper over with a zero.
        """
        indices = [int(layer) for layer in layers]
        if not n_layers:
            raise ArtifactError(
                f"cannot place layers {indices} at a depth without knowing how many layers the model has; "
                "resolve the config against its checkpoint first, so the site says two thirds through "
                "rather than layer eight"
            )
        # coerced here as well as at load, so a typo is caught where it was written
        # rather than surviving until somebody tries to read the card back
        return cls(
            layers=indices,
            fracs=[round(layer / n_layers, 6) for layer in indices],
            component=one_of(Component, component, "site.component"),
            position=one_of(Position, position, "site.position"),
        )
