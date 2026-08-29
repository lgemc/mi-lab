from typing import List

from ..view import Hint, Row, View

"""
Every layer of the current model, addressed the way this framework addresses
one: by the depth fraction it sits at.

The index column is here because a hook needs it, and the fraction column is
here because that is what transfers to a model of another size -- the same
reason an artifact writes both down. The row marked as the probe layer is the
config's `probe_layer_frac` resolved, which is the one number a config states
about where it expects the signal to live.

This view needs no checkpoint when the config already carries n_layers, and
asks for one when it does not, which is the framework's rule that sizes are
read off the weights rather than trusted from a file.

A common pipe could be: :layers | enter | :activations
"""

class Layers(View):
    """The model's layers, by index and by depth"""

    title = "layers"
    columns = ("", "layer", "depth", "of", "note")
    hints = (Hint("enter", "activations here"),)

    def rows(self) -> List[Row]:
        cfg = self.session.config()
        if cfg.n_layers is None:
            # the framework reads sizes off the checkpoint rather than a file, so a
            # config that does not state its depth genuinely needs the weights
            cfg = self.session.adapter().cfg
        probe_layer = cfg.layer()
        found = []
        for index in range(cfg.n_layers):
            frac = index / cfg.n_layers
            note = "probe layer (probe_layer_frac)" if index == probe_layer else ""
            found.append(Row(
                key=str(index),
                cells=(">" if index == probe_layer else "", index, f"{frac:.3f}", cfg.n_layers, note),
                payload=index,
            ))
        return found

    def on_enter_row(self, row: Row) -> None:
        from .activations import Activations

        self.explorer.push(Activations(self.explorer, self.session, argument=str(row.payload)))
