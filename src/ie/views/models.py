from typing import List

from ...core.config import load_config, presets
from ..view import Hint, Row, View

"""
Every model this lab knows how to load, read off configs/ without loading one.

The list is cheap because a config is a YAML file, and that is the point: you
can see what is available, what shape it is and which backend it needs before
deciding to spend a checkpoint on it. Sizes shown as `?` are the honest answer
for a config that has not been resolved against its weights yet -- the
framework reads n_layers and d_model off the checkpoint rather than trusting a
file, so a config that states them and a config that does not look different
here on purpose.

A common pipe could be: :models | enter | :layers
"""

class Models(View):
    """The configs in configs/, and which one the session is pointed at"""

    title = "models"
    columns = ("", "id", "backend", "checkpoint", "layers", "d_model", "heads", "device", "dtype")
    hints = (Hint("enter", "use this model"), Hint("y", "config"))

    def rows(self) -> List[Row]:
        found = []
        for name in presets():
            try:
                cfg = load_config(name)
            except ValueError as error:
                found.append(Row(key=name, cells=("", name, "[!]", str(error)[:60], "", "", "", "", "")))
                continue
            current = ">" if name == self.session.model_id else ""
            found.append(Row(
                key=name,
                cells=(
                    current, cfg.id, cfg.backend, cfg.hf_name,
                    cfg.n_layers if cfg.n_layers is not None else "?",
                    cfg.d_model if cfg.d_model is not None else "?",
                    cfg.n_heads if cfg.n_heads is not None else "?",
                    cfg.device, cfg.dtype,
                ),
                payload=cfg,
            ))
        return found

    def on_enter_row(self, row: Row) -> None:
        """Point the session at this model, which every other view reads from"""
        self.session.set_model(row.key)
        self.explorer.flash(f"model is now {row.key}; nothing loaded yet")
        self.reload()

    def detail(self, row: Row):
        cfg = row.payload
        if cfg is None:
            return None
        return {name: getattr(cfg, name) for name in (
            "id", "backend", "hf_name", "n_layers", "d_model", "n_heads",
            "probe_layer_frac", "batch_size", "device", "dtype", "max_new_tokens", "sae",
        )}
