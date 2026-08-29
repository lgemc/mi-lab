from typing import Optional

from ...core.config import ModelConfig, load_config
from ..schema import ModelRef

"""
The one thing every converter needs and none of them should each answer:
which model was this measured on.

A ModelRef whose hf_name was invented is worse than no artifact at all -- it
names a checkpoint that will load and produce different numbers -- so this
refuses to guess rather than filling the field in.

A common pipe could be: model_ref | Artifact | save
"""

def model_ref(cfg: Optional[ModelConfig], model_id: str) -> ModelRef:
    """Describe the model an artifact was measured on, refusing to guess at it

    A ModelRef whose hf_name was invented is worse than no artifact: it names
    a checkpoint that will load and produce different numbers.
    """
    if cfg is not None:
        return ModelRef.from_config(cfg)
    try:
        return ModelRef.from_config(load_config(model_id))
    except ValueError as error:
        raise ValueError(
            f"'{model_id}' does not name a config, so the checkpoint behind this result cannot be written "
            "down; pass the ModelConfig it was measured on"
        ) from error
