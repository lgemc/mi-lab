from dataclasses import dataclass
from typing import Optional

"""
Which checkpoint a result was measured on.

A ModelRef whose hf_name was invented is worse than no artifact at all: it
names a checkpoint that will load and produce different numbers. The sizes are
here because they are what a payload is checked against before it is applied
to anything -- a probe fit at one width, applied at another, scores confidently
and wrongly.

A common pipe could be: load_config | ModelRef.from_config | Artifact
"""

@dataclass(frozen=True)
class ModelRef:
    """Which model the measurement was made on, in enough detail to repeat it

    hf_name and revision are what another lab resolves; the sizes are what a
    loader checks a payload against before it applies it to anything.

    `dtype` is the compute dtype the measurement was made in, and stays a
    string to match ModelConfig.dtype rather than being closed here -- the set
    is torch's, and it grows. The ones this repository's configs use are
    "float32" and "bfloat16"; "float16" and the quantized names are equally
    valid to write. It matters because two runs of one checkpoint at different
    precisions are different measurements, and this is the only field that
    says so.

    `revision` is the checkpoint revision when one is pinned, and None when it
    is not -- which is the common case and a stated limit of the format: two
    quantizations of one repository are indistinguishable in a card.

    `n_layers`, `d_model` and `n_heads` are Optional in the type and
    load-bearing in practice. They are what a payload's widths are checked
    against, and `artifact check` warns when they are missing, because a
    payload that cannot be checked is one applied on trust.
    """
    id: str
    hf_name: str
    revision: Optional[str] = None
    n_layers: Optional[int] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    dtype: str = "float32"

    @classmethod
    def from_config(cls, cfg) -> "ModelRef":
        """Read a ModelRef off a resolved ModelConfig"""
        return cls(
            id=cfg.id, hf_name=cfg.hf_name, n_layers=cfg.n_layers,
            d_model=cfg.d_model, n_heads=cfg.n_heads, dtype=cfg.dtype,
        )
