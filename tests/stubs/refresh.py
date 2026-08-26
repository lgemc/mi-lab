from pathlib import Path

import torch

from src.model.adapter import load_adapter
from tests.adapter import GOLDEN, GOLDEN_FRACS, GOLDEN_PROMPTS

"""
Regenerate the golden capture. Run this deliberately -- never to make a
failing test pass, only when you have decided the new activations are the
correct ones and can say why.

Run with: python -m tests.stubs.refresh
"""

if __name__ == "__main__":
    adapter = load_adapter("gpt2-small")
    activations = adapter.capture(GOLDEN_PROMPTS, layers=adapter.cfg.layers(GOLDEN_FRACS))
    torch.save(activations, GOLDEN)
    print(f"wrote {GOLDEN} {tuple(activations.shape)}")
