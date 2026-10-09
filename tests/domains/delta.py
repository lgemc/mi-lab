"""Truncating and decomposing a fine-tune's update, on a two-block Qwen3 built from its config.

Offline: the model is random and a few dozen dimensions wide, and the "fine-tune"
is that model with a small perturbation added to every block map. What is
checked is the bookkeeping every phase 3 number rests on -- that the models
these tools build really sit between the two checkpoints:

- the whole update put back is the later model, none of it is the earlier one,
  and a top-k truncation plus its complement is the whole update;
- a decomposition with every subcomponent and the leftover on is the later
  model, with everything off it is the earlier one, and that holds after the
  subcomponents have been moved away from their initial values;
- the closed-form leftover norm is the norm of the leftover;
- reverting a subcomponent changes the model, and the gates are in [0, 1].
"""

import copy
from unittest import TestCase

import torch

from src.domains.lm.analysis import delta


def pair(seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, vocab_size=50, max_position_embeddings=64)
    before = Qwen3ForCausalLM(config).eval()
    after = copy.deepcopy(before)
    with torch.no_grad():
        for module in delta.targets(after).values():
            module.weight.add_(0.05 * torch.randn_like(module.weight))
    return before, after


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


class TestTruncation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before, cls.after = pair()
        cls.ids = torch.randint(0, 50, (2, 7))
        cls.deltas = delta.updates(cls.before, cls.after)
        cls.base = {name: m.weight.detach().float().clone() for name, m in delta.targets(cls.before).items()}
        cls.factors = delta.Factors(cls.deltas, "cpu")

    def test_every_block_map_is_a_target(self):
        names = list(delta.targets(self.before))
        self.assertEqual(len(names), 2 * 7)
        self.assertEqual(delta.kind(names[-1]), (1, "down_proj"))

    def test_all_of_the_update_is_the_later_model_and_none_is_the_earlier(self):
        work = copy.deepcopy(self.before)
        delta.truncate(work, self.base, self.factors, None)
        torch.testing.assert_close(logits(work, self.ids), logits(self.after, self.ids), atol=1e-4, rtol=1e-4)
        delta.truncate(work, self.base, self.factors, 0)
        torch.testing.assert_close(logits(work, self.ids), logits(self.before, self.ids), atol=1e-5, rtol=1e-5)

    def test_a_truncation_and_its_complement_sum_to_the_update(self):
        for name, update in self.deltas.items():
            whole = self.factors.piece(name, 3, False) + self.factors.piece(name, 3, True)
            torch.testing.assert_close(whole, update, atol=1e-5, rtol=1e-4)

    def test_the_spectrum_of_a_rank_one_update(self):
        update = torch.outer(torch.randn(12), torch.randn(9))
        spectrum = delta.Spectrum.of("model.layers.0.mlp.up_proj", update, torch.randn(12, 9))
        self.assertAlmostEqual(spectrum.stable_rank, 1.0, places=4)
        self.assertEqual(spectrum.rank_for["0.99"], 1)


class TestDecomposition(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before, cls.after = pair(1)
        cls.ids = torch.randint(0, 50, (2, 7))
        deltas = delta.updates(cls.before, cls.after)
        cls.model = copy.deepcopy(cls.before)
        cls.maps = delta.install(cls.model, deltas, dict.fromkeys(deltas, 5), hidden=8)
        # Move the subcomponents off the SVD they start from, so the checks
        # below are about the bookkeeping and not about that starting point.
        with torch.no_grad():
            for module in cls.maps.values():
                module.V.add_(0.1 * torch.randn_like(module.V))
                module.U.add_(0.1 * torch.randn_like(module.U))

    def tearDown(self):
        delta.set_masks(self.maps, None, None)

    def test_everything_on_is_the_later_model(self):
        torch.testing.assert_close(logits(self.model, self.ids), logits(self.after, self.ids), atol=1e-4, rtol=1e-4)

    def test_everything_off_is_the_earlier_model(self):
        off = {name: torch.zeros(1, 1, m.components) for name, m in self.maps.items()}
        rest = {name: torch.zeros(1, 1) for name in self.maps}
        delta.set_masks(self.maps, off, rest)
        torch.testing.assert_close(logits(self.model, self.ids), logits(self.before, self.ids),
                                   atol=1e-5, rtol=1e-5)

    def test_the_closed_form_leftover_norm(self):
        for module in self.maps.values():
            with torch.no_grad():
                self.assertAlmostEqual(float(module.rest_norm()), float(module.rest().pow(2).sum()), places=3)
        self.assertGreater(float(delta.faithfulness(self.maps)), 0.0)

    def test_reverting_a_subcomponent_moves_the_model_and_reverting_none_does_not(self):
        reference = logits(self.after, self.ids)
        delta.revert(self.maps, [])
        torch.testing.assert_close(logits(self.model, self.ids), reference, atol=1e-4, rtol=1e-4)
        first = next(iter(self.maps))
        delta.revert(self.maps, [(first, 0)])
        self.assertGreater(float((logits(self.model, self.ids) - reference).abs().max()), 1e-4)

    def test_gates_are_between_zero_and_one(self):
        module = next(iter(self.maps.values()))
        with torch.no_grad():
            gate, upper = module.gates(torch.randn(2, 7, module.V.shape[0]) * 10)
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)
        self.assertLessEqual(float(upper.max()), 1.0)
