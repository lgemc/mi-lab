"""The crosscoder on planted structure: what it must recover, and what latent scaling must tell apart"""

from unittest import TestCase

import torch

from src.methods.common.errors import CrosscoderError
from src.methods.crosscoder import (
    POST,
    PRE,
    Crosscoder,
    classify,
    latent_scaling,
    relative_norms,
    train_step,
)


class TestBatchTopK(TestCase):
    def test_training_keeps_exactly_k_per_row_on_average(self):
        coder = Crosscoder(width=8, latents=32, k=3)
        x = torch.randn(10, 2, 8)
        f = coder.encode(x, batch_topk=True)
        self.assertEqual(int((f > 0).sum()), 3 * 10)

    def test_k_outside_the_latents_is_refused(self):
        with self.assertRaises(CrosscoderError):
            Crosscoder(width=4, latents=8, k=9)


class TestClasses(TestCase):
    def test_relative_norm_sorts_hand_built_decoders(self):
        coder = Crosscoder(width=4, latents=3, k=1)
        with torch.no_grad():
            coder.W_dec.zero_()
            coder.W_dec[0, PRE, 0] = 1.0                       # only the first model
            coder.W_dec[1, POST, 1] = 1.0                      # only the second
            coder.W_dec[2, PRE, 2] = coder.W_dec[2, POST, 2] = 1.0
        norms = relative_norms(coder)
        classes = classify(norms, torch.ones(3, dtype=torch.bool))
        self.assertEqual(classes, {"pre_only": [0], "post_only": [1], "shared": [2]})

    def test_a_dead_latent_is_in_no_class(self):
        norms = torch.tensor([0.0, 1.0])
        classes = classify(norms, torch.tensor([False, True]))
        self.assertEqual(classes["pre_only"], [])
        self.assertEqual(classes["post_only"], [1])


class TestLatentScaling(TestCase):
    """One latent whose post decoder is e0 and whose pre decoder is zero: exclusive or shrunk?"""

    def coder(self) -> Crosscoder:
        coder = Crosscoder(width=4, latents=1, k=1)
        with torch.no_grad():
            coder.W_dec.zero_()
            coder.W_dec[0, POST, 0] = 1.0
            coder.W_enc.zero_()
            coder.W_enc[POST, 0, 0] = 1.0                      # the latent reads e0 in the post model
            coder.b_enc.zero_()
            coder.b_dec.zero_()
            coder.threshold.zero_()
        return coder

    def rows(self, pre_has_it: bool) -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        amount = torch.rand(256, generator=generator) + 0.5
        x = torch.zeros(256, 2, 4)
        x[:, POST, 0] = amount
        if pre_has_it:
            x[:, PRE, 0] = amount
        return x

    def test_a_truly_exclusive_latent_scales_to_zero_on_the_other_model(self):
        (row,) = latent_scaling(self.coder(), [self.rows(pre_has_it=False)], [0], side=POST)
        self.assertAlmostEqual(row["nu_error"], 0.0, places=5)
        self.assertAlmostEqual(row["nu_activation"], 0.0, places=5)

    def test_a_shrunk_decoder_scales_to_one(self):
        (row,) = latent_scaling(self.coder(), [self.rows(pre_has_it=True)], [0], side=POST)
        self.assertAlmostEqual(row["nu_error"], 1.0, places=5)
        self.assertAlmostEqual(row["nu_activation"], 1.0, places=5)


class TestTraining(TestCase):
    def test_it_finds_a_direction_only_the_second_model_has(self):
        torch.manual_seed(0)
        width, rows = 16, 512
        shared = torch.nn.functional.normalize(torch.randn(6, width), dim=-1)
        exclusive = torch.nn.functional.normalize(torch.randn(1, width), dim=-1)

        def batch():
            codes = (torch.rand(rows, 6) < 0.15).float() * (torch.rand(rows, 6) + 0.5)
            extra = (torch.rand(rows, 1) < 0.15).float() * (torch.rand(rows, 1) + 0.5)
            x = torch.zeros(rows, 2, width)
            x[:, PRE] = codes @ shared
            x[:, POST] = codes @ shared + extra @ exclusive
            return x

        coder = Crosscoder(width=width, latents=32, k=2, seed=0)
        coder.fit_scales(batch())
        optimiser = torch.optim.Adam(coder.parameters(), lr=3e-3)
        for _ in range(600):
            train_step(coder, optimiser, coder.normalise(batch()), dead_after=50)
        norms = relative_norms(coder)
        alignment = torch.nn.functional.cosine_similarity(coder.W_dec.detach()[:, POST], exclusive, dim=-1)
        best = int(alignment.argmax())
        self.assertGreater(float(alignment[best]), 0.9)
        self.assertGreater(float(norms[best]), 0.9)
