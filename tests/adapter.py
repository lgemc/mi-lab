from pathlib import Path
from unittest import TestCase, skipUnless

import torch

from src.core.config import ConfigError, Position, load_config
from src.model.adapter import BACKENDS, load_adapter

"""
Adapter tests need a real checkpoint, so they load GPT-2 small once for the
whole class and skip (loudly) if it is not available offline.

The golden capture is the important one. Later work asks questions like "does
quantization change the model's internals" -- questions you cannot answer if
you cannot first rule out that your own capture code changed. Four frozen
prompts, two layers, compared against a stored tensor.
"""

GOLDEN = Path(__file__).parent / "stubs" / "gpt2-small-capture.pt"
GOLDEN_PROMPTS = [
    "The capital of France is",
    "The Eiffel Tower stands in",
    "Water boils at a temperature of",
    "She handed the letter to",
]
GOLDEN_FRACS = [0.1, 0.65]

def _adapter():
    """Load GPT-2 small, or None if this machine cannot reach the checkpoint"""
    try:
        return load_adapter("gpt2-small")
    except Exception:
        return None

class AdapterTestCase(TestCase):
    adapter = None

    @classmethod
    def setUpClass(cls):
        cls.adapter = _adapter()
        if cls.adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")

class TestGoldenCapture(AdapterTestCase):
    @skipUnless(GOLDEN.exists(), f"no golden file at {GOLDEN}; regenerate with tests/stubs/refresh.py")
    def test_capture_still_produces_the_frozen_activations(self):
        expected = torch.load(GOLDEN)
        layers = self.adapter.cfg.layers(GOLDEN_FRACS)
        actual = self.adapter.capture(GOLDEN_PROMPTS, layers=layers)

        self.assertEqual(expected.shape, actual.shape)
        drift = (expected - actual).abs().max().item()
        self.assertLess(drift, 1e-3, f"capture drifted by {drift:.2e} from the golden file")

class TestCaptureShapes(AdapterTestCase):
    def test_last_position_collapses_the_sequence(self):
        layers = self.adapter.cfg.layers(GOLDEN_FRACS)
        activations = self.adapter.capture(GOLDEN_PROMPTS, layers=layers)
        self.assertEqual((4, 2, self.adapter.cfg.d_model), tuple(activations.shape))
        self.assertTrue(torch.isfinite(activations).all())

    def test_all_position_keeps_the_sequence(self):
        activations = self.adapter.capture(["one", "a rather longer prompt"], position=Position.ALL)
        batch, layer, seq, d_model = activations.shape
        self.assertEqual((2, 1, self.adapter.cfg.d_model), (batch, layer, d_model))
        self.assertGreater(seq, 1)

    def test_an_early_layer_is_not_a_late_layer(self):
        early, late = self.adapter.cfg.layers([0.1, 0.65])
        activations = self.adapter.capture(GOLDEN_PROMPTS, layers=[early, late])
        self.assertFalse(torch.allclose(activations[:, 0], activations[:, 1]))

    def test_defaults_to_the_configs_probe_layer(self):
        default = self.adapter.capture(["a prompt"])
        explicit = self.adapter.capture(["a prompt"], layers=[self.adapter.layer()])
        self.assertTrue(torch.allclose(default, explicit))

    def test_a_layer_this_model_does_not_have_is_refused(self):
        with self.assertRaises(ConfigError):
            self.adapter.capture(["a prompt"], layers=[self.adapter.cfg.n_layers])

    def test_capture_needs_something_to_capture(self):
        with self.assertRaises(ConfigError):
            self.adapter.capture([])

class TestBatching(AdapterTestCase):
    def test_chunking_does_not_change_the_result(self):
        """batch_size is a memory knob, not a numerical one"""
        prompts = [f"prompt number {index}" for index in range(20)]
        from dataclasses import replace

        chunked = load_adapter(replace(load_config("gpt2-small"), batch_size=3))
        self.assertTrue(torch.allclose(self.adapter.capture(prompts), chunked.capture(prompts), atol=1e-5))

class TestPadding(AdapterTestCase):
    def test_a_short_prompt_is_unaffected_by_a_long_neighbour(self):
        """Padding is neither averaged into MEAN nor mistaken for the last token"""
        together = ["hi", "a considerably longer prompt than the first one"]
        for position in (Position.LAST, Position.MEAN):
            with self.subTest(position=position.value):
                batched = self.adapter.capture(together, position=position)
                alone = self.adapter.capture(["hi"], position=position)
                self.assertTrue(torch.allclose(batched[0], alone[0], atol=1e-4))

class TestSteering(AdapterTestCase):
    def test_zero_strength_is_identical_to_no_steering(self):
        """The check that tells you the hook fires only where you think it does"""
        prompts = ["The bridge was"]
        baseline = self.adapter.generate(prompts, max_new_tokens=8)
        with self.adapter.steer(self.adapter.layer(), torch.randn(self.adapter.cfg.d_model), 0.0):
            steered = self.adapter.generate(prompts, max_new_tokens=8)
        self.assertEqual(baseline, steered)

    def test_a_real_strength_moves_the_activations(self):
        layer = self.adapter.layer()
        direction = torch.randn(self.adapter.cfg.d_model)
        baseline = self.adapter.capture(["The bridge was"], layers=[layer])
        with self.adapter.steer(layer, direction, 2.0):
            steered = self.adapter.capture(["The bridge was"], layers=[layer])
        self.assertFalse(torch.allclose(baseline, steered))

    def test_the_hook_is_removed_afterwards(self):
        baseline = self.adapter.capture(["The bridge was"])
        with self.adapter.steer(self.adapter.layer(), torch.randn(self.adapter.cfg.d_model), 2.0):
            pass
        self.assertTrue(torch.allclose(baseline, self.adapter.capture(["The bridge was"])))

class TestBackendRegistry(TestCase):
    def test_the_shipped_backend_is_registered(self):
        self.assertIn("transformers", BACKENDS)

    def test_an_unregistered_backend_says_what_is_available(self):
        with self.assertRaises(ConfigError) as caught:
            load_adapter("qwen3.5-27b")
        self.assertIn("transformers", str(caught.exception))

    def test_an_unknown_dtype_is_caught_before_the_download(self):
        from dataclasses import replace

        with self.assertRaises(ConfigError):
            load_adapter(replace(load_config("gpt2-small"), dtype="float8"))
