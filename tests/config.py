import re
from pathlib import Path
from unittest import TestCase

from src.core.config import (
    CONFIG_DIR,
    ConfigError,
    ModelConfig,
    from_mapping,
    load_config,
    presets,
)

"""
Config carries the one invariant the whole framework rests on: a depth
fraction resolves to the same *place* in any model. These tests need no
checkpoint, so they are the ones that must never be slow or skipped.
"""

def _cfg(**overrides) -> ModelConfig:
    fields = dict(id="test", backend="transformers", hf_name="gpt2", n_layers=12, d_model=768)
    fields.update(overrides)
    return ModelConfig(**fields)

class TestLayerResolution(TestCase):
    def test_two_thirds_depth_is_the_same_place_in_both_models(self):
        self.assertEqual(8, _cfg(n_layers=12).layer(0.65))
        self.assertEqual(42, _cfg(n_layers=64).layer(0.65))

    def test_defaults_to_the_configs_own_probe_fraction(self):
        self.assertEqual(_cfg(probe_layer_frac=0.5).layer(), _cfg().layer(0.5))

    def test_full_depth_clamps_to_the_last_layer(self):
        self.assertEqual(11, _cfg(n_layers=12).layer(1.0))
        self.assertEqual(0, _cfg(n_layers=12).layer(0.0))

    def test_layers_keeps_order_and_drops_duplicates(self):
        self.assertEqual([1, 8, 11], _cfg().layers([0.1, 0.65, 1.0]))
        self.assertEqual([8], _cfg().layers([0.65, 0.66]))

    def test_sweep_spans_the_whole_depth(self):
        self.assertEqual([0, 3, 6, 9, 11], _cfg().sweep(5))
        self.assertEqual([6], _cfg().sweep(1))

    def test_unresolved_config_refuses_to_guess(self):
        with self.assertRaises(ConfigError):
            _cfg(n_layers=None).layer(0.65)

class TestConfigValidation(TestCase):
    def test_an_absolute_index_in_the_fraction_field_is_rejected(self):
        with self.assertRaises(ConfigError):
            _cfg(probe_layer_frac=8)

    def test_batch_size_must_be_usable(self):
        with self.assertRaises(ConfigError):
            _cfg(batch_size=0)

    def test_sizes_that_disagree_with_the_checkpoint_are_an_error(self):
        with self.assertRaises(ConfigError):
            _cfg(d_model=512).with_sizes(n_layers=12, d_model=768)

    def test_sizes_are_stamped_in_when_the_config_left_them_open(self):
        resolved = _cfg(n_layers=None, d_model=None).with_sizes(n_layers=6, d_model=512)
        self.assertTrue(resolved.is_resolved)
        self.assertEqual(4, resolved.layer(0.65))

    def test_a_mistyped_key_is_not_silently_dropped(self):
        with self.assertRaises(ConfigError):
            from_mapping({"id": "x", "backend": "transformers", "hf_name": "gpt2", "batchsize": 4})

    def test_id_falls_back_to_the_file_name(self):
        cfg = from_mapping({"backend": "transformers", "hf_name": "gpt2"}, default_id="from-file")
        self.assertEqual("from-file", cfg.id)

class TestShippedConfigs(TestCase):
    def test_every_shipped_config_parses(self):
        self.assertTrue(presets(), f"no configs found in {CONFIG_DIR}")
        for name in presets():
            with self.subTest(name):
                cfg = load_config(name)
                self.assertEqual(name, cfg.id)
                self.assertTrue(0.0 <= cfg.probe_layer_frac <= 1.0)

    def test_a_config_can_also_be_given_as_a_path(self):
        path = CONFIG_DIR / "gpt2-small.yaml"
        self.assertEqual(load_config("gpt2-small"), load_config(str(path)))

    def test_no_shipped_config_hardcodes_a_size(self):
        for name in presets():
            with self.subTest(name):
                cfg = load_config(name)
                self.assertIsNone(cfg.n_layers)
                self.assertIsNone(cfg.d_model)

    def test_an_unknown_name_says_what_is_available(self):
        with self.assertRaises(ConfigError) as caught:
            load_config("gpt2-enormous")
        self.assertIn("gpt2-small", str(caught.exception))

class TestNoHardcodedModelFacts(TestCase):
    # 1024 is deliberately absent: it is a plausible d_model, but it is also the
    # divisor in every KiB conversion, and a check that cries wolf gets deleted.
    SIZE_LITERALS = ("768", "1600", "2048", "4096", "5120")

    def test_the_source_never_names_a_size(self):
        """The check from Module 0: grep for 768 and 5120, expect nothing

        Sizes live in checkpoints and get stamped into configs. A literal one
        in the source is how an experiment silently fuses to one model.
        """
        for path in Path("src").rglob("*.py"):
            source = path.read_text()
            for number in self.SIZE_LITERALS:
                with self.subTest(path=str(path), number=number):
                    self.assertIsNone(re.search(rf"\b{number}\b", source))
