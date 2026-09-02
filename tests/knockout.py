"""Guards on mean ablation: the means belong to one model, and an ablation is a difference.

The offline half is the cache discipline: means sliced to a band are the same
numbers rather than a new capture, and a cache read onto another model is a
refusal rather than a plausible number. The online half runs the hooks on
GPT-2 small: a captured mean has the head and MLP geometry it claims, ablating
nothing leaves the logits exactly alone, ablating a whole layer moves them,
and the extraction of a component is its module's parameters and no others.
"""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from src.methods.knockout import (
    KnockoutError,
    Means,
    ablate,
    cached_means,
    capture_means,
    component_module,
    extract,
    geometry,
    translate,
)

from .stubs.model import shared_adapter

TEXTS = ["Spanish: hola\nEnglish: hello\n\nSpanish: gato\nEnglish:", "Spanish: perro\nEnglish:"]


def synthetic(layers, tokens=10, model=None) -> Means:
    return Means(heads={layer: torch.full((2, 3), float(layer)) for layer in layers},
                 mlps={layer: torch.full((6,), float(layer)) for layer in layers}, tokens=tokens, model=model)


class TestMeans(TestCase):
    def test_a_slice_is_the_same_numbers_over_fewer_layers(self):
        means = synthetic([0, 1, 2])
        part = means.slice([1, 2])
        self.assertEqual([1, 2], part.layers)
        self.assertTrue(torch.equal(means.heads[2], part.heads[2]))
        self.assertEqual(means.tokens, part.tokens)

    def test_a_slice_outside_the_capture_is_refused(self):
        with self.assertRaises(KnockoutError):
            synthetic([0, 1]).slice([2])

    def test_it_round_trips_through_disk_with_its_stamp(self):
        stamp = {"id": "tiny", "n_layers": 3, "n_heads": 2, "d_model": 6}
        with TemporaryDirectory() as directory:
            path = synthetic([0, 1], model=stamp).save(Path(directory) / "means.pt")
            back = Means.load(path)
        self.assertEqual(stamp, back.model)
        self.assertEqual([0, 1], back.layers)
        self.assertEqual(path, back.source)
        self.assertTrue(torch.equal(torch.full((6,), 1.0), back.mlps[1]))


class TestOnline(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = adapter
        with contextlib.redirect_stdout(io.StringIO()):
            cls.means = capture_means(adapter, TEXTS, layers=[3, 4])

    def test_a_capture_has_the_geometry_it_claims(self):
        cfg = self.adapter.cfg
        self.assertEqual([3, 4], self.means.layers)
        self.assertEqual((cfg.n_heads, cfg.d_head), tuple(self.means.heads[3].shape))
        self.assertEqual((cfg.d_model,), tuple(self.means.mlps[4].shape))
        self.assertEqual(geometry(self.adapter), self.means.model)
        self.assertGreater(self.means.tokens, 0)

    def test_means_captured_on_another_model_are_refused(self):
        foreign = Means(heads=self.means.heads, mlps=self.means.mlps, tokens=1,
                        model={**geometry(self.adapter), "id": "other"})
        with self.assertRaises(KnockoutError):
            foreign.check(self.adapter)
        self.assertIs(self.means, self.means.check(self.adapter))

    def test_the_cache_is_read_back_or_sliced_from_a_superset(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            wide = Path(directory) / "means-3-4.pt"
            self.means.save(wide)
            narrow = Path(directory) / "means-4.pt"
            sliced = cached_means(self.adapter, TEXTS, [4], narrow, siblings=[wide])
            self.assertEqual([4], sliced.layers)
            self.assertTrue(torch.equal(self.means.mlps[4], sliced.mlps[4]))
            self.assertFalse(narrow.exists())      # a slice is not a new capture

    def test_ablating_nothing_changes_nothing_and_a_layer_changes_something(self):
        prompts = TEXTS[:1]
        clean = self.adapter.logits(prompts)
        with ablate(self.adapter, self.means, []):
            self.assertTrue(torch.equal(clean, self.adapter.logits(prompts)))
        with ablate(self.adapter, self.means, ["mlp:3", "heads:4"]):
            moved = self.adapter.logits(prompts)
        self.assertFalse(torch.allclose(clean, moved))
        self.assertTrue(torch.equal(clean, self.adapter.logits(prompts)))

    def test_a_component_outside_the_means_is_refused(self):
        with self.assertRaises(KnockoutError), ablate(self.adapter, self.means, ["mlp:0"]):
            pass

    def test_translate_cuts_each_completion_at_its_first_line(self):
        with contextlib.redirect_stdout(io.StringIO()):
            done = translate(self.adapter, TEXTS, chunk=1, max_new_tokens=8)
        self.assertEqual(2, len(done))
        self.assertTrue(all("\n" not in text for text in done))

    def test_extraction_is_the_module_and_only_the_module(self):
        tensors, entries = extract(self.adapter, ["mlp:2", "heads:2"])
        self.assertEqual(["mlp:2", "heads:2"], [entry["component"] for entry in entries])
        expected = sum(t.numel() for t in component_module(self.adapter, "mlp:2").parameters())
        self.assertEqual(expected, entries[0]["n_parameters"])
        self.assertTrue(all(key.startswith(("mlp:2/", "heads:2/")) for key in tensors))
        self.assertTrue(all(t.device.type == "cpu" for t in tensors.values()))
        with self.assertRaises(KnockoutError):
            component_module(self.adapter, "head:2:1")
