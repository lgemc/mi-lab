import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from src.core.config import ModelConfig, Position
from src.core.dataset import DatasetError, LabeledPrompts
from src.core.prompts import parse
from src.core.torchdata import (
    ActivationDataset,
    PromptDataset,
    StreamingPrompts,
    activation_loader,
    capture_dataset,
    collate_prompts,
    prompt_loader,
)

"""
Two things are worth testing here and neither of them is torch.

The first is that prompts come out of a loader as text: a collate that
stringifies, tokenizes or stacks would be found by the model much later and
look like a modelling result.

The second is that a capture and its labels stay lined up. Rows arrive in
batches, come back as chunks and get concatenated, and every one of those
steps is a chance for row i of the activations to stop being example i of the
file -- which no shape check would catch, because the shapes stay right.
"""

SET = """\
name: tiny
labels: bad, good

+ aaa
- bb
+ cccc
- d
+ ee
"""

class _StubAdapter:
    """A ModelAdapter that returns an activation naming the prompt it saw

    No model, no tokenizer: the point is to check the plumbing around capture,
    and a stub makes the alignment checkable because every prompt has an
    activation only it could have produced.
    """

    def __init__(self, batch_size: int = 2):
        self.cfg = ModelConfig(
            id="stub", backend="none", hf_name="none",
            n_layers=4, d_model=3, batch_size=batch_size,
        )
        self.batches = []

    def layer(self, frac=None) -> int:
        return self.cfg.layer(frac)

    def capture(self, prompts, layers=None, position=Position.LAST) -> torch.Tensor:
        self.batches.append(list(prompts))
        layers = list(layers) if layers is not None else [self.layer()]
        return torch.tensor(
            [[[float(len(text)), float(ord(text[0])), float(layer)] for layer in layers] for text in prompts]
        )

def _capture_of(text: str, layer: int) -> list:
    return [float(len(text)), float(ord(text[0])), float(layer)]

class TestPromptDataset(TestCase):
    def setUp(self):
        self.data = parse(SET)
        self.dataset = PromptDataset(self.data)

    def test_it_yields_text_and_label_in_file_order(self):
        self.assertEqual(5, len(self.dataset))
        self.assertEqual(("aaa", 1), self.dataset[0])
        self.assertEqual(("bb", 0), self.dataset[1])

    def test_it_keeps_the_dataset_behind_it(self):
        """The name, label names and groups have to survive into a capture"""
        self.assertEqual("tiny", self.dataset.data.name)
        self.assertEqual(("bad", "good"), self.dataset.data.label_names)

class TestCollate(TestCase):
    def test_text_stays_text(self):
        texts, labels = collate_prompts([("a", 1), ("b", 0)])
        self.assertEqual(["a", "b"], texts)
        self.assertTrue(torch.equal(torch.tensor([1, 0]), labels))
        self.assertEqual(torch.long, labels.dtype)

class TestPromptLoader(TestCase):
    def setUp(self):
        self.data = parse(SET)

    def test_batches_cover_every_example_once_in_order(self):
        loader = prompt_loader(self.data, batch_size=2)
        seen = [text for texts, _ in loader for text in texts]
        self.assertEqual(self.data.texts, seen)

    def test_the_last_batch_is_the_short_one(self):
        sizes = [len(texts) for texts, _ in prompt_loader(self.data, batch_size=2)]
        self.assertEqual([2, 2, 1], sizes)

    def test_shuffling_is_reproducible_from_a_seed(self):
        first = [text for texts, _ in prompt_loader(self.data, batch_size=2, shuffle=True, seed=7) for text in texts]
        again = [text for texts, _ in prompt_loader(self.data, batch_size=2, shuffle=True, seed=7) for text in texts]
        self.assertEqual(first, again)
        self.assertEqual(sorted(self.data.texts), sorted(first))

    def test_labels_ride_along_with_their_own_text(self):
        for texts, labels in prompt_loader(self.data, batch_size=2, shuffle=True, seed=1):
            for text, label in zip(texts, labels, strict=True):
                self.assertEqual(self.data.labels[self.data.texts.index(text)], int(label))

class TestStreamingPrompts(TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".prompts", delete=False)
        handle.write(SET)
        handle.close()
        self.path = handle.name

    def test_it_reads_the_whole_file_in_order(self):
        self.assertEqual(parse(SET).texts, [text for text, _ in StreamingPrompts(self.path)])

    def test_workers_shard_the_file_without_dropping_or_repeating(self):
        """Each worker takes every nth example, so the union is the file exactly once"""
        loader = prompt_loader(StreamingPrompts(self.path), batch_size=2, num_workers=2)
        seen = [text for texts, _ in loader for text in texts]
        self.assertEqual(sorted(parse(SET).texts), sorted(seen))

    def test_it_refuses_to_be_shuffled(self):
        with self.assertRaises(DatasetError):
            prompt_loader(StreamingPrompts(self.path), shuffle=True)

    def test_a_missing_file_is_refused_at_construction(self):
        with self.assertRaises(DatasetError):
            StreamingPrompts(str(Path(tempfile.gettempdir()) / "definitely-not-here.prompts"))

class TestActivationDataset(TestCase):
    def setUp(self):
        self.data = ActivationDataset(
            activations=torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
            labels=[1, 0, 1, 0],
            layers=[2, 5],
            model_id="stub",
            name="tiny",
        )

    def test_it_yields_an_activation_and_its_label(self):
        activation, label = self.data[1]
        self.assertEqual((2, 3), tuple(activation.shape))
        self.assertEqual(0, label)

    def test_layers_are_addressed_by_the_models_index_not_the_tensors(self):
        one = self.data.at(5)
        self.assertEqual([5], one.layers)
        self.assertTrue(torch.equal(self.data.activations[:, 1:2], one.activations))

    def test_asking_for_a_layer_that_was_not_captured_says_which_were(self):
        with self.assertRaises(DatasetError) as caught:
            self.data.at(3)
        self.assertIn("[2, 5]", str(caught.exception))

    def test_mismatched_labels_are_refused(self):
        with self.assertRaises(DatasetError):
            ActivationDataset(activations=torch.zeros(4, 2, 3), labels=[1, 0], layers=[2, 5], model_id="stub")

    def test_undeclared_layers_are_refused(self):
        """A tensor whose layer axis does not match its layer list has lost its provenance"""
        with self.assertRaises(DatasetError):
            ActivationDataset(activations=torch.zeros(4, 2, 3), labels=[1, 0, 1, 0], layers=[2], model_id="stub")

    def test_saving_keeps_the_provenance(self):
        path = str(Path(tempfile.mkdtemp()) / "capture.pt")
        self.data.save(path)
        again = ActivationDataset.load(path)
        self.assertTrue(torch.equal(self.data.activations, again.activations))
        self.assertEqual([2, 5], again.layers)
        self.assertEqual("stub", again.model_id)
        self.assertEqual("tiny", again.name)

    def test_a_loader_over_activations_batches_tensors(self):
        activations, labels = next(iter(activation_loader(self.data, batch_size=2, shuffle=False)))
        self.assertEqual((2, 2, 3), tuple(activations.shape))
        self.assertEqual((2,), tuple(labels.shape))

class TestCaptureDataset(TestCase):
    def setUp(self):
        self.adapter = _StubAdapter(batch_size=2)
        self.data = parse(SET)

    def test_every_row_is_the_activation_of_its_own_prompt(self):
        captured = capture_dataset(self.adapter, self.data, layers=[1, 3])
        self.assertEqual((5, 2, 3), tuple(captured.activations.shape))
        for index, text in enumerate(self.data.texts):
            self.assertEqual(_capture_of(text, 1), captured.activations[index, 0].tolist())
            self.assertEqual(_capture_of(text, 3), captured.activations[index, 1].tolist())
        self.assertEqual(self.data.labels, captured.labels)

    def test_it_captures_in_batches_of_the_configs_size(self):
        capture_dataset(self.adapter, self.data)
        self.assertEqual([2, 2, 1], [len(batch) for batch in self.adapter.batches])

    def test_provenance_comes_from_the_adapter_and_the_dataset(self):
        captured = capture_dataset(self.adapter, parse("+ x\n  - y\n"), layers=[0])
        self.assertEqual("stub", captured.model_id)
        self.assertEqual("last", captured.position)
        self.assertEqual([0, 0], captured.groups)

    def test_the_default_layer_is_the_configs_depth_fraction(self):
        self.assertEqual([self.adapter.cfg.layer()], capture_dataset(self.adapter, self.data).layers)

    def test_whole_sequences_are_refused_with_a_reason(self):
        """Position.ALL has a per-batch sequence length, so the batches do not concatenate"""
        with self.assertRaises(DatasetError) as caught:
            capture_dataset(self.adapter, self.data, position=Position.ALL)
        self.assertIn("adapter.capture", str(caught.exception))

    def test_a_shuffled_loader_drops_the_groups_rather_than_misaligning_them(self):
        loader = prompt_loader(self.data, batch_size=2, shuffle=True, seed=3)
        captured = capture_dataset(self.adapter, loader)
        self.assertIsNone(captured.groups)
        for index, label in enumerate(captured.labels):
            row = captured.activations[index, 0].tolist()
            text = self.data.texts[[_capture_of(t, self.adapter.cfg.layer()) for t in self.data.texts].index(row)]
            self.assertEqual(self.data.labels[self.data.texts.index(text)], label)

    def test_a_capture_feeds_a_probe_straight_out(self):
        captured = capture_dataset(self.adapter, LabeledPrompts(texts=["aa", "bb", "cc", "dd"], labels=[1, 0, 1, 0]))
        activations, labels = captured.at(captured.layers[0]).tensors()
        self.assertEqual((4, 1, 3), tuple(activations.shape))
        self.assertEqual([1, 0, 1, 0], labels)
