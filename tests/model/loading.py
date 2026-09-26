from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from safetensors.torch import load_file, save_file

from src.domains.lm.backend.loading import LoadError, checkpoint_files, stream

"""
The streaming loader is checked against `from_pretrained`, and its refusals on
checkpoints broken on purpose. A loader that silently kept one randomly
initialized tensor would produce a model that runs and is wrong, so the
refusals are the half that matters.

Everything here builds a tiny GPT-2 from its config and saves it, so no
download is involved; the comparison with `from_pretrained` needs a device.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(directory: Path):
    """A four-layer GPT-2 with random weights, saved as safetensors the way the hub would have it"""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=128, n_positions=32))
    model.save_pretrained(directory, safe_serialization=True)
    return model


class TestStream(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = tiny(self.root)

    def test_it_loads_what_from_pretrained_loads(self):
        from transformers import AutoModelForCausalLM

        streamed = stream(str(self.root), torch.float32, DEVICE)
        reference = AutoModelForCausalLM.from_pretrained(str(self.root), dtype=torch.float32).to(DEVICE)
        ours, theirs = streamed.state_dict(), reference.state_dict()
        self.assertEqual(set(theirs), set(ours))
        for key in theirs:
            self.assertTrue(torch.equal(theirs[key], ours[key]), key)

    def test_tied_embeddings_stay_one_tensor(self):
        streamed = stream(str(self.root), torch.float32, DEVICE)
        self.assertEqual(streamed.lm_head.weight.data_ptr(), streamed.get_input_embeddings().weight.data_ptr())

    def test_a_missing_tensor_is_refused_rather_than_left_random(self):
        path = self.root / "model.safetensors"
        tensors = load_file(path)
        dropped = next(key for key in tensors if "mlp.c_fc.weight" in key)
        del tensors[dropped]
        save_file(tensors, path, metadata={"format": "pt"})
        with self.assertRaisesRegex(LoadError, "not in the checkpoint"):
            stream(str(self.root), torch.float32, DEVICE)

    def test_a_tensor_of_the_wrong_shape_is_refused(self):
        path = self.root / "model.safetensors"
        tensors = load_file(path)
        key = next(key for key in tensors if "mlp.c_fc.weight" in key)
        tensors[key] = torch.zeros(3, 3)
        save_file(tensors, path, metadata={"format": "pt"})
        with self.assertRaisesRegex(LoadError, "is \\(3, 3\\)"):
            stream(str(self.root), torch.float32, DEVICE)

    def test_it_is_none_for_a_checkpoint_that_is_not_safetensors(self):
        (self.root / "model.safetensors").unlink()
        self.assertIsNone(checkpoint_files(str(self.root)))


class TestModelKey(TestCase):
    def test_the_two_renames_and_nothing_else(self):
        from src.domains.lm.backend.loading import model_key

        wanted = {"model.layers.0.mlp.up_proj.weight", "transformer.h.0.mlp.c_fc.weight"}
        self.assertEqual("model.layers.0.mlp.up_proj.weight",
                         model_key("model.language_model.layers.0.mlp.up_proj.weight", wanted))
        self.assertEqual("transformer.h.0.mlp.c_fc.weight", model_key("h.0.mlp.c_fc.weight", wanted, "transformer"))
        self.assertIsNone(model_key("model.visual.blocks.0.attn.weight", wanted, "model"))
