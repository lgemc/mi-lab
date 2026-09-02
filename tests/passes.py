"""Guards on the shared forward pass every hook-based scan is built on.

A hook is only as right as the pass under it: the mask has to reach the hook
before the model runs, a padded row's tokens have to line up with its trace
position for position, and the attention module has to be found by which
module owns the projection rather than by a name that differs per family.
"""

from unittest import TestCase

import torch

from src.model.passes import attention_of, forward_batches, hooked, module_owning, token_strings

from .stubs.model import shared_adapter


class TestOnline(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = adapter

    def test_batches_yield_ids_and_masks_of_the_same_shape_and_see_the_mask_first(self):
        texts = ["a short one", "a considerably longer sentence than the first", "third"]
        seen = []
        batches = list(forward_batches(self.adapter, texts, batch_size=2, on_batch=lambda mask: seen.append(mask)))
        self.assertEqual(2, len(batches))
        self.assertEqual(2, len(seen))
        for (ids, mask), early in zip(batches, seen, strict=True):
            self.assertEqual(tuple(ids.shape), tuple(mask.shape))
            self.assertTrue(torch.equal(mask, early))

    def test_token_strings_decode_only_the_real_tokens_and_rebuild_the_text(self):
        texts = ["a short one", "a considerably longer sentence than the first"]
        (ids, mask), = forward_batches(self.adapter, texts, batch_size=2)
        tokens = token_strings(self.adapter, ids, mask, row=0)
        self.assertEqual(int(mask[0].sum()), len(tokens))
        self.assertEqual(texts[0], "".join(tokens))

    def test_the_attention_module_owns_the_projection(self):
        attention = attention_of(self.adapter, 5)
        self.assertIn(self.adapter.projections[5], list(attention.children()))
        self.assertIs(attention, module_owning(self.adapter, 5, self.adapter.projections[5]))

    def test_hooked_removes_its_handles_even_when_the_body_raises(self):
        calls = []
        handle = self.adapter.blocks[0].register_forward_hook(lambda module, args, output: calls.append(1))
        with self.assertRaises(RuntimeError), hooked([handle]):
            raise RuntimeError("mid-pass")
        list(forward_batches(self.adapter, ["hello"]))
        self.assertEqual([], calls)
