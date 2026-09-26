from unittest import TestCase

import torch

from src.core.config import ConfigError, ModelConfig
from src.domains.lm.analysis.lens import logit_lens
from src.domains.lm.backend import TransformersAdapter

"""
A hybrid stack -- linear-attention blocks between softmax-attention ones -- on
a model small enough to build from its config in a second, with random weights.

What is under test is the adapter's layout knowledge, not the model: that the
residual-stream half works on every block, that the head-level half works on
the attention blocks and refuses the others by name, and that the sizes come
from the nested text config. Random weights are enough for all of it, and they
keep the 27B checkpoint this exists for off a test machine.

Needs transformers 5, which has `qwen3_5`; the default environment is on 4 for
COMET and circuit-tracer, so this skips there and runs under
`uv run --no-group comet --group hybrid`.
"""

def tiny_hybrid():
    """Four blocks, the last one softmax attention, like every fourth block of Qwen3.6"""
    try:
        from transformers import GPT2TokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError:
        return None
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    except OSError:
        return None
    tokenizer.pad_token = tokenizer.eos_token
    config = Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=8, linear_key_head_dim=8, linear_value_head_dim=8,
        full_attention_interval=4, pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).eval()
    cfg = ModelConfig(id="tiny-hybrid", backend="transformers", hf_name="tiny-hybrid", batch_size=2,
                      max_new_tokens=3)
    return TransformersAdapter(cfg, model, tokenizer)

class TestHybrid(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_hybrid()
        if cls.adapter is None:
            raise cls.skipTest(cls, "transformers has no qwen3_5 (needs the `hybrid` group) or gpt2 is unreachable")

    def test_only_the_softmax_blocks_have_heads(self):
        self.assertEqual([3], self.adapter.attention_layers)

    def test_sizes_come_from_the_text_config(self):
        self.assertEqual((4, 64, 4), (self.adapter.cfg.n_layers, self.adapter.cfg.d_model, self.adapter.cfg.n_heads))

    def test_capture_reads_every_block(self):
        residual = self.adapter.capture(["one two three", "four"], layers=range(4))
        self.assertEqual((2, 4, 64), tuple(residual.shape))
        self.assertTrue(torch.isfinite(residual).all())

    def test_the_lens_at_the_top_is_the_model(self):
        """The final block's residual through the final norm and unembedding IS the next-token distribution"""
        prompts = ["one two three", "four"]
        top = logit_lens(self.adapter, prompts, layers=[3], top_k=1)
        predicted = self.adapter.logits(prompts).argmax(dim=-1)
        for row, readouts in enumerate(top):
            self.assertEqual(self.adapter.tokenizer.decode([int(predicted[row])]), readouts[0].tokens[0])

    def test_generation_runs_through_the_linear_cache(self):
        answers = self.adapter.generate(["one two three", "four"])
        self.assertEqual(2, len(answers))

    def test_heads_are_read_on_the_attention_block(self):
        heads = self.adapter.head_outputs(["one two three"], layers=[3])
        self.assertEqual(4, heads.shape[2])

    def test_heads_on_a_linear_block_are_refused_by_name(self):
        with self.assertRaisesRegex(ConfigError, "layer 1 .* linear attention"):
            self.adapter.head_outputs(["one two three"], layers=[1])

    def test_edges_are_refused_rather_than_listing_heads_that_do_not_exist(self):
        with self.assertRaisesRegex(ConfigError, "linear attention"):
            self.adapter.edges()
