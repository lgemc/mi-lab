"""The receipts on edge ablation, because a plausible wrong graph is the failure mode.

An edge intervention is arithmetic on a sum, so it is only as good as the claim
that the sum reproduces the forward pass. Two of these tests exist because they
failed: the embedding had no live value to subtract, and reconstructing a head's
write re-entered the hook capturing that projection's input. Both produced
numbers rather than errors.
"""

from unittest import TestCase

import torch

from src.core.config import ConfigError
from src.data.tasks import build_task
from src.model.adapter import require_circuits
from src.model.backends.transformers.layout import _attention_norm

from ..stubs.model import shared_adapter

DESTINATION = 5

class EdgeTestCase(TestCase):
    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        # length-aligned by construction, which an edge needs: it is a term in one
        # position's residual sum, so both runs must have the same positions
        task = build_task("ioi", cls.adapter, size=4, seed=0)
        cls.clean, cls.corrupted = list(task.clean), list(task.corrupted)

    def upstream(self, layer=DESTINATION):
        return (["embed"]
                + [f"head:{at}:{head}" for at in range(layer)
                   for head in range(self.adapter.cfg.n_heads)]
                + [f"bias:{at}" for at in range(layer)] + [f"mlp:{at}" for at in range(layer)])

class TestDecomposition(EdgeTestCase):
    def test_the_writes_sum_to_what_the_destinations_read(self):
        """Nothing built on this graph means anything if it is not the model's graph"""
        sources = self.adapter.residual_sources(self.clean)
        gap = self.adapter.residual_remainder(sources)
        self.assertLess(gap["relative"], 1e-4, "the residual decomposition does not reproduce the forward pass")

class TestEdgePatch(EdgeTestCase):
    def test_ablating_no_edges_is_byte_identical(self):
        off = self.adapter.residual_sources(self.corrupted)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, []):
            self.assertTrue(torch.equal(base, self.adapter.logits(self.clean)))

    def test_ablating_toward_the_run_itself_changes_nothing(self):
        """The identity that caught both bugs: a no-op written the long way round

        Ablating every edge toward the values the run already has must be a
        no-op. It was not, twice -- once because `embed` had no live write to
        subtract and its counterfactual landed on top of the real one, once
        because reconstructing a head re-entered the pre-hook capturing that
        projection's input and left every later head reading zeros. Both moved
        the logits by more than two.
        """
        itself = self.adapter.residual_sources(self.clean)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(itself, [(s, f"attn:{DESTINATION}") for s in self.upstream()]):
            patched = self.adapter.logits(self.clean)
        self.assertLess(float((patched - base).abs().max()), 1e-2)

    def test_every_edge_into_a_destination_equals_replacing_its_input(self):
        """Edge ablation has to agree with the coarser thing it refines"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.adapter.edge_patch(off, [(s, f"attn:{DESTINATION}") for s in self.upstream()]):
            via_edges = self.adapter.logits(self.clean)
        donor = off["attention_in"][:, DESTINATION]
        handle = _attention_norm(self.adapter.blocks[DESTINATION], DESTINATION).register_forward_pre_hook(
            lambda module, args: (donor.to(args[0].device, args[0].dtype), *args[1:])
        )
        try:
            via_replacement = self.adapter.logits(self.clean)
        finally:
            handle.remove()
        self.assertLess(float((via_edges - via_replacement).abs().max()), 1e-2)

    def test_one_edge_moves_the_answer(self):
        off = self.adapter.residual_sources(self.corrupted)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:0:0", f"attn:{DESTINATION}")]):
            self.assertFalse(torch.equal(base, self.adapter.logits(self.clean)))

    def test_edges_compose_rather_than_overwrite(self):
        """Two edges must differ from either alone, or later edits are ignoring earlier ones"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.adapter.edge_patch(off, [("head:0:0", "attn:5")]):
            first = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:1:0", "attn:5")]):
            second = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:0:0", "attn:5"), ("head:1:0", "attn:5")]):
            both = self.adapter.logits(self.clean)
        self.assertFalse(torch.equal(both, first))
        self.assertFalse(torch.equal(both, second))

    def test_a_non_causal_edge_is_refused(self):
        """An edge that does not exist must raise, not quietly ablate nothing"""
        off = self.adapter.residual_sources(self.corrupted)
        for edge in (("mlp:5", "attn:5"), ("head:7:0", "attn:3"), ("mlp:9", "mlp:2")):
            with self.subTest(edge=edge), self.assertRaises(ConfigError), \
                    self.adapter.edge_patch(off, [edge]):
                pass

    def test_runs_of_different_lengths_are_refused(self):
        """A mixture of positions is a mixture of different tokens"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.assertRaises(ConfigError), self.adapter.edge_patch(off, [("head:0:0", "attn:5")]):
            self.adapter.logits(["a much shorter prompt"])

class TestEdgeGate(EdgeTestCase):
    """The gate is what a pruned circuit runs on, and it had no receipts of its own

    `edge_patch` is covered above and `edge_gate` was covered by neither: it
    reuses the same helpers through a different arithmetic, and the run that
    exercised it first was a whole-model prune, where a wrong reconstruction
    reads as a bad circuit rather than as a bug.
    """

    def gates(self, value, edges=None):
        device = self.adapter.model.device
        return {edge: torch.full((), value, device=device)
                for edge in (edges if edges is not None else self.adapter.edges())}

    def test_every_gate_open_is_the_model_itself(self):
        """g = 1 subtracts nothing, so the identity has to be exact and not merely close"""
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_gate(self.gates(1.0)):
            self.assertTrue(torch.equal(base, self.adapter.logits(self.clean)))

    def test_a_shut_gate_is_the_same_edge_alone_or_among_all_of_them(self):
        """The receipt on reusing one reconstruction for every destination that reads it

        A source's write is rebuilt once per forward pass and shared across the
        destinations gating it -- 505 rebuilds rather than 14140 on a 28-layer
        model, which is what makes an edge run affordable there. It is only
        sound if the shared tensor is the one each destination would have built
        for itself, and the difference would be a plausible wrong graph rather
        than an error: closing the edge inside a full gate dict has to land
        exactly where closing it on its own does.
        """
        every = self.adapter.edges()
        for edge in (every[5], every[len(every) // 2], ("mlp:3", f"attn:{DESTINATION}")):
            with self.subTest(edge=edge):
                with self.adapter.edge_gate(self.gates(0.0, [edge])):
                    alone = self.adapter.logits(self.clean)
                among = self.gates(1.0)
                among[edge] = torch.zeros((), device=self.adapter.model.device)
                with self.adapter.edge_gate(among):
                    together = self.adapter.logits(self.clean)
                self.assertTrue(torch.equal(alone, together))
                self.assertFalse(torch.equal(alone, self.adapter.logits(self.clean)),
                                 "the edge was closed and nothing moved")

    def test_the_gate_survives_a_kv_cache(self):
        """The receipt the server's edge backbone rests on

        `edge_gate`'s hooks capture per-forward activations and rebuild each
        source's write from them. Incremental decoding hands them one position
        at a time and reads every earlier one out of the KV cache, where the
        keys and values were computed on earlier steps -- under the edits, but
        not recomputed. Whether that composes is not obvious from the code, and
        the failure would be a circuit that generates plausible different text
        over HTTP than it does in the lab. So it is measured: greedy decoding
        with the cache must produce the same tokens as an uncached loop, under
        a mask that actually closes something.
        """
        every = self.adapter.edges()
        device = self.adapter.model.device
        one = torch.ones((), device=device)
        zero = torch.zeros((), device=device)
        # every third edge shut: enough to move the output well away from the
        # model's own, which is what makes agreement meaningful
        gates = {edge: (zero if index % 3 == 0 else one)
                 for index, edge in enumerate(every)}
        prompt, steps = "The capital of France is", 4

        self.adapter.tokenizer.padding_side = "right"
        encoded = self.adapter.tokenizer([prompt], return_tensors="pt").to(device)
        ids = encoded["input_ids"]

        uncached = []
        current = ids
        for _ in range(steps):
            with self.adapter.edge_gate(gates):
                logits = self.adapter.model(current, use_cache=False).logits
            nxt = logits[0, -1].argmax().view(1, 1)
            uncached.append(int(nxt))
            current = torch.cat([current, nxt], dim=1)

        with self.adapter.edge_gate(gates):
            generated = self.adapter.model.generate(
                **encoded, max_new_tokens=steps, do_sample=False, use_cache=True,
                pad_token_id=self.adapter.tokenizer.eos_token_id)
        cached = generated[0, ids.shape[1]:].tolist()

        self.assertEqual(uncached, cached,
                         "edge_gate is not cache-safe; the serve backbone must force use_cache=False")

    def test_both_paths_through_the_gate_agree(self):
        """The inference path skips work the training path cannot, and must still agree

        `edge_gate` has two implementations of one equation. Under
        `torch.enable_grad` it accumulates `(1 - g) * live` one edge at a time,
        because every gate needs a node in the graph for its gradient -- open
        ones included, whose term is exactly zero. Under `no_grad` it drops the
        open edges (a provable no-op) and reduces the rest in one sum, which is
        ~3 kernel launches per destination rather than ~3 per edge and is the
        difference between 13x the full model and single digits when serving.

        Two implementations of one equation is two chances to be wrong, so this
        asserts they are the same equation. Bit-exactness is not claimed and is
        not the point: a sum over a stack reduces in a different order than a
        running subtraction, which is float noise. Order 1 is a real
        disagreement and is what this catches.
        """
        every = self.adapter.edges()
        device = self.adapter.model.device
        one = torch.ones((), device=device)
        zero = torch.zeros((), device=device)
        for label, gates in (
            ("binary", {edge: (zero if index % 3 == 0 else one)
                        for index, edge in enumerate(every)}),
            # fractional gates never occur when serving a circuit, but the
            # weighted branch exists for them and is otherwise untested
            ("fractional", {edge: (torch.full((), 0.25, device=device) if index % 3 == 0 else one)
                            for index, edge in enumerate(every)}),
        ):
            with self.subTest(gates=label):
                with torch.enable_grad(), self.adapter.edge_gate(gates):
                    training = self.adapter.model(
                        self.adapter.tokenizer(self.clean, return_tensors="pt",
                                               padding=True).to(device)["input_ids"],
                        use_cache=False).logits.detach()
                with torch.no_grad(), self.adapter.edge_gate(gates):
                    serving = self.adapter.model(
                        self.adapter.tokenizer(self.clean, return_tensors="pt",
                                               padding=True).to(device)["input_ids"],
                        use_cache=False).logits
                gap = (training - serving).abs().max() / training.abs().max()
                self.assertLess(float(gap), 1e-4, f"{label}: the two paths disagree")
