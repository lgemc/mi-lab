"""Guards on reading a learned mask back as a circuit, and on loading it into a model.

The synthetic half builds a mask by hand and checks the reduction to the
component vocabulary: which layer a parameter belongs to, which projection it
is (GPT-2's `c_proj` is both an attention and an MLP name, and the branch has
to decide), and the counts per component and per layer. The online half loads
a mask into GPT-2 small and checks that the weights come back exactly, which
is the promise every sheaf evaluation rests on.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from src.methods.gates import (
    GateError,
    budget,
    by_layer,
    circuit_loaded,
    circuit_path,
    kind_of,
    layer_of,
    load_circuit,
    open_count,
    pack,
    parse_layers,
    per_component,
    ranked,
    summary,
    unpack,
)
from src.methods.sheaves import gateable

from .stubs.model import shared_adapter


def mask(*shape, open_share: float) -> torch.Tensor:
    """Gate logits with the first `open_share` of the entries positive and the rest negative"""
    logits = -torch.ones(*shape)
    flat = logits.view(-1)
    flat[: round(open_share * flat.numel())] = 1.0
    return logits


class TestNames(TestCase):
    def test_the_layer_is_read_from_any_block_container_name(self):
        self.assertEqual(3, layer_of("transformer.h.3.attn.c_attn.weight"))
        self.assertEqual(21, layer_of("model.layers.21.mlp.down_proj.weight"))
        self.assertEqual(-1, layer_of("transformer.wte.weight"))

    def test_c_proj_is_filed_by_its_branch(self):
        """The trap: the same leaf name is the attention output in one branch and the MLP output in the other"""
        self.assertEqual("attn.out", kind_of("transformer.h.0.attn.c_proj.weight"))
        self.assertEqual("mlp.out", kind_of("transformer.h.0.mlp.c_proj.weight"))

    def test_the_other_families_resolve_too(self):
        self.assertEqual("attn.q", kind_of("model.layers.1.self_attn.q_proj.weight"))
        self.assertEqual("attn.qkv", kind_of("transformer.h.1.attn.c_attn.weight"))
        self.assertEqual("mlp.in", kind_of("model.layers.1.mlp.gate_proj.weight"))
        self.assertEqual("mlp.other", kind_of("model.layers.1.mlp.mystery.weight"))
        self.assertEqual("attn.other", kind_of("model.layers.1.self_attn.mystery.weight"))


class TestParseLayers(TestCase):
    def test_a_band_a_list_and_all(self):
        self.assertEqual([21, 22, 23], parse_layers("21-23"))
        self.assertEqual([21, 23, 26], parse_layers("26,21,23"))
        self.assertIsNone(parse_layers("all"))
        self.assertIsNone(parse_layers(""))

    def test_nonsense_is_refused_with_the_grammar(self):
        with self.assertRaises(GateError) as caught:
            parse_layers("twenty")
        self.assertIn("21-27", str(caught.exception))


class TestReduction(TestCase):
    def setUp(self):
        self.gates = {
            "transformer.h.0.attn.c_attn.weight": mask(8, 24, open_share=0.25),
            "transformer.h.0.attn.c_proj.weight": mask(8, 8, open_share=0.5),
            "transformer.h.0.mlp.c_proj.weight": mask(32, 8, open_share=0.0),
            "transformer.h.1.mlp.c_fc.weight": mask(8, 32, open_share=1.0),
        }

    def test_open_count_is_over_every_tensor(self):
        opened, total = open_count(self.gates)
        self.assertEqual(192 + 64 + 256 + 256, total)
        self.assertEqual(48 + 32 + 0 + 256, opened)

    def test_packing_keeps_every_bit_and_nothing_else(self):
        """One bit per gate, back to the same mask, on shapes that do not fill a byte"""
        gates = {**self.gates, "transformer.h.2.mlp.c_fc.weight": mask(5, 3, open_share=0.4)}
        packed = pack(gates)
        self.assertEqual(2, packed["transformer.h.2.mlp.c_fc.weight"]["bits"].numel(), "15 gates in 2 bytes")
        self.assertEqual(torch.uint8, packed["transformer.h.0.attn.c_attn.weight"]["bits"].dtype)
        back = unpack(packed)
        for name, logits in gates.items():
            self.assertEqual(torch.bool, back[name].dtype)
            self.assertEqual(logits.shape, back[name].shape)
            self.assertTrue(torch.equal(logits > 0, back[name]), name)
        self.assertEqual(open_count(gates), open_count(back))
        self.assertEqual(per_component(gates), per_component(back))
        odd = "transformer.h.2.mlp.c_fc.weight"
        self.assertTrue(torch.equal(back[odd], unpack(pack(back))[odd]), "a bool mask packs the same as its logits")

    def test_a_packed_file_that_cannot_hold_its_shape_is_refused(self):
        packed = pack(self.gates)
        packed["transformer.h.0.attn.c_proj.weight"]["shape"] = [8, 9]
        with self.assertRaises(GateError):
            unpack(packed)

    def test_a_directory_prefers_the_logits_and_reads_the_mask_without_them(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(GateError):
                circuit_path(root, "ioi")
            torch.save(pack(self.gates), root / "sheaf-ioi-mask.pt")
            self.assertEqual(root / "sheaf-ioi-mask.pt", circuit_path(root, "ioi"))
            self.assertEqual(open_count(self.gates), open_count(load_circuit(circuit_path(root, "ioi"))))
            torch.save(self.gates, root / "sheaf-ioi-gates.pt")
            self.assertEqual(root / "sheaf-ioi-gates.pt", circuit_path(root, "ioi"))
            self.assertTrue(torch.equal(self.gates["transformer.h.0.attn.c_proj.weight"],
                                        load_circuit(circuit_path(root, "ioi"))["transformer.h.0.attn.c_proj.weight"]))

    def test_components_and_layers_add_up(self):
        table = per_component(self.gates)
        self.assertEqual({"open": 48, "total": 192}, table[(0, "attn.qkv")])
        self.assertEqual({"open": 0, "total": 256}, table[(0, "mlp.out")])
        self.assertEqual({0: 80, 1: 256}, by_layer(table))
        self.assertEqual((1, "mlp.in"), ranked(table)[0][0])

    def test_a_budget_is_the_model_plus_state_plus_one_graph(self):
        cost = budget(n_gates=2 ** 30, weight_bytes=2, model_bytes=2 ** 30)
        self.assertEqual(16.0, cost["optimizer_gib"])
        self.assertEqual(2.0, cost["originals_gib"])
        self.assertEqual(1.0, cost["model_gib"])
        self.assertEqual(11.0, cost["graph_gib"])
        self.assertEqual(30.0, cost["peak_gib"])


class TestOnline(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = adapter

    def test_loading_a_circuit_zeroes_and_then_restores_the_weights(self):
        """The eval is a difference against the unmasked model, so the restore has to be exact"""
        targets = gateable(self.adapter, layers=[0])
        name = next(iter(targets))
        before = targets[name].detach().clone()
        gates = {name: -torch.ones_like(before)}
        with circuit_loaded(self.adapter, gates):
            self.assertEqual(0.0, float(targets[name].detach().abs().sum()))
        self.assertTrue(torch.equal(before, targets[name].detach()))

    def test_a_bool_mask_loads_the_same_as_its_logits(self):
        targets = gateable(self.adapter, layers=[0])
        name = next(iter(targets))
        before = targets[name].detach().clone()
        logits = torch.randn_like(before)
        with circuit_loaded(self.adapter, {name: logits}):
            from_logits = targets[name].detach().clone()
        with circuit_loaded(self.adapter, unpack(pack({name: logits}))):
            from_mask = targets[name].detach().clone()
        self.assertTrue(torch.equal(from_logits, from_mask))
        self.assertTrue(torch.equal(before, targets[name].detach()))

    def test_a_tensor_the_model_does_not_have_is_refused(self):
        gates = {"transformer.h.99.mlp.c_fc.weight": torch.ones(2, 2)}
        with self.assertRaises(GateError), circuit_loaded(self.adapter, gates):
            pass

    def test_the_summary_names_every_head_of_a_gated_layer(self):
        targets = gateable(self.adapter, layers=[2])
        gates = {name: torch.ones_like(parameter) for name, parameter in targets.items()}
        circuit = summary(self.adapter, gates)
        self.assertEqual(circuit["n_gates"], circuit["n_open"])
        self.assertEqual({"2": circuit["n_gates"]}, circuit["by_layer"])
        self.assertEqual(self.adapter.cfg.n_heads, len(circuit["heads"]))
        self.assertTrue(all(row["layer"] == 2 and row["density"] == 1.0 for row in circuit["heads"]))

    def test_an_empty_mask_has_nothing_to_summarize(self):
        with self.assertRaises(GateError):
            summary(self.adapter, {})
