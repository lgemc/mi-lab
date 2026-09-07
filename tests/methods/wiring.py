"""Reducing an edge set to a circuit, and refusing to call noise a replication.

No model: `wiring.reduce` is arithmetic over two lists of tuples, which is the
point of it living in `methods` rather than in a script. The receipts are the
denominators (a degree is against what a source could reach, never against the
model) and the hypergeometric test, which exists because the boolean it
replaced called 9-against-7.2 an enrichment.
"""

from unittest import TestCase

from src.data.ioi import WANG_HEADS
from src.methods.circuits.wiring import WiringError, against, reduce

# a two-layer toy: embed and one head per layer, writing into attn/mlp
EVERY = [
    ("embed", "attn:0"), ("embed", "mlp:0"), ("embed", "attn:1"), ("embed", "mlp:1"),
    ("head:0:0", "mlp:0"), ("head:0:0", "attn:1"), ("head:0:0", "mlp:1"),
    ("mlp:0", "attn:1"), ("mlp:0", "mlp:1"),
    ("head:1:0", "mlp:1"),
]


class TestReduce(TestCase):
    def test_a_source_kept_everywhere_is_not_split(self):
        circuit = reduce([("embed", "attn:0"), ("embed", "mlp:0"),
                          ("embed", "attn:1"), ("embed", "mlp:1")], EVERY)
        self.assertEqual(circuit["sources_split"], 0)
        self.assertEqual(circuit["sources_all_kept"], 1)
        row = next(r for r in circuit["components"] if r["component"] == "embed")
        self.assertFalse(row["split"])
        self.assertEqual((row["out_kept"], row["out_available"]), (4, 4))

    def test_a_source_kept_for_some_readers_is_split(self):
        """The claim a weight mask cannot make, and the reason this module exists"""
        circuit = reduce([("head:0:0", "mlp:0")], EVERY)
        self.assertEqual(circuit["sources_split"], 1)
        row = next(r for r in circuit["components"] if r["component"] == "head:0:0")
        self.assertTrue(row["split"])
        self.assertEqual((row["out_kept"], row["out_available"]), (1, 3))
        self.assertEqual(row["readers"], ["mlp:0"])

    def test_a_degree_is_against_what_the_source_could_reach(self):
        """head:1:0 can only reach one destination, so one edge is 100% of it"""
        circuit = reduce([("head:1:0", "mlp:1"), ("embed", "attn:0")], EVERY)
        late = next(r for r in circuit["components"] if r["component"] == "head:1:0")
        early = next(r for r in circuit["components"] if r["component"] == "embed")
        self.assertEqual(late["reach"], 1.0)
        self.assertEqual(early["out_available"], 4)
        self.assertLess(early["reach"], late["reach"], "a raw out-degree would order these wrongly")

    def test_components_are_the_sources_with_a_surviving_path(self):
        circuit = reduce([("head:0:0", "attn:1"), ("mlp:0", "mlp:1")], EVERY)
        self.assertEqual(circuit["heads_in_circuit"], ["head:0:0"])
        self.assertEqual(circuit["mlps_in_circuit"], ["mlp:0"])
        self.assertEqual(circuit["sources_all_cut"], 2)  # embed and head:1:0

    def test_an_edge_from_another_model_is_refused(self):
        """A plausible wrong graph is the failure mode; silence would be worse"""
        with self.assertRaises(WiringError):
            reduce([("head:9:6", "attn:10")], EVERY)

    def test_the_empty_circuit_reduces_rather_than_raising(self):
        circuit = reduce([], EVERY)
        self.assertEqual(circuit["n_edges_open"], 0)
        self.assertEqual(circuit["components"], [])
        self.assertEqual(circuit["sources_all_cut"], 4)


class TestAgainst(TestCase):
    """The published comparison, which has to be able to say 'no'"""

    def test_the_gpt2_ioi_overlap_is_not_significant(self):
        """The measured case: 9 of 26 from 40 of 144 heads is p 0.26, not a replication

        This is the number that a `found > expected` boolean called an
        enrichment. It is the receipt on the test, not on the circuit.
        """
        hits = ["0.1", "0.10", "3.0", "5.9", "7.3", "7.9", "9.6", "9.9", "10.10"]
        # 31 heads that are *not* in the reference, so the overlap is exactly 9
        filler = [f"{layer}.{head}" for layer in range(12) for head in range(12)
                  if f"{layer}.{head}" not in WANG_HEADS][:31]
        kept = [f"head:{h.replace('.', ':')}" for h in hits + filler]
        result = against(kept, WANG_HEADS, 144)
        self.assertEqual(result["kept"], 40)
        self.assertEqual(result["found"], 9)
        self.assertAlmostEqual(result["expected_by_chance"], 7.22, places=2)
        self.assertGreater(result["p_value"], 0.05)

    def test_a_perfect_recovery_is_significant(self):
        result = against([f"head:{h.replace('.', ':')}" for h in WANG_HEADS], WANG_HEADS, 144)
        self.assertEqual(result["found"], len(WANG_HEADS))
        self.assertLess(result["p_value"], 1e-6)
        self.assertEqual(result["missing"], [])

    def test_no_overlap_reports_every_reference_head_missing(self):
        result = against(["head:0:2", "head:0:3"], WANG_HEADS, 144)
        self.assertEqual(result["found"], 0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(len(result["missing"]), len(WANG_HEADS))
