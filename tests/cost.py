"""Guards on the FLOPs bookkeeping the pre-registered ceiling is measured in.

The arithmetic is pinned against a hand-computable model so a change to what
is counted shows up as a number and not a vibe; the matched draw is held to
the tolerance it claims; and the on-disk report has to read back through
`from_dict`, because that is how every later step gets its cost model.
"""

from unittest import TestCase

from src.methods.cost import (
    MATCH_TOLERANCE,
    CostError,
    CostModel,
    Dimensions,
    matched_draw,
    report,
)

# small enough to check by hand: 4 layers of 2 heads, d_model 8, d_head 4, d_ff 16
TINY = Dimensions(hf_name="none/tiny", d_model=8, n_layers=4, n_heads=2, n_kv_heads=2, d_head=4, d_ff=16)
CONTEXT = 10


class TestCostModel(TestCase):
    def setUp(self):
        self.cost = CostModel.from_dimensions(TINY, CONTEXT)

    def test_the_macs_per_head_and_mlp_are_the_stated_sums(self):
        # q and o: 2 * 8 * 4 = 64; k and v at full kv heads: 64; scores: 2 * 10 * 4 = 80
        self.assertEqual(64 + 64 + 80, self.cost.head_macs)
        self.assertEqual(3 * 8 * 16, self.cost.mlp_macs)

    def test_grouped_kv_heads_share_their_projection(self):
        grouped = CostModel.from_dimensions(Dimensions(**{**TINY.__dict__, "n_kv_heads": 1}), CONTEXT)
        self.assertEqual(self.cost.head_macs - 32, grouped.head_macs)

    def test_totals_compose_from_the_parts(self):
        self.assertEqual(2 * self.cost.head_macs, self.cost.attention_macs)
        self.assertEqual(self.cost.attention_macs + self.cost.mlp_macs, self.cost.layer_macs)
        self.assertEqual(4 * self.cost.layer_macs, self.cost.total_macs)

    def test_a_head_group_costs_every_head_of_its_layer(self):
        self.assertEqual(self.cost.head_macs, self.cost.macs(["head:1:0"]))
        self.assertEqual(2 * self.cost.head_macs, self.cost.macs(["heads:1"]))
        self.assertEqual(self.cost.mlp_macs, self.cost.macs(["mlp:1"]))

    def test_the_whole_model_is_a_share_of_one(self):
        everything = [f"mlp:{layer}" for layer in range(4)] + [f"heads:{layer}" for layer in range(4)]
        self.assertAlmostEqual(1.0, self.cost.share(everything))

    def test_it_round_trips_through_its_own_dict(self):
        self.assertEqual(self.cost, CostModel.from_dict(self.cost.to_dict()))

    def test_it_reads_the_report_layout_too(self):
        """The file `phase1b_flops` writes is the one every later step reads its costs from"""
        data = report(TINY, self.cost, [2, 3], (0.75, 1.0))
        self.assertEqual(self.cost, CostModel.from_dict(data))

    def test_something_else_is_refused_by_name(self):
        with self.assertRaises(CostError):
            CostModel.from_dict({"model": {}, "per_component_macs": {}})


class TestReport(TestCase):
    def test_the_candidate_set_is_priced_as_a_share_of_the_model(self):
        cost = CostModel.from_dimensions(TINY, CONTEXT)
        data = report(TINY, cost, [2, 3], (0.75, 1.0))
        candidate = data["candidate_set"]
        self.assertEqual([2, 3], candidate["layers"])
        self.assertEqual(4, candidate["heads"])
        self.assertEqual(2, candidate["mlps"])
        # two of four layers, whole: half the model
        self.assertAlmostEqual(0.5, candidate["total_share"], places=3)
        self.assertAlmostEqual(0.25, data["masked_fraction_examples"]["p=0.5"], places=3)

    def test_the_quarter_line_says_whether_heads_alone_can_reach_it(self):
        cost = CostModel.from_dimensions(TINY, CONTEXT)
        line = report(TINY, cost, [3], (0.75, 1.0))["quarter_flops_line"]
        self.assertEqual(8, line["heads_in_model"])
        self.assertIn("heads_equal_to_25pct", line)


class TestMatchedDraw(TestCase):
    def setUp(self):
        self.cost = CostModel.from_dimensions(TINY, CONTEXT)
        self.pool = [f"mlp:{layer}" for layer in range(4)]
        self.pool += [f"head:{layer}:{h}" for layer in range(4) for h in range(2)]

    def test_a_draw_lands_within_tolerance_of_a_reachable_target(self):
        target = self.cost.share(["mlp:0", "head:1:0"])
        chosen, error = matched_draw(self.pool, self.cost, target, seed=0)
        self.assertLessEqual(error, MATCH_TOLERANCE)
        self.assertAlmostEqual(target, self.cost.share(chosen), delta=target * MATCH_TOLERANCE)

    def test_the_same_seed_draws_the_same_set(self):
        target = self.cost.share(["mlp:0", "mlp:1"])
        self.assertEqual(matched_draw(self.pool, self.cost, target, seed=3),
                         matched_draw(self.pool, self.cost, target, seed=3))

    def test_a_target_below_the_cheapest_component_cannot_be_matched(self):
        with self.assertRaises(CostError):
            matched_draw(self.pool, self.cost, self.cost.share(["head:0:0"]) / 4, seed=0)
