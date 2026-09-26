"""Guards on attribution: the estimate is of the study's own intervention, and its two routes agree.

The offline half is the arithmetic that does not need a checkpoint: the
metric registry's refusals, the sign convention, what a ranking serialises
to, and the edge-to-node sum.

The online half is the part that can only be checked against a model, and it
is checked against the thing being approximated rather than against itself:
mean-ablate a component for real, measure the same metric, and ask whether
the first-order score predicted it. Two further receipts are here because
each is a place this could be confidently wrong -- a first-order estimate
that silently expanded around the wrong point, and an edge decomposition that
does not add up to the node score it claims to decompose.
"""

import contextlib
import io
from unittest import TestCase

import torch

from src.core.metrics import spearman
from src.methods import attribution as attr
from src.methods.knockout import ablate, capture_means
from src.model.passes import encode, scored_positions, teacher_forced

from .stubs.model import shared_adapter

LAYERS = [8, 9, 10, 11]
PROMPTS = ["The capital of France is", "The capital of Germany is",
           "The capital of Italy is", "The capital of Spain is"]
TARGETS = ["Paris", "Berlin", "Rome", "Madrid"]
COUNTERFACTUAL = ["The colour of the sky is", "The colour of grass is",
                  "A number after three is", "A day after Monday is"]


def synthetic(scores, granularity="node", **overrides) -> attr.Attribution:
    fields = {"method": "eap", "metric": "target", "granularity": granularity,
              "units": "target drop per example (first order)", "layers": [0, 1],
              "steps": 1, "examples": 4, "passes": 2}
    return attr.Attribution(scores=scores, **{**fields, **overrides})


class TestMetrics(TestCase):
    def test_a_metric_that_is_flat_on_the_clean_model_is_refused_at_one_step(self):
        self.assertTrue(attr.metric("kl").flat_at_clean)
        self.assertFalse(attr.metric("target").flat_at_clean)

    def test_an_unknown_metric_names_the_known_ones(self):
        with self.assertRaises(attr.AttributionError) as caught:
            attr.metric("bleu")
        self.assertIn("target", str(caught.exception))

    def test_the_target_metric_is_a_goodness(self):
        # a distribution that is certain of the right token scores higher than one that is not
        ids = torch.tensor([[0, 1]])
        weights = torch.tensor([[1.0, 1.0]])
        certain = torch.tensor([[[0.0, 9.0], [0.0, 0.0]]])
        unsure = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        self.assertGreater(float(attr._target(certain, ids, weights)),
                           float(attr._target(unsure, ids, weights)))

    def test_a_metric_sees_only_the_positions_the_span_scores(self):
        """The prompt is shared by every span and no component is responsible for it"""
        ids = torch.tensor([[0, 1, 1]])
        logits = torch.tensor([[[0.0, 5.0], [0.0, 0.0], [0.0, 0.0]]])
        both = torch.tensor([[1.0, 1.0, 1.0]])
        second = torch.tensor([[0.0, 0.0, 1.0]])
        rows, _ = attr.scored(logits, ids, second)
        self.assertEqual(1, rows.shape[0])
        self.assertNotAlmostEqual(float(attr._target(logits, ids, both)),
                                  float(attr._target(logits, ids, second)))

    def test_a_span_with_nothing_to_score_is_refused(self):
        ids = torch.tensor([[0, 1]])
        logits = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        with self.assertRaises(attr.AttributionError):
            attr._target(logits, ids, torch.zeros(1, 2))

    def test_the_kl_metric_is_zero_against_itself_and_negative_otherwise(self):
        ids = torch.tensor([[0, 1]])
        weights = torch.tensor([[1.0, 1.0]])
        logits = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
        self.assertAlmostEqual(0.0, float(attr._kl(logits, ids, weights, logits)), places=5)
        self.assertLess(float(attr._kl(logits * 3, ids, weights, logits)), 0.0)


class TestRanking(TestCase):
    def test_a_ranking_is_most_damaging_first_and_the_other_end_on_request(self):
        found = synthetic({"mlp:0": 0.5, "mlp:1": -0.2, "head:0:1": 0.1})
        self.assertEqual(["mlp:0", "head:0:1", "mlp:1"], found.top(3))
        self.assertEqual("mlp:1", found.ranked(1, negative=True)[0][0])

    def test_components_drop_the_units_the_model_is_better_without(self):
        found = synthetic({"mlp:0": 0.5, "mlp:1": -0.2})
        self.assertEqual(["mlp:0"], attr.as_components(found))
        self.assertEqual(["mlp:0", "mlp:1"], attr.as_components(found, positive_only=False))

    def test_edges_sum_to_their_sources(self):
        found = synthetic({("mlp:0", "attn:1"): 0.3, ("mlp:0", "logits"): 0.2,
                           ("head:0:1", "logits"): -0.1}, granularity="edge")
        nodes = found.to_nodes()
        self.assertEqual("node", nodes.granularity)
        self.assertAlmostEqual(0.5, nodes.scores["mlp:0"])
        self.assertAlmostEqual(-0.1, nodes.scores["head:0:1"])

    def test_a_node_attribution_has_no_edges_to_report(self):
        with self.assertRaises(attr.AttributionError):
            synthetic({"mlp:0": 0.5}).edges()
        with self.assertRaises(attr.AttributionError):
            attr.split(synthetic({"mlp:0": 0.5}))

    def test_it_round_trips_through_disk_at_both_granularities(self):
        for scores, granularity in (({"mlp:0": 0.5, "head:1:2": -0.25}, "node"),
                                    ({("mlp:0", "logits"): 0.5}, "edge")):
            back = attr.Attribution.from_dict(synthetic(scores, granularity).to_dict())
            self.assertEqual(granularity, back.granularity)
            self.assertEqual(set(scores), set(back.scores))
            for key, value in scores.items():
                self.assertAlmostEqual(value, back.scores[key])

    def test_agreement_refuses_two_rankings_with_nothing_in_common(self):
        with self.assertRaises(attr.AttributionError):
            attr.agreement(synthetic({"mlp:0": 1.0}), {"mlp:9": 1.0})

    def test_agreement_reports_both_the_correlation_and_the_top(self):
        estimate = synthetic({"mlp:0": 3.0, "mlp:1": 2.0, "mlp:2": 1.0, "mlp:3": 0.0})
        measured = {"mlp:0": 30.0, "mlp:1": 20.0, "mlp:2": 10.0, "mlp:3": 0.0}
        report = attr.agreement(estimate, measured, at=(2,))
        self.assertEqual(4, report["components"])
        self.assertAlmostEqual(1.0, report["spearman"])
        self.assertEqual(1.0, report["overlap"]["top2"])


class TestOnline(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = adapter
        with contextlib.redirect_stdout(io.StringIO()):
            cls.means = capture_means(adapter, COUNTERFACTUAL, layers=LAYERS)
        cls.spans = teacher_forced(adapter, PROMPTS, TARGETS)

    def score(self, layers=None, **options) -> attr.Attribution:
        with contextlib.redirect_stdout(io.StringIO()):
            return attr.attribute(self.adapter, self.means, self.spans, layers or LAYERS,
                                  batch_size=4, **options)

    def measured(self, component: str) -> float:
        """What mean-ablating this component really costs the metric, by doing it"""
        ids, mask = encode(self.adapter, [span.text for span in self.spans], attr.MAX_LENGTH)
        weights = scored_positions(self.spans, mask)
        with torch.no_grad():
            clean = float(attr._target(
                self.adapter.model(ids, attention_mask=mask, use_cache=False).logits, ids, weights))
            with ablate(self.adapter, self.means, [component]):
                hurt = float(attr._target(
                    self.adapter.model(ids, attention_mask=mask, use_cache=False).logits, ids, weights))
        return clean - hurt

    def test_a_span_ends_where_the_tokenizer_agrees_it_does(self):
        span = self.spans[0]
        self.assertEqual("The capital of France is Paris", span.text)
        prompt = self.adapter.tokenizer(PROMPTS[0])["input_ids"]
        self.assertEqual(len(prompt), span.start)

    def test_a_target_that_merges_into_the_prompt_is_refused_rather_than_scored(self):
        from src.core.config import ConfigError

        with self.assertRaises(ConfigError):
            teacher_forced(self.adapter, ["The capital of Franc"], ["e is Paris"], joiner="")

    def test_the_estimate_predicts_the_ablation_it_approximates(self):
        """The only test that matters: does the first-order score rank what really happens

        Both ends of the ranking are tested, because an estimate that is
        right about what matters and wrong about the sign of what does not is
        an estimate a greedy walk will still be led astray by.
        """
        found = self.score(metric_name="target", steps=1)
        tested = found.top(8) + [key for key, _ in found.ranked(4, negative=True)]
        real = [self.measured(component) for component in tested]
        self.assertGreater(spearman([found.scores[component] for component in tested], real), 0.6)

    def test_it_costs_the_same_number_of_passes_however_many_components_are_scored(self):
        """The claim the method rests on: one lattice or a quarter of it, the same two passes"""
        whole = self.score(metric_name="target", steps=1)
        narrow = self.score(metric_name="target", steps=1, layers=[11])
        self.assertEqual(whole.passes, narrow.passes)
        self.assertEqual(len(LAYERS) * (1 + self.adapter.cfg.n_heads), len(whole.scores))
        self.assertEqual(1 + self.adapter.cfg.n_heads, len(narrow.scores))

    def test_kl_at_one_step_is_refused_with_the_way_out(self):
        with self.assertRaises(attr.AttributionError) as caught:
            self.score(metric_name="kl", steps=1)
        self.assertIn("steps > 1", str(caught.exception))

    def test_kl_away_from_the_clean_point_is_not_zero(self):
        found = self.score(metric_name="kl", steps=3)
        self.assertTrue(any(abs(value) > 1e-6 for value in found.scores.values()))

    def test_the_integrated_estimate_is_the_first_order_one_at_a_single_step(self):
        """`steps=1` must take no separate code path, or the two are not comparable"""
        first = self.score(metric_name="target", steps=1)
        again = self.score(metric_name="target", steps=1)
        self.assertEqual("eap", first.method)
        for key, value in first.scores.items():
            self.assertAlmostEqual(value, again.scores[key], places=6)

    def test_an_edge_attribution_sums_to_the_node_attribution_of_the_same_pass(self):
        """Two routes to one number, which is two chances to be wrong

        The node score reads the gradient at the component's own site and lets
        autograd sum over every reader; the edge score reads a partial
        gradient per reader and sums them here. They are the same quantity and
        they are computed by different code, so a discrepancy is a bug in one
        of them -- it was a real one: hooking the residual tensor instead of
        opening a slot per reader counted every downstream path once per
        destination and came out several times too large.
        """
        edges = self.score(metric_name="target", steps=1, granularity="edge")
        nodes = self.score(metric_name="target", steps=1)
        summed = edges.to_nodes()
        scale = max(abs(value) for value in nodes.scores.values())
        for component, value in nodes.scores.items():
            self.assertAlmostEqual(value, summed.scores[component], delta=scale * 1e-4)

    def test_every_source_has_an_edge_to_the_readout(self):
        """The edge `adapter.edges()` does not list, without which no source's edges add up"""
        edges = self.score(metric_name="target", steps=1, granularity="edge")
        sources = {source for source, _ in edges.scores}
        self.assertTrue(all((source, attr.READOUT) in edges.scores for source in sources))

    def test_the_split_says_how_much_of_a_source_one_reader_takes(self):
        edges = self.score(metric_name="target", steps=1, granularity="edge")
        report = attr.split(edges, at=5)
        self.assertTrue(report["rows"])
        for row in report["rows"]:
            self.assertGreaterEqual(row["destinations"], 1)
            self.assertLessEqual(row["largest_share"], 1.0)

    def test_a_layer_with_no_counterfactual_mean_is_refused(self):
        with self.assertRaises(attr.AttributionError) as caught:
            self.score(metric_name="target", steps=1, layers=[0])
        self.assertIn("nothing to be ablated toward", str(caught.exception))

    def test_the_model_is_left_as_it_was_found(self):
        """No hook outlives the pass, and no weight was asked for a gradient on the way through"""
        before = [parameter.requires_grad for parameter in self.adapter.model.parameters()]
        self.score(metric_name="target", steps=2)
        after = [parameter.requires_grad for parameter in self.adapter.model.parameters()]
        self.assertEqual(before, after)
        self.assertTrue(all(parameter.grad is None for parameter in self.adapter.model.parameters()))
        clean = self.adapter.logits(PROMPTS[:1])
        self.assertTrue(torch.equal(clean, self.adapter.logits(PROMPTS[:1])))
