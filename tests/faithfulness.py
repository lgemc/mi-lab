"""The methodology surface, checked against the claims it was built from.

2407.08734's findings are the specification here, not decoration: if resample
does not sit below mean, or the two aggregation orders agree, or the four
direction/set experiments give one answer, then this module is not measuring
what that paper measured and the numbers it produces mean something else.
"""

from unittest import TestCase

from src.data.tasks import build_task
from src.methods.circuits import CircuitError, discover, require_circuits
from src.methods.faithfulness import (
    AGGREGATIONS,
    Methodology,
    measure,
    report,
    sensitivity,
)

from .stubs.model import shared_adapter


class FaithfulnessTestCase(TestCase):
    """Discovers a circuit rather than naming one, because the premise matters

    An arbitrary handful of heads is not a circuit, and on one of those the
    relationships these tests assert genuinely invert: restoring four
    hand-picked heads recovered 0.21 while destroying them left 0.86, which is
    the model routing around a set too small to matter rather than anything
    wrong with the measurement. Every claim here is about what a *discovered*
    circuit does, so the fixture discovers one.
    """
    adapter = None
    task = None
    heads = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        cls.task = build_task("ioi", cls.adapter, size=8, seed=0)
        cls.heads = discover(cls.adapter, cls.task, max_heads=8).heads

class TestMethodology(TestCase):
    def test_an_unknown_axis_value_is_refused(self):
        for field, bad in (("value", "gaussian"), ("direction", "sideways"),
                           ("ablated", "everything"), ("aggregation", "geometric")):
            with self.subTest(field=field), self.assertRaises(CircuitError):
                Methodology(**{field: bad})

    def test_the_axes_this_repo_cannot_vary_are_stated_not_omitted(self):
        """An artifact saying `component: node` and one saying nothing are different claims"""
        m = Methodology()
        self.assertEqual(("head", "node", "all"), (m.granularity, m.component, m.positions))

class TestMeasurement(FaithfulnessTestCase):
    def test_an_empty_circuit_is_refused(self):
        with self.assertRaises(CircuitError):
            measure(self.adapter, self.task, [])

    def test_every_aggregation_reports_the_same_distribution(self):
        """The orders differ in how they summarize, not in what they measured"""
        scores = [measure(self.adapter, self.task, self.heads, Methodology(aggregation=a))
                  for a in AGGREGATIONS]
        self.assertEqual(1, len({tuple(s.per_example) for s in scores}))

    def test_restoring_the_circuit_recovers_more_than_destroying_it_leaves(self):
        restored = measure(self.adapter, self.task, self.heads, Methodology(direction="restore"))
        destroyed = measure(self.adapter, self.task, self.heads, Methodology(direction="destroy"))
        self.assertGreater(restored.score, destroyed.score)

    def test_the_score_carries_the_tuple_that_produced_it(self):
        one = measure(self.adapter, self.task, self.heads, Methodology(value="zero"))
        self.assertEqual("zero", one.methodology.value)
        self.assertIn("zero", one.methodology.label())

class TestThePapersClaims(FaithfulnessTestCase):
    """Each test here is a finding of 2407.08734 restated as an assertion"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = sensitivity(cls.adapter, cls.task, cls.heads)

    def find(self, **kw):
        return next(r for r in self.results
                    if all(getattr(r.methodology, k) == v for k, v in kw.items()))

    def test_resample_sits_below_mean_ablation(self):
        for ablated in ("circuit", "complement"):
            with self.subTest(ablated=ablated):
                mean = self.find(direction="destroy", ablated=ablated, value="mean",
                                 aggregation="ratio_of_means").score
                resample = self.find(direction="destroy", ablated=ablated, value="resample",
                                     aggregation="ratio_of_means").score
                self.assertLess(resample, mean)

    def test_the_two_aggregation_orders_disagree(self):
        a = self.find(direction="destroy", ablated="circuit", value="resample",
                      aggregation="ratio_of_means").score
        b = self.find(direction="destroy", ablated="circuit", value="resample",
                      aggregation="mean_of_ratios").score
        self.assertNotAlmostEqual(a, b, places=3)

    def test_running_the_circuit_alone_has_two_answers(self):
        """restore/circuit and destroy/complement both claim it, and disagree"""
        restore = self.find(direction="restore", ablated="circuit",
                            aggregation="ratio_of_means").score
        delete = self.find(direction="destroy", ablated="complement", value="resample",
                           aggregation="ratio_of_means").score
        self.assertGreater(abs(restore - delete), 0.1)

    def test_per_example_spread_is_wide_enough_to_hide_a_failure(self):
        """A mean near 0.9 is compatible with examples reproducing almost nothing"""
        restored = self.find(direction="restore", ablated="circuit", aggregation="ratio_of_means")
        self.assertGreater(restored.score - restored.worst, 0.2)

    def test_the_report_states_the_spread_rather_than_one_number(self):
        rendered = report(self.results)
        self.assertGreater(rendered["spread"], 0.5)
        self.assertEqual(len(self.results), rendered["n_measurements"])
        self.assertIn("node", rendered["axes"]["component"])
