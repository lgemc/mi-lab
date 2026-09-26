from unittest import TestCase

from src.domains.lm.analysis.leaderboard import markdown, pooled, rank

"""
The leaderboard is where a readout comparison is quietly made unfair, so what
it pools and what it refuses to rank are tested on hand-built results files.
"""


def item(hit: bool, leaked: bool, layer=None):
    return {"hit_layers": [layer or 5] if hit else [], "leaked": leaked, "best_layer": (layer or 5) if hit else None}


def state(method: str, config: str, banks):
    """A results file with the shape `wsbench_baseline` writes; each bank is a list of items"""
    held = {}
    for family, items in banks.items():
        clean = [row for row in items if not row["leaked"]]
        held[family] = {"items": items, "summary": {
            "items": len(items),
            "hit_rate": sum(bool(row["hit_layers"]) for row in items) / len(items),
            "leak_rate": sum(row["leaked"] for row in items) / len(items),
            "clean_hit_rate": sum(bool(row["hit_layers"]) for row in clean) / len(clean) if clean else None,
        }}
    return {"method": method, "config": config, "banks": held}


class TestPooled(TestCase):
    def test_the_clean_rate_is_pooled_over_clean_items_not_averaged_over_banks(self):
        """Bank a: 1 clean item, hit. Bank b: 3 clean items, none hit. Averaging rates says 50%; it is 25%"""
        row = pooled(state("m", "c", {
            "a": [item(True, False), item(True, True)],
            "b": [item(False, False), item(False, False), item(False, False)],
        }))
        self.assertEqual(0.25, row["clean_hit_rate"])
        self.assertEqual(0.4, row["hit_rate"])
        self.assertEqual(0.2, row["leak_rate"])

    def test_a_file_from_before_methods_had_names_is_the_logit_lens(self):
        legacy = state("x", "c", {"a": [item(True, False)]})
        del legacy["method"]
        self.assertEqual("logit-lens", pooled(legacy)["method"])


class TestRank(TestCase):
    BANKS = ("a", "b")

    def rows(self):
        return [
            pooled(state("logit-lens", "big", {"a": [item(True, False)], "b": [item(False, False)]})),
            pooled(state("jlens", "big", {"a": [item(True, False)], "b": [item(True, False)]})),
            pooled(state("smoke", "big", {"a": [item(True, False)]})),
        ]

    def test_complete_rows_are_ranked_by_clean_hit_rate(self):
        ranked = rank(self.rows(), expected=self.BANKS)
        self.assertEqual([("jlens", 1), ("logit-lens", 2)],
                         [(row["method"], row["rank"]) for row in ranked if row["complete"]])

    def test_a_partial_run_is_listed_and_not_ranked(self):
        """A subset that happens to be easy would otherwise top the table"""
        (smoke,) = [row for row in rank(self.rows(), expected=self.BANKS) if row["method"] == "smoke"]
        self.assertEqual((None, False), (smoke["rank"], smoke["complete"]))

    def test_a_lone_partial_run_is_not_complete_just_because_it_is_the_widest(self):
        lone = [pooled(state("logit-lens", "small", {"a": [item(True, False)]}))]
        self.assertFalse(rank(lone, expected=self.BANKS)[0]["complete"])

    def test_models_are_never_ranked_against_each_other(self):
        other = pooled(state("logit-lens", "other", {"a": [item(True, False)], "b": [item(True, False)]}))
        rows = [*self.rows()[:2], other]
        ranked = rank(rows, expected=self.BANKS)
        self.assertEqual(1, next(row for row in ranked if row["config"] == "other")["rank"])

    def test_the_page_puts_every_complete_method_side_by_side_per_bank(self):
        page = markdown(rank(self.rows(), expected=self.BANKS))
        self.assertIn("| bank | items | jlens | logit-lens |", page)
        self.assertIn("| incomplete | smoke |", page)
