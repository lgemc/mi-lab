"""The d-separation verdicts and adjustment criteria against the textbook cases"""

import unittest

from atlas.causal import DAG, analyze, d_separated, do_rule, frontdoor, is_backdoor_set, is_frontdoor_set


def g(edges, latent=()):
    nodes = sorted({n for e in edges for n in e})
    return DAG(nodes, edges, set(latent))


class TestDSeparation(unittest.TestCase):
    def test_chain_is_blocked_by_its_middle(self):
        chain = g([("X", "M"), ("M", "Y")])
        self.assertFalse(d_separated(chain, ["X"], ["Y"], []))
        self.assertTrue(d_separated(chain, ["X"], ["Y"], ["M"]))

    def test_fork_is_blocked_by_its_middle(self):
        fork = g([("Z", "X"), ("Z", "Y")])
        self.assertFalse(d_separated(fork, ["X"], ["Y"], []))
        self.assertTrue(d_separated(fork, ["X"], ["Y"], ["Z"]))

    def test_collider_is_opened_by_conditioning_on_it_or_a_descendant(self):
        collider = g([("X", "C"), ("Y", "C"), ("C", "D")])
        self.assertTrue(d_separated(collider, ["X"], ["Y"], []))
        self.assertFalse(d_separated(collider, ["X"], ["Y"], ["C"]))
        self.assertFalse(d_separated(collider, ["X"], ["Y"], ["D"]))

    def test_a_cycle_is_refused(self):
        with self.assertRaises(ValueError):
            g([("A", "B"), ("B", "A")])


class TestAdjustment(unittest.TestCase):
    def test_m_bias_the_empty_set_is_valid_and_the_collider_is_not(self):
        m = g([("U1", "X"), ("U1", "M"), ("U2", "M"), ("U2", "Y"), ("X", "Y")])
        self.assertTrue(is_backdoor_set(m, "X", "Y", set()))
        self.assertFalse(is_backdoor_set(m, "X", "Y", {"M"}))
        self.assertTrue(is_backdoor_set(m, "X", "Y", {"M", "U1"}))

    def test_a_descendant_of_the_treatment_is_never_a_backdoor_set(self):
        mediated = g([("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y")])
        self.assertFalse(is_backdoor_set(mediated, "X", "Y", {"Z", "M"}))
        self.assertEqual(analyze(mediated, "X", "Y", [])["backdoor_sets"], [["Z"]])

    def test_front_door_graph(self):
        fd = g([("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")], latent=["U"])
        self.assertTrue(is_frontdoor_set(fd, "X", "Y", {"M"}))
        self.assertEqual(analyze(fd, "X", "Y", [])["backdoor_sets"], [])

    def test_front_door_formula_recovers_the_truth_that_the_naive_estimate_misses(self):
        r = frontdoor(0.5, [0.2, 0.8], [0.1, 0.9], [[0.1, 0.5], [0.3, 0.8]])
        self.assertAlmostEqual(r["effects"]["frontdoor"], r["effects"]["truth"], places=9)
        self.assertGreater(abs(r["effects"]["naive"] - r["effects"]["truth"]), 0.05)


class TestDoCalculus(unittest.TestCase):
    def test_front_door_derivation_steps(self):
        fd = g([("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")], latent=["U"])
        # P(m|do(x)) = P(m|x): rule 2, (M ⊥ X) in G with arrows out of X removed
        self.assertTrue(do_rule(fd, 2, ["M"], [], ["X"], [])["holds"])
        # P(y|do(m)) = sum_x P(y|m,x)P(x): rule 2 given X
        self.assertTrue(do_rule(fd, 2, ["Y"], [], ["M"], ["X"])["holds"])
        # but not without X: the back door M <- X <- U -> Y is open
        self.assertFalse(do_rule(fd, 2, ["Y"], [], ["M"], [])["holds"])
        # P(x|do(m)) = P(x): rule 3
        self.assertTrue(do_rule(fd, 3, ["X"], [], ["M"], [])["holds"])


if __name__ == "__main__":
    unittest.main()
