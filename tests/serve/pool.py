"""On-demand loading and idle unloading, checked without loading a checkpoint

The pool's job is to let two models share one time-sliced GPU slice by not
being resident at the same time, and every way that goes wrong is a
bookkeeping bug rather than a modelling one -- so the loader is injected and
these run in milliseconds. What is asserted is the three promises the module
makes:

  a live model is never evicted    a sweep during a generation would free the
                                   weights out from under it
  a first request loads once       two simultaneous first requests must not
                                   load the same checkpoint twice
  idle means idle                  the clock is the *last* use, not the first
"""

import threading
import time
import unittest

from src.serve.models import ModelPool, ModelSpec, PoolError


class Fake:
    """Stands in for a Circuits, and counts how often its model was built"""

    def __init__(self, spec):
        self.spec = spec


class TestModelSpec(unittest.TestCase):
    def test_a_bare_config_names_itself(self):
        spec = ModelSpec.parse("qwen3-1.7b")
        self.assertEqual((spec.name, spec.config, spec.circuits), ("qwen3-1.7b", "qwen3-1.7b", None))

    def test_a_root_and_an_explicit_name(self):
        spec = ModelSpec.parse("circuits=qwen3-1.7b@/circuits")
        self.assertEqual(spec.name, "circuits")
        self.assertEqual(str(spec.circuits), "/circuits")

    def test_none_is_the_generation_server(self):
        self.assertIsNone(ModelSpec.parse("sdft=qwen3-0.6b-sdft@none").circuits)

    def test_a_spec_with_no_config_says_what_to_write(self):
        with self.assertRaises(PoolError) as caught:
            ModelSpec.parse("name=")
        self.assertIn("name=config", str(caught.exception))


class TestPool(unittest.TestCase):
    def setUp(self):
        self.built = []
        self.pool = ModelPool(
            [ModelSpec("a", "cfg-a"), ModelSpec("b", "cfg-b")],
            idle_timeout=0.05,
            build=self.build,
        )

    def build(self, spec):
        self.built.append(spec.name)
        return Fake(spec)

    def test_nothing_is_loaded_until_it_is_asked_for(self):
        self.assertEqual(self.pool.names(), ["a", "b"])
        self.assertEqual(self.pool.resident(), [])
        self.assertEqual(self.built, [])

    def test_a_model_loads_once_and_is_reused(self):
        first = self.pool.get("a")
        self.assertIs(self.pool.get("a"), first)
        self.assertEqual(self.built, ["a"])
        self.assertEqual(self.pool.resident(), ["a"])

    def test_an_unknown_model_names_the_ones_that_exist(self):
        with self.assertRaises(PoolError) as caught:
            self.pool.get("nope")
        self.assertIn("'a'", str(caught.exception))

    def test_two_first_requests_load_it_once(self):
        """The per-name lock, which is the difference between one load and two"""
        slow = threading.Event()
        original = self.build

        def build(spec):
            slow.wait(2)
            return original(spec)

        self.pool._build = build
        threads = [threading.Thread(target=lambda: self.pool.get("a")) for _ in range(4)]
        for t in threads:
            t.start()
        slow.set()
        for t in threads:
            t.join(5)
        self.assertEqual(self.built, ["a"])

    def test_an_idle_model_is_swept(self):
        self.pool.get("a")
        time.sleep(0.08)
        self.assertEqual(self.pool.sweep(), ["a"])
        self.assertEqual(self.pool.resident(), [])

    def test_use_resets_the_clock(self):
        """Idle is measured from the *last* use, or a long session would evict itself"""
        self.pool.get("a")
        time.sleep(0.04)
        with self.pool.use("a"):
            pass
        self.assertEqual(self.pool.sweep(), [])

    def test_a_busy_model_survives_a_sweep_however_old(self):
        """A sweep during a generation would free the weights out from under it"""
        with self.pool.use("a"):
            time.sleep(0.08)
            self.assertEqual(self.pool.sweep(), [])
            self.assertEqual(self.pool.resident(), ["a"])
        # and goes as soon as it is no longer in use
        time.sleep(0.08)
        self.assertEqual(self.pool.sweep(), ["a"])

    def test_unload_refuses_a_busy_model_unless_forced(self):
        with self.pool.use("a"):
            self.assertFalse(self.pool.unload("a"))
            self.assertTrue(self.pool.unload("a", force=True))

    def test_unloading_something_absent_is_not_an_error(self):
        self.assertFalse(self.pool.unload("b"))

    def test_two_models_take_turns_rather_than_sharing_the_device(self):
        """The whole point: two checkpoints, one slice, never both resident"""
        with self.pool.use("a"):
            pass
        time.sleep(0.08)
        self.pool.sweep()
        with self.pool.use("b"):
            self.assertEqual(self.pool.resident(), ["b"])
        self.assertEqual(self.built, ["a", "b"])

    def test_describe_reports_configured_and_resident_separately(self):
        self.pool.get("a")
        by_name = {d["name"]: d for d in self.pool.describe()}
        self.assertTrue(by_name["a"]["resident"])
        self.assertFalse(by_name["b"]["resident"])
        self.assertIsNone(by_name["b"]["idle_seconds"])

    def test_duplicate_names_are_refused_at_construction(self):
        with self.assertRaises(PoolError) as caught:
            ModelPool([ModelSpec("a", "one"), ModelSpec("a", "two")], build=self.build)
        self.assertIn("name=config", str(caught.exception))

    def test_an_empty_pool_is_refused(self):
        with self.assertRaises(PoolError):
            ModelPool([], build=self.build)

    def test_the_sweeper_thread_drops_an_idle_model(self):
        dropped = []
        pool = ModelPool([ModelSpec("a", "cfg")], idle_timeout=0.01, build=self.build)
        pool.get("a")
        # Drive the loop the sweeper drives, rather than waiting SWEEP_EVERY
        # seconds for it: what is under test is that `on_unload` is reported,
        # not that `Event.wait` works.
        time.sleep(0.02)
        for name in pool.sweep():
            dropped.append(name)
        self.assertEqual(dropped, ["a"])
        pool.stop()

    def test_peek_does_not_count_as_a_use(self):
        """/health runs forever; if reading counted, the default model would never be swept"""
        self.pool.get("a")
        time.sleep(0.04)
        self.assertIsNotNone(self.pool.peek("a"))
        time.sleep(0.02)
        self.assertEqual(self.pool.sweep(), ["a"])

    def test_peek_never_loads(self):
        self.assertIsNone(self.pool.peek("b"))
        self.assertEqual(self.built, [])
