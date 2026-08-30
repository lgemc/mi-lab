"""The one checkpoint every online test shares.

GPT-2 small costs half a gigabyte resident, and this suite has more than a
dozen TestCase classes that want one. Loading it per class costs that a dozen
times over: on a CPU that is merely slow, and on a GPU it is an out-of-memory
error partway through the run.

Which would be survivable if it were reported. It is not: every online
setUpClass here turns a failure into `skipTest("gpt2-small is not
available")`, so an exhausted GPU comes back as a machine without a
checkpoint and the suite passes with a third of its tests quietly skipped.
That is the whole reason this module exists rather than a lru_cache on
load_adapter.

So the adapter is loaded once per process and handed out, and an
out-of-memory error is re-raised rather than disguised. Tests must not mutate
what they get: capture, logits and patch all leave the model as they found it,
and a test that needs a different config -- another batch_size, another dtype
-- builds its own and is the reason `shared` takes a name.
"""

_LOADED = {}

def shared_adapter(name: str = "gpt2-small"):
    """GPT-2 small, loaded once per process, or None if this machine cannot reach it

    None means the checkpoint is unreachable -- no network, no cache -- which
    is a skip. It never means the checkpoint would not fit, because that is a
    result and a skip is not.
    """
    if name not in _LOADED:
        try:
            from src.model.adapter import load_adapter

            _LOADED[name] = load_adapter(name)
        except Exception as error:
            if type(error).__name__ == "OutOfMemoryError":
                raise
            _LOADED[name] = None
    return _LOADED[name]
