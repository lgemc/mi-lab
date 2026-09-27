"""
The atlas: interpretability ideas small enough to see.

Every module here is one lesson's model, and every model is either a graph with
a probability table or a network hand-wired (or trained in a second) on purpose
so that the thing being explained is exactly true in it. That is the trade: a
toy cannot tell you what GPT-2 does, and it can show you what an instrument
would find if the story were true, which is what a learner needs first.

Nothing here imports torch or mi-lab's `src/`. The server has to answer a slider
in tens of milliseconds for a stranger on a phone, so everything is numpy and
every trainer is small enough to run inside a request and be cached after.
"""
