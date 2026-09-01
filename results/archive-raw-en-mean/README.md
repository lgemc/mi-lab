# Archived: results measured against the raw-English mean

Everything in this directory was produced by the first Phase 1b sweep, whose
mean-ablation reference distribution was the raw English reference sentences of
the WMT-500 dev subset -- no few-shot skeleton, no `Spanish:`/`English:` labels,
no in-context task.

That distribution differs from the eval prompts in the prompt format and the
in-context task as well as in translation, so a component mean-ablated toward it
is told to forget how to continue a prompt at all. Wang et al. (2211.00593 3)
require the opposite: p_ABC keeps the p_IOI templates and varies only the names,
"because using p_IOI would not remove enough information helpful for the task".
Zhang et al. (2502.11806 3) build their X- on the same principle.

The visible symptom: ablating all nine candidate MLPs collapsed the output to
`"a a a a a a ..."` (BLEU 6.06) -- a broken model, not an ablated circuit. The
0.643 dCOMET that cleared the pre-registered 0.020 threshold is therefore not
evidence of a translation circuit, and no number in these files should be quoted.

Kept because the measurements are real and the contrast against the counterfactual
mean is itself worth reporting: same components, same eval, reference distribution
the only difference.
