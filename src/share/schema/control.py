from dataclasses import dataclass

"""
One check run against a claim, and what it returned.

A control is not a metric. A metric describes the result; a control describes
an attempt to make the result go away -- ablating the circuit against a task it
does not claim to explain, or steering with a norm-matched random vector at the
same layer. Keeping them apart is what lets the schema insist the attempt was
recorded even when nobody made one.

A common pipe could be: ablate elsewhere | Control | Controls
"""

@dataclass(frozen=True)
class Control:
    """One check run against a claim, and what it returned

    `name` says what was controlled against and stays prose: under
    `cross_task` it is the other task's name ("greater-than", "docstring"),
    under `random_baseline` it is what the random thing was ("norm-matched
    random direction", "random circuit of the same size"). There is no closed
    set because there is no closed set of things worth ruling out.

    `metric` says which scale `value` is on, and takes the same names a Metric
    does -- "recovery", "logits", "auc", "share". A control quoted in units
    nobody stated is the defect Metric exists to fix, so it is required here
    for the same reason.
    """
    name: str
    metric: str
    value: float
    notes: str = ""
