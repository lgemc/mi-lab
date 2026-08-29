from dataclasses import dataclass

"""
The interval a fraction is a fraction of.

A recovery of 0.9 means "restored 90% of the distance from corrupted back to
clean". Against a corruption that barely moved the model, that is noise scaled
up to look like a finding. Any normalized score is a share of an interval
somebody chose, and shipping it without that interval is shipping a percentage
of an unstated whole.

A common pipe could be: baselines | Span | recovery
"""

@dataclass(frozen=True)
class Span:
    """The two baselines every fractional number here is measured between

    A recovery is 0 at `corrupted` and 1 at `clean`. Quoting one without the
    other is quoting a percentage of an unstated whole.

    `metric` names the scale the two baselines are on, and stays prose because
    the choice of scale is the experiment's, not the format's. The only one
    this repository writes is "logit_difference" -- the logit of the correct
    answer minus the logit of the distractor. A cross-entropy or a normalized
    KL would be as valid here, which is exactly why it is not an enum.
    """
    metric: str
    clean: float
    corrupted: float

    @property
    def span(self) -> float:
        return self.clean - self.corrupted
