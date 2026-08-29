"""
How this subsystem refuses.

One error type, raised with a message that says what to do instead rather than
what went wrong. Every check in the schema is a mistake that otherwise surfaces
as a plausible wrong number somewhere downstream, so the message is the only
place the reader learns which mistake it was.

A common pipe could be: validate | ArtifactError | fix the card
"""

class ArtifactError(ValueError):
    """Raised when an artifact is incomplete, self-contradictory, or not one at all"""
