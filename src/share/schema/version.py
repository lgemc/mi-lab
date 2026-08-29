"""
What this format is called, and which one of it this is.

Kept in its own module because storage reads it to stamp a card, the schema
reads it to refuse a card written by another version, and neither should have
to import the other to find out.

The version is refused rather than negotiated. A reader handed a card from a
version it does not implement stops and says so, because "safely ignore the
part I do not understand" is exactly the silent drop the format exists to
prevent.

A common pipe could be: save | FORMAT + VERSION | load
"""

FORMAT = "mia"
VERSION = "0.2"
