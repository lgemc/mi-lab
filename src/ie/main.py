import argparse

from . import registry
from .app import Explorer
from .session import Session

"""
Starting the explorer, and the few things worth saying before it opens.

Arguments here are the session's starting point and nothing else -- everything
they set is changeable from inside with `:model`, `:prompt` and `:root`, so
this is a convenience rather than a second interface. The CLI stays the place
where things are computed; `ie` is the place where what was computed is read.

Run with: python -m src.ie   (or the `ie` script)
"""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ie",
        description="Interpretability explorer: navigate a model, its runs and the artifacts they shipped.",
    )
    parser.add_argument("resource", nargs="?", default=registry.ROOT,
                        help=f"view to open on; one of {', '.join(sorted(registry.VIEWS))}")
    parser.add_argument("--model", default="gpt2-small", help="config to point the session at")
    parser.add_argument("--root", default="outputs", help="where to look for runs and artifacts")
    parser.add_argument("--prompt", default=None, help="the text the token and activation views work over")
    return parser

def run(argv=None) -> None:
    """Open the explorer on a session built from the arguments"""
    args = build_parser().parse_args(argv)
    session = Session(model_id=args.model, root=args.root)
    if args.prompt:
        session.set_prompt(args.prompt)
    Explorer(session=session, start=args.resource).run()

if __name__ == "__main__":
    run()
