"""Read a running journal: where it is, what it costs, where the curve is going.

`tail -f metrics.jsonl` is already a live view and this does not replace it --
it is what you want when the question is "is this worth waiting for" rather
than "what was the last line". Three things a raw tail cannot do: divide
elapsed by step to project a finish, put the loss terms beside each other so a
ramping price is visible as the thing moving them, and say whether the run is
still alive or died at step 1378 an hour ago.

Reads only. Nothing here writes to the journal, so pointing it at a run in
flight cannot disturb it, and pointing it at a finished one works the same way.

A common pipe could be: journals | latest | progress | rows | table

Run: uv run python -m scripts.watch                       # the newest journal
     uv run python -m scripts.watch --list                # every journal
     uv run python -m scripts.watch <dir> --rows 20
     uv run python -m scripts.watch <dir> --follow        # redraw until it ends
"""

import argparse
import time

from src.telemetry.journal import (
    TelemetryError,
    env_root,
    journals,
    latest,
    progress,
    read_metrics,
    read_run,
)
from src.telemetry.observe import duration

COLUMNS = ("step", "elapsed", "loss", "faith", "open", "complete", "density", "hard_accuracy", "price", "noise")

def cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # Density runs to the millionths and a price runs to the thousands, so
        # one format for both prints either as 0.00 or as scientific noise.
        if value and abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)

def table(rows, columns) -> str:
    present = [name for name in columns if any(row.get(name) is not None for row in rows)]
    extra = sorted({key for row in rows for key in row} - set(columns))
    present.extend(extra)
    widths = {name: max(len(name), *(len(cell(row.get(name))) for row in rows))
              for name in present} if rows else {}
    head = "  ".join(name.rjust(widths[name]) for name in present)
    body = ["  ".join(cell(row.get(name)).rjust(widths[name]) for name in present)
            for row in rows]
    return "\n".join([head, "-" * len(head), *body])

def show(directory, count: int) -> dict:
    run, state = read_run(directory), progress(directory)
    rows = read_metrics(directory)
    params = run.get("params", {})
    band = params.get("layers")
    print(f"{run.get('name', directory)}  [{state['status']}]")
    print(f"  {directory}")
    print(f"  {params.get('gates', 0) / 1e6:.0f}M gates · "
          f"{'all layers' if band is None else f'layers {band}'} · "
          f"{params.get('distinct_prompts', '?')} distinct prompts")
    if state["total"]:
        print(f"  step {state['step']} of {state['total']} ({state['fraction']:.1%}) · "
              f"{duration(state['elapsed'])} elapsed · eta {duration(state['eta_seconds'])} · "
              f"{state['seconds_per_step']:.2f}s/step")
    if run.get("error"):
        print(f"  error: {run['error']}")
    print()
    # First and last: the curve's endpoints are what says whether anything is
    # happening, and on a 2000-step run the middle is what the tail already shows.
    window = rows[:1] + rows[-count:] if len(rows) > count else rows
    if window:
        print(table(window, COLUMNS))
    else:
        print("  no metrics logged yet")
    return state

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default=None,
                        help="a journal directory; default is the newest under MI_LAB_JOURNALS")
    parser.add_argument("--rows", type=int, default=12, help="how many recent rows to show")
    parser.add_argument("--follow", action="store_true", help="redraw until the run ends")
    parser.add_argument("--every", type=float, default=15.0, help="seconds between redraws")
    parser.add_argument("--list", action="store_true", dest="listing")
    args = parser.parse_args()

    root = env_root()
    if args.listing:
        found = journals(root)
        if not found:
            raise SystemExit(f"no journals under {root}; set MI_LAB_JOURNALS or run something first")
        for directory in found:
            try:
                run = read_run(directory)
                print(f"{run.get('status', '?'):>9}  {directory}")
            except TelemetryError as error:
                print(f"{'unreadable':>9}  {directory}  ({error})")
        return

    directory = args.directory or latest(root)
    if directory is None:
        raise SystemExit(
            f"no journals under {root}. Set MI_LAB_JOURNALS to where they live, or name a "
            f"directory as the first argument."
        )
    while True:
        state = show(directory, args.rows)
        if not args.follow or state["status"] != "running":
            return
        time.sleep(max(1.0, args.every))
        print()

if __name__ == "__main__":
    main()
