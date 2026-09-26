"""The WorkspaceBench leaderboard: every readout run so far, ranked per model.

Reads every `results/wsbench/<method>-<config>.json` that `wsbench_baseline`
wrote and writes `leaderboard.md` and `leaderboard.json` beside them. Loads no
model, so it is seconds and can be rerun after every method lands. Tracked
like everything else: the pooled numbers go to MLflow as `<config>.<method>.*`
and both files as artifacts.

Run: uv run python -m scripts.wsbench_leaderboard
"""

import json
import sys
from pathlib import Path

from src.domains.lm.analysis.leaderboard import markdown, pooled, rank
from src.domains.lm.data.workspacebench import SINGLE_TOKEN_BANKS
from src.telemetry.observe import log
from src.telemetry.tracking import track

RESULTS = Path("results/wsbench")
EXPERIMENT = "mi-lab-wsbench"
ROW_METRICS = ("clean_hit_rate", "hit_rate", "leak_rate", "median_first_hit_layer", "items")


def main() -> None:
    files = sorted(path for path in RESULTS.glob("*.json") if not path.name.startswith("leaderboard"))
    rows = [pooled(json.loads(path.read_text())) for path in files]
    rows = [row for row in rows if row["items"]]
    if not rows:
        raise SystemExit(f"no results under {RESULTS}; run scripts.wsbench_baseline first")
    ranked = rank(rows, expected=SINGLE_TOKEN_BANKS)
    with track("leaderboard", EXPERIMENT, outputs=[RESULTS / "leaderboard.md", RESULTS / "leaderboard.json"],
               params={"runs": len(ranked), "sources": ",".join(path.name for path in files)},
               tags={"script": "wsbench_leaderboard"}) as tracker:
        (RESULTS / "leaderboard.json").write_text(json.dumps(ranked, indent=2) + "\n")
        (RESULTS / "leaderboard.md").write_text(markdown(ranked))
        tracker.log(0, {f"{row['config']}.{row['method']}.{key}": row[key]
                        for row in ranked if row["complete"] for key in ROW_METRICS})
        tracker.flush()
    print(markdown(ranked))
    log(f"-> {RESULTS / 'leaderboard.md'}")


if __name__ == "__main__":
    sys.exit(main())
