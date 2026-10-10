"""Phase 3: SFT's update against SDFT's, read off their decompositions

`phase3_delta_pd` decomposes each method's science-stage update into rank-1
pieces and records, on training text, how often each piece's gate opens on
the earlier task (tool use), the later one (science) and prose. 02 §II.2
deduced that SDFT forgets less because its update stays inside the computation
the model already does; in a decomposition of the update that has a direct
reading: **how much of the update the earlier task runs through, and how much
of that it shares with the later one.**

Per decomposition:

- pieces alive, and pieces open per position on each kind of text;
- the earlier task's **load**: the update's pieces open per tool-use position,
  against the later task's own count -- how much of the update tool use is
  exposed to;
- **sharing**: of the pieces open on tool use (rate above `--floor`), the
  share also open on science at no less than half their tool-use rate, and
  the overlap (Jaccard) of the two sets;
- the reverts: forgetting and learning kept (log-prob) for the 256 pieces most
  used on tool use and for 256 random live ones.

Reads results only; no model is loaded.

Run: uv run python -m scripts.phase3_delta_compare
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.telemetry.observe import log, set_log_file
from src.telemetry.tracking import tracked_main

PAIRS = {
    "SFT s42": "diff-qwen3-0.6b-sft-tooluse-to-science",
    "SDFT s42": "diff-qwen3-0.6b-sdft-tooluse-to-science",
    "SFT s43": "diff-qwen3-0.6b-sft-tooluse-to-science-s43",
    "SDFT s43": "diff-qwen3-0.6b-sdft-tooluse-to-science-s43",
}
ROOT = Path("results")
OUT = ROOT / "phase3-delta-compare.json"


def describe(report: Dict[str, Any], floor: float) -> Dict[str, Any]:
    gates = report["gate_open"]
    tool, science, prose = gates["earlier"], gates["later"], gates["prose"]
    used = [i for i, rate in enumerate(tool) if rate > floor]
    on_science = {i for i, rate in enumerate(science) if rate > floor}
    shared = [i for i in used if science[i] >= 0.5 * tool[i]]
    top = sorted(range(len(tool)), key=lambda i: -tool[i])[:256]
    variants = report["variants"]

    def kept(name: str) -> Dict[str, float]:
        entry = variants.get(name, {})
        return {"forgetting": entry.get("forgetting_kept"), "learning": entry.get("learning_kept")}

    return {
        "alive": report["alive"],
        "open_per_position": report["open_per_position"],
        "used_on_tooluse": len(used),
        "shared_share": len(shared) / len(used) if used else None,
        "jaccard_tooluse_science": len(set(used) & on_science) / len(set(used) | on_science) if used else None,
        "top256_open": {"tooluse": mean(tool[i] for i in top), "science": mean(science[i] for i in top),
                        "prose": mean(prose[i] for i in top)},
        "revert256_used": kept("revert 256 most used on earlier"),
        "revert256_random": kept("revert 256 random alive"),
        "gated_with_leftover": kept("gated, with leftover"),
        "leftover_alone": kept("leftover alone"),
    }


def main(argv: List[str]) -> int:
    floor = float(next((w.split("=", 1)[1] for w in argv if w.startswith("--floor=")), 0.01))
    set_log_file(ROOT / "phase3-delta-compare.log")
    table = {}
    for label, directory in PAIRS.items():
        path = ROOT / directory / "phase3-delta-pd.json"
        if not path.exists():
            log(f"{label}: no decomposition at {path}, skipped")
            continue
        table[label] = describe(json.loads(path.read_text()), floor)
    OUT.write_text(json.dumps({"floor": floor, "pairs": table}, indent=1))
    for label, row in table.items():
        top, opened = row["top256_open"], row["open_per_position"]
        reverted, random_ = row["revert256_used"], row["revert256_random"]
        log(f"{label:9s} alive {row['alive']:5d}  open/pos tool {opened['earlier']:5.0f} "
            f"sci {opened['later']:5.0f} prose {opened['prose']:5.0f}  "
            f"used on tool {row['used_on_tooluse']:5d}  shared {row['shared_share']:.2f}  "
            f"jaccard {row['jaccard_tooluse_science']:.2f}  "
            f"top256 open tool {top['tooluse']:.3f} sci {top['science']:.3f}  "
            f"revert256 forg {reverted['forgetting']:+.2f} learn {reverted['learning']:+.2f}  "
            f"random forg {random_['forgetting']:+.2f}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MI_LAB_RESULTS", str(ROOT))
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[OUT]))
