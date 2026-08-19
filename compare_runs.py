"""CLI for diffing two saved evaluation results.

Prints a before/after table for every summary metric and flags regressions.
Exits with code 1 when any metric regressed so it can gate a CI pipeline.

Usage:  py compare_runs.py results/before.json results/after.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eval_harness.compare import compare_runs


def load_run(path: str) -> dict:
    """Load a saved result JSON file as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: py compare_runs.py <run_a.json> <run_b.json>")
        return 2
    report = compare_runs(load_run(sys.argv[1]), load_run(sys.argv[2]))
    print(f"Run A: {report['a']['timestamp']} (n={report['a']['n']})")
    print(f"Run B: {report['b']['timestamp']} (n={report['b']['n']})")
    print("")
    header = f"{'metric':<14}{'before':>10}{'after':>10}{'delta':>10}  status"
    print(header)
    print("-" * len(header))
    regressions = []
    for name, info in sorted(report["metrics"].items()):
        before = "n/a" if info["before"] is None else f"{info['before']:.3f}"
        after = "n/a" if info["after"] is None else f"{info['after']:.3f}"
        delta = "n/a" if info["delta"] is None else f"{info['delta']:+.3f}"
        print(f"{name:<14}{before:>10}{after:>10}{delta:>10}  {info['status']}")
        if info["status"] == "regressed":
            regressions.append(name)
    print("")
    if regressions:
        print(f"REGRESSIONS: {', '.join(regressions)}")
        return 1
    print("No regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
