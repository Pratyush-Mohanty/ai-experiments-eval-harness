"""Compare two evaluation runs and flag metric regressions and improvements."""

from __future__ import annotations

from typing import Union

from .reporting import RunLike, _to_dict


def compare_runs(run_a: RunLike, run_b: RunLike) -> dict:
    """Diff two runs, returning per-metric before/after/delta/status.

    Status is ``improved``, ``regressed``, ``unchanged`` (within a small
    epsilon), or ``missing`` when a metric exists in only one run.
    """
    a = _to_dict(run_a)
    b = _to_dict(run_b)
    metric_names = sorted(set(a["summary"]) | set(b["summary"]))
    metrics: dict = {}
    for name in metric_names:
        before = a["summary"].get(name)
        after = b["summary"].get(name)
        if before is None or after is None:
            metrics[name] = {
                "before": before,
                "after": after,
                "delta": None,
                "status": "missing",
            }
            continue
        delta = after - before
        if delta > 0.0005:
            status = "improved"
        elif delta < -0.0005:
            status = "regressed"
        else:
            status = "unchanged"
        metrics[name] = {
            "before": before,
            "after": after,
            "delta": round(delta, 3),
            "status": status,
        }
    return {
        "a": {"timestamp": a["timestamp"], "n": a["n"]},
        "b": {"timestamp": b["timestamp"], "n": b["n"]},
        "metrics": metrics,
    }
