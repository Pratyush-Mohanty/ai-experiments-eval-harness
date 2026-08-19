"""Result serialization: JSON, CSV, and Markdown reporting.

All writers accept either an EvalRun object (from the runner) or a plain dict
(as loaded from a previously saved result file), so reporting and comparison
share one normalization path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from .runner import EvalRun

RunLike = Union[EvalRun, dict]


def _to_dict(run: RunLike) -> dict:
    """Normalize an EvalRun or a dict into a serializable result dict."""
    if isinstance(run, dict):
        return run
    return {
        "timestamp": run.timestamp,
        "n": len(run.items),
        "summary": run.summary,
        "items": [
            {
                "question": item.question,
                "answer": item.answer,
                "golden": item.golden,
                "context": item.context,
                "metrics": item.metrics,
            }
            for item in run.items
        ],
    }


def write_json(run: RunLike, path) -> Path:
    """Write the full result (summary + per-item metrics) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_dict(run), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def write_csv(run: RunLike, path) -> Path:
    """Write one row per item with a column per metric."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_dict(run)
    metric_names = sorted(data["summary"])
    fieldnames = ["question", "answer", "golden"] + metric_names
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in data["items"]:
            row = {
                "question": item["question"],
                "answer": item["answer"],
                "golden": item["golden"],
            }
            row.update(item["metrics"])
            writer.writerow(row)
    return path


def write_markdown(run: RunLike, path) -> Path:
    """Write a human-readable summary with a metrics table and per-item rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_dict(run)
    lines = ["# Eval Harness Report", ""]
    lines.append(f"- Timestamp: {data['timestamp']}")
    lines.append(f"- Items evaluated: {data['n']}")
    lines.append("")
    lines.append("## Metric summary")
    lines.append("")
    lines.append("| metric | mean |")
    lines.append("|---|---|")
    for name in sorted(data["summary"]):
        lines.append(f"| {name} | {data['summary'][name]} |")
    lines.append("")
    lines.append("## Per-item")
    lines.append("")
    metric_names = sorted(data["summary"])
    header = ["question"] + metric_names
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for item in data["items"]:
        row = [item["question"]]
        row.extend(str(item["metrics"].get(name, "")) for name in metric_names)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
