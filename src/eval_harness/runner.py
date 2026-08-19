"""Evaluation runner: drive a golden dataset through an injectable answer_fn.

The answer_fn is the seam where users plug in their own RAG pipeline. The runner
applies every configured metric per item and rolls the results into a summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from . import metrics as m
from .dataset import GoldenDataset

AnswerFn = Callable[[str, str], str]
MetricFn = Callable[[str, str, str, str], float]


def default_metrics() -> Dict[str, MetricFn]:
    """Offline, API-free metrics used when no custom metrics are supplied."""

    def exact(answer: str, golden: str, context: str, question: str) -> float:
        return m.exact_accuracy(answer, golden)

    def contains(answer: str, golden: str, context: str, question: str) -> float:
        return m.contains_accuracy(answer, golden)

    def faithfulness(answer: str, golden: str, context: str, question: str) -> float:
        return m.faithfulness_check(answer, context)

    return {
        "accuracy": exact,
        "contains": contains,
        "faithfulness": faithfulness,
    }


@dataclass
class ItemScore:
    """Per-item eval result: the answer produced plus its metric values."""

    question: str
    answer: str
    golden: str
    context: str
    metrics: Dict[str, float]


@dataclass
class EvalRun:
    """A complete evaluation run: items plus a rolled-up summary."""

    timestamp: str
    items: List[ItemScore]
    summary: Dict[str, float]


class EvalRunner:
    """Runs a golden dataset through an answer function and scores it."""

    def __init__(self, metrics: Optional[Dict[str, MetricFn]] = None):
        self.metrics = metrics or default_metrics()

    def run(self, dataset: GoldenDataset, answer_fn: AnswerFn) -> EvalRun:
        """Evaluate every golden item and return an EvalRun with a summary."""
        items: List[ItemScore] = []
        for item in dataset.to_list():
            answer = answer_fn(item.question, item.context)
            values = {
                name: round(
                    fn(answer, item.answer, item.context, item.question), 3
                )
                for name, fn in self.metrics.items()
            }
            items.append(
                ItemScore(
                    question=item.question,
                    answer=answer,
                    golden=item.answer,
                    context=item.context,
                    metrics=values,
                )
            )
        summary: Dict[str, float] = {}
        if items:
            summary = {
                name: round(
                    sum(item.metrics[name] for item in items) / len(items), 3
                )
                for name in self.metrics
            }
        return EvalRun(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            items=items,
            summary=summary,
        )
