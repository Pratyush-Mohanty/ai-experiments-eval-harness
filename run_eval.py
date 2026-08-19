"""CLI entry point for the eval harness.

Runs the golden dataset through a default answer_fn (a live OpenAI chat call
when OPENAI_API_KEY is set, otherwise a stub that answers "I don't know."),
scores every item, and writes results/evals_latest.{json,csv,md}.

Usage:  py run_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eval_harness.dataset import GoldenDataset
from eval_harness.reporting import write_csv, write_json, write_markdown
from eval_harness.runner import EvalRunner, default_metrics

RESULTS_DIR = Path("results")


def make_answer_fn():
    """Return the default answer_fn: live LLM when possible, else a stub."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        def answer_fn(question: str, context: str) -> str:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer using ONLY the context. If the context "
                        "does not answer the question, say 'I don't know'. Be concise.",
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        return answer_fn

    def answer_fn(question: str, context: str) -> str:
        return "I don't know."

    return answer_fn


def main() -> int:
    dataset = GoldenDataset.from_json("datasets/golden_dataset.json")
    runner = EvalRunner(metrics=default_metrics())
    run = runner.run(dataset, make_answer_fn())
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = write_json(run, RESULTS_DIR / "evals_latest.json")
    csv_path = write_csv(run, RESULTS_DIR / "evals_latest.csv")
    md_path = write_markdown(run, RESULTS_DIR / "evals_latest.md")
    print(f"Evaluated {len(run.items)} items.")
    for name, value in run.summary.items():
        print(f"  {name}: {value}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
