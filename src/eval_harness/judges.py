"""LLM-as-a-judge wrappers.

These judges need an OpenAI API key and a network connection. Every call is
guarded: constructing a judge never fails, but invoking it without a key raises
a clear RuntimeError so callers can fall back to the offline heuristics in
metrics.py.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional


class LLMJudge:
    """LLM-as-a-judge wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily build the OpenAI client, failing loudly without a key."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY is not set; cannot run the LLM judge.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Run one chat completion and return the trimmed content."""
        response = self._get_client().chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _split_facts(answer: str) -> List[str]:
        """Split an answer into claim-sized sentences."""
        parts = re.split(r"[.!?\n]", answer)
        return [part.strip() for part in parts if len(part.split()) > 3]

    def judge_faithfulness(self, answer: str, context: str) -> float:
        """Fraction of claims the model confirms are supported by the context."""
        facts = self._split_facts(answer)
        if not facts:
            return 1.0
        supported = 0
        for fact in facts:
            verdict = self._chat(
                "Answer only YES or NO: is the following claim supported by the context?",
                f"Claim: {fact}\nContext: {context}",
            )
            if "YES" in verdict.upper():
                supported += 1
        return supported / len(facts)

    def judge_relevance(self, answer: str, question: str) -> float:
        """Model-rated 0.0-1.0 score for how well the answer addresses the question."""
        verdict = self._chat(
            "Rate how well the answer addresses the question on a scale of 0.0 to 1.0. "
            "Output only the number.",
            f"Question: {question}\nAnswer: {answer}",
        )
        try:
            return max(0.0, min(float(verdict), 1.0))
        except ValueError:
            return 0.0

    def llm_as_judge(self, question: str, answer: str, rubric: str) -> float:
        """Score an answer against a free-form rubric string."""
        verdict = self._chat(
            f"Score the answer against this rubric. Output only the number.\nRubric: {rubric}",
            f"Question: {question}\nAnswer: {answer}",
        )
        try:
            return max(0.0, float(verdict))
        except ValueError:
            return 0.0
