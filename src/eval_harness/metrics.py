"""Pure, offline metrics.

Every function here is deterministic and makes no API calls, so the metrics can
run in CI, in tests, and in offline batch jobs. Functions that need retrieval
lists (MRR, hit-rate) accept the ranked results and the relevant gold strings.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

STOPWORDS = {
    "i",
    "don",
    "t",
    "dont",
    "know",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "not",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "but",
    "you",
    "your",
    "we",
    "they",
    "them",
    "their",
    "our",
    "my",
    "me",
    "he",
    "she",
    "him",
    "her",
    "us",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "has",
    "have",
    "had",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "there",
    "here",
}


def exact_accuracy(answer: str, golden: str) -> float:
    """Return 1.0 when the answer exactly matches the golden answer."""
    if answer.strip().lower() == golden.strip().lower():
        return 1.0
    return 0.0


def contains_accuracy(answer: str, golden: str) -> float:
    """Return 1.0 when the golden answer appears inside the answer."""
    if golden.strip().lower() and golden.strip().lower() in answer.strip().lower():
        return 1.0
    return 0.0


def _sentences(text: str) -> List[str]:
    """Split text into non-trivial sentences."""
    parts = re.split(r"[.!?\n]", text)
    return [part.strip() for part in parts if len(part.split()) >= 2]


def _content_tokens(text: str) -> Set[str]:
    """Lowercased, stopword-stripped tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def faithfulness_check(answer: str, context: str) -> float:
    """Heuristic faithfulness: fraction of answer sentences grounded in context.

    A sentence is supported when at least half of its content tokens appear in
    the context. Sentences with no content tokens are ignored so that refusals
    like "I don't know." are not penalized.
    """
    sentences = _sentences(answer)
    if not sentences:
        return 1.0
    context_tokens = _content_tokens(context)
    supported = 0
    scored = 0
    for sentence in sentences:
        sentence_tokens = _content_tokens(sentence)
        if not sentence_tokens:
            continue
        scored += 1
        overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
        if overlap >= 0.5:
            supported += 1
    if scored == 0:
        return 1.0
    return supported / scored


def _normalize(entries: Iterable[str]) -> Set[str]:
    """Normalize a collection of retrieved/relevant strings for matching."""
    return {entry.strip().lower() for entry in entries if entry and entry.strip()}


def has_hit(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> bool:
    """Return True when any relevant result appears within the top-k."""
    relevant_set = _normalize(relevant)
    for item in retrieved[:k]:
        if item.strip().lower() in relevant_set:
            return True
    return False


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Mean reciprocal rank of the first relevant hit within the top-k."""
    relevant_set = _normalize(relevant)
    for rank, item in enumerate(retrieved[:k], start=1):
        if item.strip().lower() in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate(hits: Sequence[bool]) -> float:
    """Fraction of queries that had at least one relevant result."""
    if not hits:
        return 0.0
    return sum(1 for hit in hits if hit) / len(hits)