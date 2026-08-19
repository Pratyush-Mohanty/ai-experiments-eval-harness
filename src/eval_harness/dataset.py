"""Golden dataset loading and validation.

The golden dataset is the fixed "lab exam" every prompt, retriever, and model
change must pass. Items are validated with pydantic so bad data fails loudly
before any metric is computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    """A single question/answer/context triple from the golden dataset."""

    question: str
    answer: str
    context: str
    tags: List[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    """A validated collection of golden items."""

    items: List[GoldenItem] = Field(default_factory=list)

    @classmethod
    def from_json(cls, path) -> "GoldenDataset":
        """Load and validate a golden dataset from a JSON file or array."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return cls(items=[GoldenItem.model_validate(item) for item in raw])
        return cls.model_validate(raw)

    @classmethod
    def from_dicts(cls, dicts: List[dict]) -> "GoldenDataset":
        """Build a golden dataset from a list of dictionaries."""
        return cls(items=[GoldenItem.model_validate(item) for item in dicts])

    def to_list(self) -> List[GoldenItem]:
        """Return the underlying items as a plain list."""
        return list(self.items)
