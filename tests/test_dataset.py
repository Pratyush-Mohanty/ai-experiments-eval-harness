"""Tests for golden dataset loading and pydantic validation."""

import json

import pytest
from pydantic import ValidationError

from eval_harness.dataset import GoldenDataset, GoldenItem


def test_from_json_loads_valid_file(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {"question": "Q1?", "answer": "A1", "context": "C1"},
                {"question": "Q2?", "answer": "A2", "context": "C2"},
            ]
        ),
        encoding="utf-8",
    )
    dataset = GoldenDataset.from_json(path)
    assert len(dataset.to_list()) == 2
    assert dataset.to_list()[0].question == "Q1?"
    assert dataset.to_list()[1].context == "C2"


def test_from_json_accepts_object_root(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps({"items": [{"question": "Q?", "answer": "A", "context": "C"}]}),
        encoding="utf-8",
    )
    dataset = GoldenDataset.from_json(path)
    assert len(dataset.items) == 1


def test_rejects_missing_fields():
    with pytest.raises(ValidationError):
        GoldenItem.model_validate({"question": "Q?", "answer": "A"})
    with pytest.raises(ValidationError):
        GoldenDataset.from_dicts([{"question": "Q?", "context": "C"}])


def test_rejects_wrong_types():
    with pytest.raises(ValidationError):
        GoldenItem.model_validate({"question": 42, "answer": "A", "context": "C"})


def test_validates_against_pydantic():
    item = GoldenItem.model_validate({"question": "Q?", "answer": "A", "context": "C"})
    assert isinstance(item, GoldenItem)


def test_to_list_returns_golden_items():
    dataset = GoldenDataset.from_dicts(
        [{"question": "Q?", "answer": "A", "context": "C", "tags": ["x"]}]
    )
    items = dataset.to_list()
    assert len(items) == 1
    assert items[0].tags == ["x"]
