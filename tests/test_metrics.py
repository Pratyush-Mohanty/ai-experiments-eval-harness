"""Tests for the pure, offline metrics in eval_harness.metrics."""

from eval_harness import metrics as m


def test_exact_accuracy_positive():
    assert m.exact_accuracy("hello world", "hello world") == 1.0


def test_exact_accuracy_case_insensitive():
    assert m.exact_accuracy("  Hello World ", "hello world") == 1.0


def test_exact_accuracy_negative():
    assert m.exact_accuracy("hello world", "hello") == 0.0


def test_contains_accuracy_positive():
    assert m.contains_accuracy("the answer is hello world", "hello world") == 1.0


def test_contains_accuracy_case_insensitive():
    assert m.contains_accuracy("The Answer Is Hello World", "hello world") == 1.0


def test_contains_accuracy_negative():
    assert m.contains_accuracy("completely different", "hello") == 0.0


def test_mrr_at_k_first_rank():
    assert m.mrr_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_mrr_at_k_third_rank():
    assert m.mrr_at_k(["a", "b", "c"], ["c"], k=3) == 1.0 / 3.0


def test_mrr_at_k_rank_beyond_k():
    assert m.mrr_at_k(["a", "c", "b"], ["c"], k=1) == 0.0


def test_mrr_at_k_no_hit():
    assert m.mrr_at_k(["a", "b"], ["z"], k=3) == 0.0


def test_has_hit_positive():
    assert m.has_hit(["a", "b"], ["b"], k=2) is True


def test_has_hit_negative():
    assert m.has_hit(["a", "b"], ["z"], k=2) is False


def test_hit_rate_mixed():
    assert m.hit_rate([True, False, True]) == 2.0 / 3.0


def test_hit_rate_empty():
    assert m.hit_rate([]) == 0.0
