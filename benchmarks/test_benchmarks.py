"""Performance benchmarks for gemhall metrics, judge, and prompts modules."""

import pytest

from gemhall.judge import judge_validity, normalize, numbers_equal, str_or_num_equal
from gemhall.metrics import Record, aggregate, behavior_checks, score_item
from gemhall.prompts import build_conf_prompt, is_idk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_records(n: int, thresholds: list[float]) -> list[Record]:
    """Generate a list of synthetic Record objects for benchmarking."""
    records = []
    for i in range(n):
        for t in thresholds:
            abstained = i % 5 == 0
            correct = i % 3 != 0
            records.append(
                Record(
                    id=str(i),
                    t=t,
                    question=f"What is {i} + {i}?",
                    gold=str(i + i),
                    unknown_ok=False,
                    pred="IDK" if abstained else str(i + i if correct else i),
                    abstained=abstained,
                    correct=correct,
                    score=score_item(
                        answered=not abstained, correct=correct, t=t
                    ),
                )
            )
    return records


THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
SMALL_RECORDS = _make_records(20, THRESHOLDS)
LARGE_RECORDS = _make_records(200, THRESHOLDS)


# ---------------------------------------------------------------------------
# score_item benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_score_item_correct_answer():
    """Benchmark score_item for a correct, answered case."""
    score_item(answered=True, correct=True, t=0.5)


@pytest.mark.benchmark
def test_score_item_incorrect_answer():
    """Benchmark score_item for an incorrect, answered case."""
    score_item(answered=True, correct=False, t=0.7)


@pytest.mark.benchmark
def test_score_item_abstained():
    """Benchmark score_item when the model abstains."""
    score_item(answered=False, correct=False, t=0.5)


# ---------------------------------------------------------------------------
# aggregate benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_aggregate_small():
    """Benchmark aggregate over a small record set."""
    aggregate(SMALL_RECORDS)


@pytest.mark.benchmark
def test_aggregate_large():
    """Benchmark aggregate over a large record set."""
    aggregate(LARGE_RECORDS)


# ---------------------------------------------------------------------------
# behavior_checks benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_behavior_checks():
    """Benchmark behavior_checks on aggregated metrics."""
    metrics = aggregate(LARGE_RECORDS)
    behavior_checks(metrics)


# ---------------------------------------------------------------------------
# judge benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_normalize():
    """Benchmark normalize on a typical prediction string."""
    normalize("  The answer is 42.  ")


@pytest.mark.benchmark
def test_numbers_equal_matching():
    """Benchmark numbers_equal with matching numeric strings."""
    numbers_equal("42.0", "42")


@pytest.mark.benchmark
def test_str_or_num_equal():
    """Benchmark str_or_num_equal with a matching pair."""
    str_or_num_equal("Paris", "  paris  ")


@pytest.mark.benchmark
def test_judge_validity_exact_correct():
    """Benchmark judge_validity for a correct exact-match case."""
    judge_validity(pred="Paris", gold="Paris", unknown_ok=False)


@pytest.mark.benchmark
def test_judge_validity_unknown_ok():
    """Benchmark judge_validity when unknown_ok is True."""
    judge_validity(pred="IDK", gold="Paris", unknown_ok=True)


# ---------------------------------------------------------------------------
# prompts benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_build_conf_prompt():
    """Benchmark building a confidence-calibrated prompt."""
    build_conf_prompt("What is the capital of France?", 0.7)


@pytest.mark.benchmark
def test_is_idk_positive():
    """Benchmark is_idk on a positive IDK token."""
    is_idk("IDK")


@pytest.mark.benchmark
def test_is_idk_negative():
    """Benchmark is_idk on a non-IDK answer."""
    is_idk("Paris")
