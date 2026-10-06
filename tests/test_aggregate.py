from app.aggregate import RunResult, aggregate


def run(i, verdict, conf="high", error=None):
    return RunResult(index=i, swapped=i % 2 == 1, verdict=verdict, confidence=conf, error=error)


def test_unanimous_high_confidence_is_confident():
    agg = aggregate([run(0, "B"), run(1, "B"), run(2, "B", "medium"), run(3, "B")])
    assert agg.status == "confident"
    assert agg.verdict == "B"
    assert agg.votes == {"A": 0, "B": 4}


def test_one_low_confidence_downgrades_to_review():
    agg = aggregate([run(0, "A"), run(1, "A", "low"), run(2, "A"), run(3, "A")])
    assert agg.status == "review"
    assert agg.verdict == "A"
    assert "low confidence" in agg.explanation


def test_three_to_one_majority_is_review():
    agg = aggregate([run(0, "A"), run(1, "A"), run(2, "B"), run(3, "A")])
    assert agg.status == "review"
    assert agg.verdict == "A"
    assert "disagreed" in agg.explanation


def test_even_split_is_unclear_with_no_verdict():
    agg = aggregate([run(0, "A"), run(1, "B"), run(2, "A"), run(3, "B")])
    assert agg.status == "unclear"
    assert agg.verdict is None


def test_failed_run_prevents_confident():
    runs = [run(0, "A"), run(1, "A"), run(2, "A"), run(3, None, error="rate limited")]
    agg = aggregate(runs)
    assert agg.status == "review"  # 3 of 4 planned agree
    assert agg.verdict == "A"
    assert "failed" in agg.explanation


def test_majority_is_of_planned_runs_not_successful_runs():
    # 2 agree, 2 failed: 2 is not a strict majority of 4 planned runs.
    runs = [run(0, "A"), run(1, "A"), run(2, None, error="x"), run(3, None, error="x")]
    agg = aggregate(runs)
    assert agg.status == "unclear"
    assert agg.verdict is None


def test_planned_can_exceed_returned_runs():
    agg = aggregate([run(0, "B"), run(1, "B")], planned=4)
    assert agg.status == "unclear"


def test_all_failed_is_error():
    agg = aggregate([run(0, None, error="boom"), run(1, None, error="boom")])
    assert agg.status == "error"
    assert agg.verdict is None
    assert "boom" in agg.explanation


def test_no_runs_is_error():
    agg = aggregate([], planned=4)
    assert agg.status == "error"


def test_single_run_is_never_confident():
    agg = aggregate([run(0, "A")])
    assert agg.status == "review"
    assert "one run" in agg.explanation


def test_two_runs_both_orders_can_be_confident():
    agg = aggregate([run(0, "A"), run(1, "A")])
    assert agg.status == "confident"


def test_representative_is_highest_confidence_majority_run():
    runs = [run(0, "B", "medium"), run(1, "B", "high"), run(2, "A", "high"), run(3, "B", "medium")]
    agg = aggregate(runs)
    assert agg.verdict == "B"
    assert agg.representative == 1


def test_representative_prefers_lowest_index_on_ties():
    agg = aggregate([run(0, "A"), run(1, "A"), run(2, "A")])
    assert agg.representative == 0


def test_unclear_still_has_a_representative_for_display():
    agg = aggregate([run(0, "A", "medium"), run(1, "B", "high")])
    assert agg.status == "unclear"
    assert agg.representative == 1


def test_verdict_with_invalid_letter_counts_as_failed():
    agg = aggregate([run(0, "A"), RunResult(1, True, verdict="C", confidence="high")])
    assert agg.succeeded == 1
    assert agg.status == "unclear"  # 1 of 2 planned runs is not a majority
