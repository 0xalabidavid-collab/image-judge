import asyncio
import io

from PIL import Image

from app.config import JudgeConfig
from app.images import near_identical, prepare_image
from app.judge import JudgeError, Task, evaluate
from app.rubric import build_user_content, system_prompt, RUBRICS
from conftest import FakeJudge, png_bytes

CFG = JudgeConfig(model="fake", runs=4, rubric_version="v1", effort="high")


def task(a_color=(220, 30, 30), b_color=(30, 160, 60)):
    return Task(
        prompt="Make the circle red",
        originals=[prepare_image(png_bytes((40, 90, 200)))],
        a=prepare_image(png_bytes(a_color)),
        b=prepare_image(png_bytes(b_color)),
    )


def test_half_the_runs_are_swapped():
    judge = FakeJudge()
    result = asyncio.run(evaluate(task(), CFG, judge))
    assert sorted(c["swapped"] for c in judge.calls) == [False, False, True, True]
    assert [r["swapped"] for r in result["runs"]] == [False, True, False, True]
    assert result["aggregate"]["status"] == "confident"
    assert result["aggregate"]["verdict"] == "A"


def test_position_biased_judge_is_unclear_not_forced():
    # Always picks whatever was shown first: A in normal runs, B in swapped runs.
    judge = FakeJudge(lambda t, swapped: ("B" if swapped else "A", "high"))
    result = asyncio.run(evaluate(task(), CFG, judge))
    assert result["aggregate"]["status"] == "unclear"
    assert result["aggregate"]["verdict"] is None


def test_cache_reuses_runs_by_index(db):
    judge = FakeJudge()
    asyncio.run(evaluate(task(), CFG, judge, db=db))
    assert len(judge.calls) == 4
    # Same task with more runs: first 4 come from cache, only 2 new calls.
    result = asyncio.run(evaluate(task(), CFG.with_overrides(runs=6), judge, db=db))
    assert len(judge.calls) == 6
    assert [r["cached"] for r in result["runs"]] == [True] * 4 + [False] * 2


def test_cache_is_keyed_on_rubric(db):
    judge = FakeJudge()
    asyncio.run(evaluate(task(), CFG, judge, db=db))
    asyncio.run(evaluate(task(), CFG.with_overrides(rubric_version="v2"), judge, db=db))
    assert len(judge.calls) == 8


def test_use_cache_false_calls_again(db):
    judge = FakeJudge()
    asyncio.run(evaluate(task(), CFG, judge, db=db))
    asyncio.run(evaluate(task(), CFG, judge, db=db, use_cache=False))
    assert len(judge.calls) == 8


def test_fatal_error_stops_remaining_runs():
    judge = FakeJudge(error=JudgeError("bad key", fatal=True))
    result = asyncio.run(evaluate(task(), CFG, judge))
    assert result["aggregate"]["status"] == "error"
    assert len(judge.calls) < 4 or all(r["error"] for r in result["runs"])
    assert any("bad key" in (r["error"] or "") for r in result["runs"])


def test_non_fatal_errors_are_recorded_per_run():
    calls = {"n": 0}

    class Flaky(FakeJudge):
        async def judge_once(self, task, config, swapped, notes):
            calls["n"] += 1
            if calls["n"] == 1:
                raise JudgeError("rate limited")
            return await super().judge_once(task, config, swapped, notes)

    result = asyncio.run(evaluate(task(), CFG, Flaky()))
    assert result["aggregate"]["status"] == "confident"
    assert sum(1 for r in result["runs"] if r["error"]) == 0
    assert calls["n"] == CFG.runs + 1


def test_slow_runs_respect_evaluation_deadline(monkeypatch):
    class SlowJudge:
        async def judge_once(self, task, config, swapped, notes):
            await asyncio.sleep(1)
            raise AssertionError("deadline did not cancel the call")

    monkeypatch.setattr("app.judge.settings.api_timeout_s", 0.01)
    result = asyncio.run(evaluate(task(), CFG.with_overrides(runs=2), SlowJudge()))
    assert result["aggregate"]["status"] == "error"
    assert all("evaluation deadline" in run["error"] for run in result["runs"])


def test_identical_results_short_circuit_without_api_calls():
    judge = FakeJudge()
    result = asyncio.run(evaluate(task(a_color=(1, 2, 3), b_color=(1, 2, 3)), CFG, judge))
    assert judge.calls == []
    assert result["aggregate"]["status"] == "unclear"


def test_unchanged_result_is_flagged_to_the_model():
    judge = FakeJudge()
    t = task(a_color=(220, 30, 30), b_color=(40, 90, 200))  # B == original
    asyncio.run(evaluate(t, CFG, judge))
    notes = judge.calls[0]["notes"]
    assert notes == ["Result B is visually identical to Original 1 (no visible edit)."]


def test_near_identical_tolerates_jpeg_but_not_small_edits():
    base = png_bytes((40, 90, 200), size=(800, 600))
    img = Image.open(io.BytesIO(base))
    jpg = io.BytesIO()
    img.save(jpg, format="JPEG", quality=85)
    edited = png_bytes((40, 90, 200), size=(800, 600), text="HELLO")
    a, b, c = prepare_image(base), prepare_image(jpg.getvalue()), prepare_image(edited)
    assert near_identical(a, b)
    assert not near_identical(a, c)


def test_prompt_labels_follow_the_images():
    orig, a, b = object(), object(), object()
    content = build_user_content("p", [orig], [("B", b), ("A", a)], [])
    texts = [c if isinstance(c, str) else None for c in content]
    i_b, i_a = texts.index("RESULT B:"), texts.index("RESULT A:")
    assert i_b < i_a
    assert content[i_b + 1] is b and content[i_a + 1] is a
    assert texts.index("ORIGINAL 1:") < i_b


def test_all_rubrics_load():
    for v in RUBRICS:
        assert "RESULT A" in system_prompt(v)


def test_difference_map_local_edit_and_aspect_mismatch():
    from app.images import difference_map
    orig = prepare_image(png_bytes((40, 90, 200), size=(400, 300)))
    small_edit = prepare_image(png_bytes((40, 90, 200), size=(400, 300), text="HELLO"))
    recolour = prepare_image(png_bytes((220, 30, 30), size=(400, 300)))
    heat, changed = difference_map(small_edit, orig)
    assert heat is not None and 0 < changed < 0.05            # only the text area lit up
    _, changed_big = difference_map(recolour, orig)
    assert changed_big > changed                              # the whole circle changed
    wide = prepare_image(png_bytes(size=(600, 300)))
    assert difference_map(wide, orig) == (None, None)         # different framing: no map


def test_v3_sends_difference_maps_and_uses_v3_schema():
    from app.judge import judge_content
    from app.rubric import JUDGMENT_SCHEMA_V3, schema_for
    assert schema_for("v3") is JUDGMENT_SCHEMA_V3
    assert schema_for("v5") is JUDGMENT_SCHEMA_V3
    assert schema_for("v1") is not JUDGMENT_SCHEMA_V3
    assert JUDGMENT_SCHEMA_V3["required"][:3] == ["originals_summary", "edit_type", "first_impression"]

    judge = FakeJudge()
    t = task()
    asyncio.run(evaluate(t, CFG.with_overrides(rubric_version="v3"), judge))
    assert [label for label, _ in t.difference_maps] == ["A", "B"]
    assert any("differs over about" in n for n in judge.calls[0]["notes"])
    texts = [c for c in judge_content(t, True, []) if isinstance(c, str)]
    maps = [x for x in texts if x.startswith("DIFFERENCE MAP")]
    assert maps[0].startswith("DIFFERENCE MAP, Result A") and maps[1].startswith("DIFFERENCE MAP, Result B")


def test_v1_gets_no_difference_maps():
    t = task()
    asyncio.run(evaluate(t, CFG, FakeJudge()))
    assert t.difference_maps is None


def test_v3_rubric_contains_project_guidelines():
    text = system_prompt("v3")
    for phrase in ("Prompt compliance is where you start, not where you stop", "three legs",
                   "Global edits", "circle blob", "stock look", "happier to receive"):
        assert phrase in text
