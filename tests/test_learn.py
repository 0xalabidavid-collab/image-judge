"""Train / test / judge-only runs over task sets, with a scripted fake judge."""

import asyncio

import pytest

from app import benchmark as bm
from app import learn
from app.config import JudgeConfig
from conftest import make_judgment, png_bytes

CFG = JudgeConfig(model="fake", runs=2, rubric_version="v1", effort="high")


class LearningJudge:
    """Judges 'A' unless it has a lesson saying otherwise; writes and merges lessons on request."""

    def __init__(self, answer="A", lesson_fix="B", flag_label=False):
        self.answer, self.lesson_fix, self.flag_label = answer, lesson_fix, flag_label
        self.judge_calls, self.lesson_calls, self.merge_calls = [], [], []

    async def judge_once(self, task, config, swapped, notes):
        verdict = self.lesson_fix if config.lessons else self.answer
        self.judge_calls.append({"lessons": config.lessons, "guidelines": config.guidelines})
        return {"judgment": make_judgment(verdict, "high"), "model": "fake", "usage": {}}

    async def complete_json(self, system, content, schema, config):
        if "lessons" in schema["properties"] and "changes" in schema["properties"]:
            self.merge_calls.append(content)
            current = [l for l in content[0].splitlines()[1:] if l.startswith("- ")]
            new = [l for l in content[1].splitlines()[1:] if l.startswith("- ")]
            merged = sorted({l[2:] for l in current + new})
            return {"data": {"lessons": merged, "changes": "merged"}, "model": "fake", "usage": {}}
        self.lesson_calls.append(content)
        text = "\n".join(c for c in content if isinstance(c, str))
        assert "CORRECT ANSWER (from the human labeller): Result B" in text
        assert sum(not isinstance(c, str) for c in content) == 3  # original, A, B
        return {"data": {"what_judge_missed": "B follows the prompt", "label_seems_wrong": self.flag_label,
                         "lessons": ["Check the requested colour in both results."]},
                "model": "fake", "usage": {}}


@pytest.fixture
def images(tmp_path):
    paths = {}
    for name, col in {"orig": (40, 90, 200), "a": (220, 30, 30), "b": (30, 160, 60)}.items():
        p = tmp_path / f"{name}.png"
        p.write_bytes(png_bytes(col))
        paths[name] = str(p)
    return {"originals": [paths["orig"]], "a": paths["a"], "b": paths["b"]}


@pytest.fixture(autouse=True)
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)


def make_set(db, images, labels, name="Train 1"):
    sid = db.create_set(name)
    for label in labels:
        db.add_set_task(sid, "Make the circle green", images, label)
    return sid


def test_train_learns_from_mistakes_and_activates_new_version(db, images):
    sid = make_set(db, images, ["B", "B", "A"])  # judge says A: two mistakes
    judge = LearningJudge()
    out = asyncio.run(learn.train(db, sid, CFG, judge))

    assert out.run.metrics["confident_accuracy"] == pytest.approx(1 / 3)
    assert len(judge.lesson_calls) == 2  # only the two wrong tasks
    assert out.lessons == ["Check the requested colour in both results."]
    active = db.active_knowledge()
    assert active["id"] == out.knowledge_id and active["lessons"] == out.lessons
    run = db.get_benchmark(out.run.run_id, include_items=False)
    assert run["status"] == "done" and run["mode"] == "train" and run["learned_knowledge_id"] == out.knowledge_id
    assert run["metrics"]["training"]["cases_reviewed"] == 2


def test_test_run_uses_lessons_and_baseline_does_not(db, images):
    train_set = make_set(db, images, ["B", "B"])
    test_set = make_set(db, images, ["B", "B", "B"], name="Test 1")
    judge = LearningJudge()
    trained = asyncio.run(learn.train(db, train_set, CFG, judge))
    cfg = CFG.with_knowledge(db.get_knowledge(trained.knowledge_id))
    main, baseline = asyncio.run(learn.run_set(db, test_set, "test", cfg, judge, compare_baseline=True))
    assert main.metrics["confident_accuracy"] == 1.0      # lessons fixed it
    assert baseline.metrics["confident_accuracy"] == 0.0  # untrained still wrong
    assert db.get_benchmark(baseline.run_id, include_items=False)["baseline_of"] == main.run_id


def test_no_mistakes_means_no_new_version(db, images):
    sid = make_set(db, images, ["A", "A"])
    out = asyncio.run(learn.train(db, sid, CFG, LearningJudge()))
    assert out.knowledge_id is None and db.active_knowledge() is None
    assert out.run.metrics["training"]["new_lessons"] == 0


def test_flagged_label_produces_no_lesson(db, images):
    sid = make_set(db, images, ["B"])
    out = asyncio.run(learn.train(db, sid, CFG, LearningJudge(flag_label=True)))
    assert out.knowledge_id is None
    assert out.run.metrics["training"]["flagged_labels"] == ["%05d" % db.get_set(sid)["tasks"][0]["id"]]


def test_training_skips_unlabelled_tasks(db, images):
    sid = make_set(db, images, ["B", None, None])
    judge = LearningJudge()
    out = asyncio.run(learn.train(db, sid, CFG, judge))
    assert out.run.metrics["tasks"] == 1


def test_lessons_accumulate_across_rounds(db, images):
    judge = LearningJudge()
    first = asyncio.run(learn.train(db, make_set(db, images, ["B"]), CFG, judge))

    class Second(LearningJudge):
        async def judge_once(self, task, config, swapped, notes):
            return {"judgment": make_judgment("A", "high"), "model": "fake", "usage": {}}  # still wrong

        async def complete_json(self, system, content, schema, config):
            if "changes" in schema["properties"]:
                return await super().complete_json(system, content, schema, config)
            return {"data": {"what_judge_missed": "x", "label_seems_wrong": False,
                             "lessons": ["Count objects before comparing."]}, "model": "fake", "usage": {}}

    cfg = CFG.with_knowledge(db.get_knowledge(first.knowledge_id))
    second = asyncio.run(learn.train(db, make_set(db, images, ["B"], name="Train 2"), cfg, Second()))
    assert set(second.lessons) == {"Check the requested colour in both results.", "Count objects before comparing."}
    assert db.get_knowledge(second.knowledge_id)["parent_id"] == first.knowledge_id


def test_judge_only_then_label_afterwards_rescores(db, images):
    sid = make_set(db, images, [None, None], name="Judge me")
    main, _ = asyncio.run(learn.run_set(db, sid, "judge", CFG, LearningJudge()))
    assert main.metrics["total"] == 0 and main.metrics["unlabeled"] == 2
    for t in db.get_set(sid)["tasks"]:
        db.set_task_label(t["id"], "A")
    run = learn.refresh_labels(db, db.get_benchmark(main.run_id))
    assert run["metrics"]["total"] == 2 and run["metrics"]["confident_accuracy"] == 1.0


def test_pick_learning_cases_order():
    def item(tid, status, correct, runs=True):
        return {"task_id": tid, "status": status, "correct": correct, "runs": [{}] if runs else []}
    items = [item("1", "confident", True), item("2", "review", True), item("3", "unclear", None),
             item("4", "confident", False), item("5", "review", False), item("6", "error", None),
             item("7", "unclear", None, runs=False)]  # identical A/B: nothing to learn
    assert [i["task_id"] for i in learn.pick_learning_cases(items, 10)] == ["4", "5", "3", "2"]
    assert len(learn.pick_learning_cases(items, 2)) == 2


def test_knowledge_changes_cache_key(db, images):
    from app.images import prepare_image
    from app.judge import Task, cache_key
    t = Task("p", [prepare_image(png_bytes())], prepare_image(png_bytes((1, 2, 3))), prepare_image(png_bytes((9, 9, 9))))
    plain = cache_key(t, CFG, 0)
    with_lesson = cache_key(t, CFG.with_knowledge({"id": 1, "guidelines": "", "lessons": ["x"]}), 0)
    assert plain != with_lesson
    assert cache_key(t, CFG.with_knowledge(None), 0) == plain


def test_system_prompt_includes_knowledge():
    from app.judge import judge_system_prompt
    cfg = CFG.with_knowledge({"id": 1, "guidelines": "Prefer the sharper image on ties.", "lessons": ["Count first."]})
    text = judge_system_prompt(cfg)
    assert "PROJECT GUIDELINES" in text and "Prefer the sharper image" in text and "- Count first." in text
    assert "LESSONS" not in judge_system_prompt(CFG)


def test_luck_chance_matches_sign_test():
    assert learn.luck_chance(0, 0) == 1.0
    assert learn.luck_chance(6, 0) == pytest.approx(1 / 64)
    assert learn.luck_chance(6, 1) == pytest.approx(8 / 128)  # 1 in 16
    assert learn.luck_chance(3, 2) == pytest.approx(0.5)
    assert learn.luck_chance(0, 5) == learn.luck_chance(5, 0)  # harm is judged the same way


def test_compare_with_baseline():
    def it(tid, correct, label="A"):
        return {"task_id": tid, "correct": correct, "label": label}
    now = [it("1", True), it("2", True), it("3", False), it("4", None), it("5", True, label=None),
           it("6", True), it("7", True), it("8", True), it("9", True)]
    before = [it("1", False), it("2", None), it("3", True), it("4", None), it("5", False, label=None),
              it("6", False), it("7", False), it("8", False), it("9", True)]
    cmp = learn.compare_with_baseline(now, before)
    assert cmp["fixed"] == ["1", "2", "6", "7", "8"] and cmp["broke"] == ["3"]  # 5 ignored: no answer
    assert cmp["luck_chance"] == pytest.approx(7 / 64) and cmp["verdict"] == "maybe"
    assert learn.compare_with_baseline([], [])["verdict"] == "no_change"
    six = [it(str(i), True) for i in range(6)]
    assert learn.compare_with_baseline(six, [it(str(i), False) for i in range(6)])["verdict"] == "helped"
    assert learn.compare_with_baseline([it(str(i), False) for i in range(6)], six)["verdict"] == "hurt"
