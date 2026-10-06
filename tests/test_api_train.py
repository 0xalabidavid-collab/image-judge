"""HTTP API for task sets, knowledge versions and train/test/judge runs."""

import time

import pytest
from fastapi.testclient import TestClient

from app import benchmark as bm
from app import main
from conftest import png_bytes
from test_learn import LearningJudge


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    # These tests cover the plain learn-and-activate path; the gated path is in test_gate.py.
    monkeypatch.setattr(main.settings, "auto_gate", False, raising=False)
    with TestClient(main.create_app(db=db, judge=LearningJudge())) as c:  # context manager runs startup
        yield c


def task_files(i=0):
    return [("originals", ("o.png", png_bytes((40, 90, 200 - i)), "image/png")),
            ("result_a", ("a.png", png_bytes((220, 30, 30 + i)), "image/png")),
            ("result_b", ("b.png", png_bytes((30, 160, 60 + i)), "image/png"))]


def add_task(client, label="B", set_name=None, set_id=None, i=0):
    data = {"prompt": "Make the circle green", "label": label}
    if set_id:
        data["set_id"] = str(set_id)
    else:
        data["new_set_name"] = set_name
    res = client.post("/api/sets/tasks", data=data, files=task_files(i))
    assert res.status_code == 200, res.text
    return res.json()


def wait_for(client, run_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/benchmarks/{run_id}").json()
        if run["status"] != "running" and (not run.get("baseline") or run["baseline"]["status"] != "running"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish: {run}")


def test_save_tasks_creates_set_and_counts(client):
    first = add_task(client, "B", set_name="Train 1")
    second = add_task(client, "", set_id=first["set"]["id"], i=1)
    assert second["set"] == {**second["set"], "name": "Train 1", "tasks": 2, "labeled": 1}
    s = client.get(f"/api/sets/{first['set']['id']}").json()
    assert [t["label"] for t in s["tasks"]] == ["B", None]
    img = s["tasks"][0]["images"]["a"].replace("\\", "/").split("/")[-1]
    assert client.get(f"/api/images/{img}").status_code == 200
    assert client.get("/api/images/..%2F..%2Fsecret.txt").status_code == 404


def test_train_then_test_flow(client):
    sid = add_task(client, "B", set_name="Train 1")["set"]["id"]
    add_task(client, "B", set_id=sid, i=1)
    res = client.post("/api/runs", json={"mode": "train", "set_id": sid, "runs": 2})
    assert res.status_code == 200, res.text
    run = wait_for(client, res.json()["id"])
    assert run["status"] == "done" and run["mode"] == "train"
    assert run["metrics"]["training"]["new_lessons"] == 2
    assert run["learned_knowledge"]["lessons"] == ["Check the requested colour in both results."]
    assert client.get("/api/config").json()["active_knowledge"]["lessons"] == 1

    tid = add_task(client, "B", set_name="Test 1", i=5)["set"]["id"]
    res = client.post("/api/runs", json={"mode": "test", "set_id": tid, "runs": 2, "compare_baseline": True})
    run = wait_for(client, res.json()["id"])
    assert run["metrics"]["confident_accuracy"] == 1.0
    assert run["baseline"]["metrics"]["confident_accuracy"] == 0.0
    assert run["comparison"]["fixed"] and run["comparison"]["broke"] == []


def test_judge_only_then_mark_answers(client):
    sid = add_task(client, "", set_name="Batch")["set"]["id"]
    run = wait_for(client, client.post("/api/runs", json={"mode": "judge", "set_id": sid, "runs": 2}).json()["id"])
    assert run["metrics"]["total"] == 0
    task_id = run["items"][0]["set_task_id"]
    assert client.patch(f"/api/tasks/{task_id}", json={"label": "A"}).status_code == 200
    run = client.get(f"/api/benchmarks/{run['id']}").json()
    assert run["metrics"]["total"] == 1 and run["items"][0]["correct"] is True


def test_train_needs_answers(client):
    sid = add_task(client, "", set_name="No answers")["set"]["id"]
    res = client.post("/api/runs", json={"mode": "train", "set_id": sid})
    assert res.status_code == 400 and "correct answer" in res.json()["detail"]


def test_knowledge_edit_and_activate(client):
    k1 = client.post("/api/knowledge", json={"guidelines": "Prefer exact text.", "lessons": ["One", " ", "Two"]}).json()["id"]
    items = client.get("/api/knowledge").json()["items"]
    assert items[0]["id"] == k1 and items[0]["lessons"] == ["One", "Two"] and items[0]["active"]
    assert client.post("/api/knowledge/activate", json={"id": None}).status_code == 200
    assert client.get("/api/config").json()["active_knowledge"] is None
    client.post("/api/knowledge/activate", json={"id": k1})
    assert client.get("/api/config").json()["active_knowledge"]["id"] == k1


def test_failed_run_records_error(db, tmp_path, monkeypatch):
    from app.judge import JudgeError
    from conftest import FakeJudge
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    with TestClient(main.create_app(db=db, judge=FakeJudge(error=JudgeError("usage limit reached", fatal=True)))) as c:
        sid = add_task(c, "B", set_name="T")["set"]["id"]
        run = wait_for(c, c.post("/api/runs", json={"mode": "test", "set_id": sid}).json()["id"])
    assert run["status"] == "failed" and "usage limit" in run["metrics"]["error"]


def test_import_into_set(client, tmp_path):
    from test_benchmark import row, write_csv_dataset
    write_csv_dataset(tmp_path / "ds", [row(1, "A"), row(2, "B")])
    sid = client.post("/api/sets", json={"name": "Imported"}).json()["id"]
    assert client.post(f"/api/sets/{sid}/import", json={"path": str(tmp_path / "ds")}).json() == {"imported": 2}
    assert [t["label"] for t in client.get(f"/api/sets/{sid}").json()["tasks"]] == ["A", "B"]


def test_duplicate_set_name_rejected(client):
    client.post("/api/sets", json={"name": "X"})
    assert client.post("/api/sets", json={"name": "X"}).status_code == 400


def wait_learning(client, eval_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/api/evaluations/{eval_id}/learning").json()
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("learning did not finish")


def test_two_quick_corrections_both_kept(client):
    """Learning runs one at a time, so a second correction builds on the first instead of overwriting it."""
    judge = client.app.state.judge
    calls = iter(["Rule one.", "Rule two."])
    original = judge.complete_json

    async def complete_json(system, content, schema, config):
        if "changes" in schema["properties"]:
            return await original(system, content, schema, config)
        return {"data": {"what_judge_missed": "x", "label_seems_wrong": False, "lessons": [next(calls)]},
                "model": "fake", "usage": {}}
    judge.complete_json = complete_json
    ids = [client.post("/api/evaluate", data={"prompt": "p"}, files=task_files(i)).json()["id"] for i in (1, 2)]
    for i in ids:
        client.post(f"/api/evaluations/{i}/feedback", json={"verdict_correct": False, "reason": "r"})
    for i in ids:
        assert wait_learning(client, i)["status"] == "done"
    lessons = client.get("/api/knowledge").json()["items"][0]["lessons"]
    assert set(lessons) == {"Rule one.", "Rule two."}


def test_correcting_a_verdict_learns_immediately(client):
    """Evaluate -> mark Wrong with a reason -> lesson learned -> next evaluation uses it."""
    files = task_files()
    res = client.post("/api/evaluate", data={"prompt": "Make the circle green"}, files=files)
    first = res.json()
    assert first["aggregate"]["verdict"] == "A"  # the fake judge says A until it has lessons

    fb = client.post(f"/api/evaluations/{first['id']}/feedback",
                     json={"verdict_correct": False, "reason": "B is green as asked; A stayed red."}).json()
    assert fb["true_label"] == "B" and fb["learning"] == "started"  # learns in the background
    learned = wait_learning(client, first["id"])
    assert learned["status"] == "done", learned
    assert learned["result"]["new_lessons"] == ["Check the requested colour in both results."]
    assert client.get("/api/config").json()["active_knowledge"]["id"] == learned["result"]["knowledge_id"]
    judge = client.app.state.judge
    assert "B is green as asked" in "\n".join(c for c in judge.lesson_calls[-1] if isinstance(c, str))

    # Filed under "My tasks" with the right answer.
    my = next(s for s in client.get("/api/sets").json()["items"] if s["name"] == "My tasks")
    assert my["tasks"] == 1 and my["labeled"] == 1

    second = client.post("/api/evaluate", data={"prompt": "Make the circle green"}, files=task_files(3)).json()
    assert second["aggregate"]["verdict"] == "B"  # uses the new lesson


def test_correct_and_confident_does_not_learn(client):
    client.app.state.judge.answer = "B"
    ev = client.post("/api/evaluate", data={"prompt": "Make the circle green"}, files=task_files()).json()
    fb = client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": True}).json()
    assert fb["learning"] is None and client.get("/api/config").json()["active_knowledge"] is None


def test_changing_feedback_updates_same_task(client):
    ev = client.post("/api/evaluate", data={"prompt": "p"}, files=task_files()).json()
    client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": False, "learn": False})
    client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": True, "learn": False})
    sid = next(s["id"] for s in client.get("/api/sets").json()["items"] if s["name"] == "My tasks")
    tasks = client.get(f"/api/sets/{sid}").json()["tasks"]
    assert len(tasks) == 1 and tasks[0]["label"] == "A"
