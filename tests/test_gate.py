"""Gated auto-training: corrections make candidate lessons that only go live after a held-out test."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app import autotrain, main
from app import benchmark as bm
from app.autotrain import MY_TASKS_SET, decide, gate_candidate
from conftest import png_bytes
from test_api_train import add_task, task_files, wait_learning
from test_learn import LearningJudge


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    monkeypatch.setattr(main.settings, "auto_gate", True, raising=False)
    monkeypatch.setattr(main.settings, "gate_every", 1, raising=False)
    monkeypatch.setattr(main.settings, "gate_min_pool", 3, raising=False)
    monkeypatch.setattr(main.settings, "gate_max_tasks", 20, raising=False)
    monkeypatch.setattr(main.settings, "gate_margin", 0.05, raising=False)
    with TestClient(main.create_app(db=db, judge=LearningJudge())) as c:
        yield c


def my_tasks(client, labels):
    """File tasks (with the given right answers) into the 'My tasks' pool; returns their ids."""
    ids = []
    for i, label in enumerate(labels):
        out = add_task(client, label, set_name=MY_TASKS_SET, i=i + 10)
        ids.append(out["id"])
    return ids


def wait_gate(client, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get("/api/knowledge").json()["gate"]["state"]
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError("the gate test did not finish")


def run_gate(db, cid):
    return asyncio.run(gate_candidate(db, cid, main.settings.judge, LearningJudge()))


def candidate(db, learned_from=(), lessons=("Check the requested colour in both results.",)):
    return db.add_knowledge("", list(lessons), "test candidate", activate=False, status="candidate",
                            learned_from=list(learned_from))


# --- the decision rule -------------------------------------------------------
def test_decision_rule():
    base = {"tasks": 20, "right": 10, "confident_wrong": 2}
    assert decide(base, {"tasks": 20, "right": 12, "confident_wrong": 2}, 0.05)[0] == "accepted"
    assert decide(base, {"tasks": 20, "right": 10, "confident_wrong": 1}, 0.05)[0] == "rejected"  # no gain
    assert decide(base, {"tasks": 20, "right": 9, "confident_wrong": 0}, 0.05)[0] == "rejected"   # worse
    verdict, why = decide(base, {"tasks": 20, "right": 16, "confident_wrong": 3}, 0.05)
    assert verdict == "rejected" and "confident mistakes" in why  # more right, but more sure-and-wrong
    small = {"tasks": 4, "right": 1, "confident_wrong": 0}
    assert decide(small, {"tasks": 4, "right": 2, "confident_wrong": 0}, 0.05)[0] == "accepted"  # needs at least 1


def test_pool_excludes_tasks_the_lessons_were_learned_from():
    class T:
        def __init__(self, i):
            self.set_task_id = i
    pool = autotrain.pick_pool([T(i) for i in range(1, 9)], {2, 4, 6}, 20)
    assert sorted(t.set_task_id for t in pool) == [1, 3, 5, 7, 8]
    assert [t.set_task_id for t in autotrain.pick_pool([T(i) for i in range(1, 9)], set(), 3)] == \
        [t.set_task_id for t in autotrain.pick_pool([T(i) for i in range(1, 9)], set(), 3)]  # stable order
    assert len(autotrain.pick_pool([T(i) for i in range(1, 9)], set(), 3)) == 3


# --- the gate ----------------------------------------------------------------
def test_a_better_candidate_is_accepted_and_switched_on(client, db):
    my_tasks(client, ["B"] * 4)  # the fake judge says A without lessons and B with them
    cid = candidate(db)
    result = run_gate(db, cid)
    assert result["decision"] == "accepted" and result["candidate"]["right"] == 4 and result["baseline"]["right"] == 0
    assert db.get_knowledge(cid)["status"] == "accepted" and db.active_knowledge()["id"] == cid
    assert db.get_knowledge(cid)["gate_result"]["decision"] == "accepted"


def test_a_worse_candidate_is_rejected_and_the_current_lessons_stay(client, db):
    my_tasks(client, ["A"] * 4)  # without lessons it is right; with them it flips to B and is wrong
    current = db.add_knowledge("", ["old lesson"], "current", activate=True)
    cid = candidate(db, lessons=("old lesson", "bad new lesson"))
    result = run_gate(db, cid)
    assert result["decision"] == "rejected"
    assert db.get_knowledge(cid)["status"] == "rejected" and db.active_knowledge()["id"] == current


def test_it_waits_when_there_are_too_few_unseen_tasks(client, db):
    ids = my_tasks(client, ["B"] * 4)
    cid = candidate(db, learned_from=ids[:2])  # only 2 tasks left that it was not learned from; needs 3
    result = run_gate(db, cid)
    assert result["decision"] == "waiting" and result["pool"] == 2
    assert db.get_knowledge(cid)["status"] == "candidate" and db.active_knowledge() is None


def test_tasks_it_learned_from_are_never_tested_on(client, db):
    ids = my_tasks(client, ["B"] * 5)
    cid = candidate(db, learned_from=ids[:2])
    result = run_gate(db, cid)
    assert result["pool"] == 3 and result["decision"] == "accepted"


# --- through the app ---------------------------------------------------------
def evaluate(client, i=0):
    res = client.post("/api/evaluate", data={"prompt": "Make the circle green"}, files=task_files(i))
    assert res.status_code == 200, res.text
    return res.json()


def test_a_correction_makes_a_candidate_that_is_not_used_yet(client, db):
    first = evaluate(client)
    client.post(f"/api/evaluations/{first['id']}/feedback",
                json={"verdict_correct": False, "reason": "B is green as asked; A stayed red."})
    learned = wait_learning(client, first["id"])
    assert learned["status"] == "done" and learned["result"]["status"] == "candidate"
    assert client.get("/api/config").json()["active_knowledge"] is None  # still judging with no new lessons
    item = client.get("/api/knowledge").json()["items"][0]
    assert item["status"] == "candidate" and not item["active"]
    assert item["added"] == ["Check the requested colour in both results."] and item["new_tasks"] == 1
    again = evaluate(client, 1)  # the next judgment does not use the candidate
    assert again["aggregate"]["verdict"] == "A"


def test_enough_corrections_trigger_the_test_and_a_pass_switches_it_on(client, db):
    my_tasks(client, ["B"] * 3)  # unseen tasks to test on
    first = evaluate(client, 1)
    client.post(f"/api/evaluations/{first['id']}/feedback", json={"verdict_correct": False, "reason": "B is green."})
    wait_learning(client, first["id"])
    state = wait_gate(client)
    assert state["status"] == "done" and state["result"]["decision"] == "accepted", state
    assert client.get("/api/config").json()["active_knowledge"]["id"] == state["knowledge_id"]
    assert evaluate(client, 2)["aggregate"]["verdict"] == "B"  # now it is in use


def test_two_corrections_build_on_one_candidate(client, db):
    ids = []
    for i in (0, 1):
        ev = evaluate(client, i)
        client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": False, "reason": "B is green.",
                                                                  "learn": True})
        wait_learning(client, ev["id"])
        ids.append(ev["id"])
        if i == 0:
            main.settings.gate_every = 5  # keep the first from being tested before the second arrives
    items = client.get("/api/knowledge").json()["items"]
    statuses = [k["status"] for k in items]
    assert statuses.count("candidate") == 1 and statuses.count("superseded") == 1
    latest = next(k for k in items if k["status"] == "candidate")
    assert latest["new_tasks"] == 1 and len(latest["learned_from"]) == 2


def test_test_now_needs_a_waiting_candidate(client):
    assert client.post("/api/knowledge/gate").status_code == 400


def test_reject_a_candidate_but_not_the_version_in_use(client, db):
    cid = candidate(db)
    assert client.post(f"/api/knowledge/{cid}/reject").status_code == 200
    assert db.get_knowledge(cid)["status"] == "rejected"
    live = db.add_knowledge("", ["x"], "live", activate=True)
    assert client.post(f"/api/knowledge/{live}/reject").status_code == 400
    assert client.post("/api/knowledge/9999/reject").status_code == 404


def test_undo_goes_back_past_versions_that_never_passed(client, db):
    good = db.add_knowledge("", ["a"], "good", activate=True)
    bad = db.add_knowledge("", ["a", "b"], "bad", parent_id=good, activate=False, status="candidate")
    db.set_knowledge_status(bad, "rejected")
    now = db.add_knowledge("", ["a", "c"], "now", parent_id=bad, activate=True)
    assert client.post("/api/knowledge/undo").json() == {"active": good}
    assert db.active_knowledge()["id"] == good and db.active_knowledge()["id"] != now
    assert client.post("/api/knowledge/undo").status_code == 400  # nothing earlier


# --- reasons -----------------------------------------------------------------
def test_answers_without_a_reason_are_listed_and_can_be_explained(client, db):
    ev = evaluate(client)
    client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": False, "learn": False})
    listing = client.get("/api/answers/missing-reasons").json()
    assert [i["id"] for i in listing["items"]] == [ev["id"]]
    assert listing["coverage"] == {"answers": 1, "with_reason": 0}

    assert client.post(f"/api/evaluations/{ev['id']}/reason", json={"reason": "short"}).status_code == 400
    ok = client.post(f"/api/evaluations/{ev['id']}/reason",
                     json={"reason": "B is green as the prompt asks; A stayed red.", "learn": True})
    assert ok.status_code == 200 and ok.json()["learning"] == "started"
    wait_learning(client, ev["id"])
    assert client.get("/api/answers/missing-reasons").json()["items"] == []
    assert client.get("/api/knowledge").json()["reasons"] == {"answers": 1, "with_reason": 1}


def test_a_reason_needs_a_saved_answer(client):
    ev = evaluate(client)
    assert client.post(f"/api/evaluations/{ev['id']}/reason",
                       json={"reason": "B is green as the prompt asks."}).status_code == 404


def test_right_and_sure_answers_do_not_need_a_reason(client, db):
    ev = evaluate(client)  # the fake judge is confident
    client.post(f"/api/evaluations/{ev['id']}/feedback", json={"verdict_correct": True})
    assert client.get("/api/answers/missing-reasons").json()["items"] == []
