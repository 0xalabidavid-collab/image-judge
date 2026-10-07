import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.images import MAX_LONG_EDGE, MAX_PIXELS, prepare_image
from conftest import FakeJudge, png_bytes


@pytest.fixture
def client(db, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    judge = FakeJudge(lambda t, swapped: ("B", "high"))
    return TestClient(main.create_app(db=db, judge=judge))


def files(n_originals=1):
    out = [("originals", (f"o{i}.png", png_bytes((40, 90, 200 - i)), "image/png")) for i in range(n_originals)]
    out += [("result_a", ("a.png", png_bytes((220, 30, 30)), "image/png")),
            ("result_b", ("b.png", png_bytes((30, 160, 60)), "image/png"))]
    return out


def test_evaluate_and_feedback_flow(client):
    res = client.post("/api/evaluate", data={"prompt": "Make it green"}, files=files())
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["aggregate"]["status"] == "confident"
    assert body["aggregate"]["verdict"] == "B"
    assert len(body["runs"]) == main.settings.judge.runs

    fb = client.post(f"/api/evaluations/{body['id']}/feedback", json={"verdict_correct": False})
    body_fb = fb.json()
    assert body_fb["ok"] and body_fb["verdict_correct"] is False and body_fb["true_label"] == "A"

    ev = client.get(f"/api/evaluations/{body['id']}").json()
    assert ev["feedback"] == {"verdict_correct": False, "true_label": "A"}
    stats = client.get("/api/evaluations").json()["feedback_stats"]
    assert stats["confident"] == {"labeled": 1, "correct": 0, "with_verdict": 1}


def test_feedback_on_unclear_needs_true_label(client, db):
    eval_id = db.add_evaluation("p", {"originals": [], "a": "", "b": ""}, {},
                                {"aggregate": {"status": "unclear", "verdict": None}, "runs": []})
    assert client.post(f"/api/evaluations/{eval_id}/feedback", json={"verdict_correct": True}).status_code == 400
    res = client.post(f"/api/evaluations/{eval_id}/feedback", json={"true_label": "B"})
    assert res.json()["true_label"] == "B" and res.json()["verdict_correct"] is None


def test_rejects_too_many_originals(client):
    res = client.post("/api/evaluate", data={"prompt": "x"}, files=files(11))
    assert res.status_code == 400


def test_accepts_ten_originals(client):
    res = client.post("/api/evaluate", data={"prompt": "x"}, files=files(10))
    assert res.status_code == 200, res.text


def test_rejects_non_image(client):
    bad = files()
    bad[-1] = ("result_b", ("b.txt", b"hello", "text/plain"))
    res = client.post("/api/evaluate", data={"prompt": "x"}, files=bad)
    assert res.status_code == 400
    assert "image" in res.json()["detail"]


def test_api_error_surfaces_as_502(db, tmp_path, monkeypatch):
    from app.judge import JudgeError
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    c = TestClient(main.create_app(db=db, judge=FakeJudge(error=JudgeError("API key missing", fatal=True))))
    res = c.post("/api/evaluate", data={"prompt": "x"}, files=files())
    assert res.status_code == 502
    assert "API key missing" in res.json()["detail"]


def test_pages_served(client):
    assert "Image Judge" in client.get("/").text
    assert "benchmark" in client.get("/benchmark").text.lower()


def test_resize_limits_and_alpha():
    big = Image.new("RGBA", (4000, 3000), (255, 0, 0, 128))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    p = prepare_image(buf.getvalue())
    assert max(p.width, p.height) <= MAX_LONG_EDGE
    assert p.width * p.height <= MAX_PIXELS * 1.01
    assert p.media_type == "image/png"  # transparency preserved


def test_benchmark_blank_fields_use_defaults(client, tmp_path, monkeypatch):
    from app import benchmark as bm
    from test_benchmark import row, write_csv_dataset
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    write_csv_dataset(tmp_path / "ds", [row(1, "B")])
    res = client.post("/api/benchmarks", json={"dataset": str(tmp_path / "ds"), "rubric_version": "", "model": ""})
    assert res.status_code == 200, res.text
    run = client.get(f"/api/benchmarks/{res.json()['id']}").json()
    assert run["config"]["rubric_version"] == main.settings.judge.rubric_version
    assert run["config"]["model"] == main.settings.judge.model


def test_benchmark_rejects_bad_dataset(client, tmp_path):
    res = client.post("/api/benchmarks", json={"dataset": str(tmp_path / "nope")})
    assert res.status_code == 400


def test_every_page_is_served_and_linked_from_the_menu(client):
    pages = ["/", "/train", "/lessons", "/benchmark", "/help"]
    for path in pages:
        res = client.get(path)
        assert res.status_code == 200 and "text/html" in res.headers["content-type"], path
        for other in pages[1:]:
            assert f'href="{other}"' in res.text, f"{path} has no menu link to {other}"


def test_config_says_where_data_is_saved(client):
    storage = client.get("/api/config").json()["storage"]
    assert storage["database_online"] is False and "not saved online" in storage["database"]
    assert storage["images_online"] is False and "not saved online" in storage["images"]
    assert "://" not in str(storage) and "key" not in str(storage).lower()  # words only: no addresses or keys


@pytest.fixture
def live_client(db, tmp_path, monkeypatch):
    """A client whose event loop stays open between requests, so background jobs can finish."""
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    judge = FakeJudge(lambda t, swapped: ("B", "high"))
    with TestClient(main.create_app(db=db, judge=judge)) as c:
        yield c


def wait_job(client, job, timeout=20):
    import time
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/api/evaluate/jobs/{job}").json()
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("the background judgment did not finish")


def test_background_evaluation_starts_then_finishes(live_client):
    client = live_client
    res = client.post("/api/evaluate/start", data={"prompt": "Make it green"}, files=files())
    assert res.status_code == 200, res.text
    st = wait_job(client, res.json()["job"])
    assert st["status"] == "done"
    body = st["result"]
    assert body["aggregate"]["verdict"] == "B" and len(body["runs"]) == main.settings.judge.runs
    assert client.get(f"/api/evaluations/{body['id']}").status_code == 200  # saved like a normal evaluation


def test_background_evaluation_reports_a_failure_in_words(db, tmp_path, monkeypatch):
    from app.judge import JudgeError
    from conftest import FakeJudge
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    app = main.create_app(db=db, judge=FakeJudge(error=JudgeError("usage limit reached", fatal=True)))
    with TestClient(app) as c:
        job = c.post("/api/evaluate/start", data={"prompt": "Make it green"}, files=files()).json()["job"]
        st = wait_job(c, job)
    assert st["status"] == "failed" and "usage limit" in st["error"]


def test_background_evaluation_checks_its_input_and_unknown_jobs(client):
    assert client.post("/api/evaluate/start", data={"prompt": "  "}, files=files()).status_code == 400
    assert client.get("/api/evaluate/jobs/doesnotexist").status_code == 404
