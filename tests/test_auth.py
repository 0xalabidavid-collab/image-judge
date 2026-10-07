"""Hosted-app login: every page and API call needs a Supabase session; answers record who gave them."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from test_api_train import task_files

USERS = {"good-token": {"email": "ann@example.com"}, "new-token": {"email": "ann@example.com"}}


def fake_supabase():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "anon-key"
        path, query = request.url.path, request.url.query.decode()
        if path == "/auth/v1/token" and "grant_type=password" in query:
            creds = json.loads(request.content)
            if creds["password"] == "right":
                return httpx.Response(200, json={"access_token": "good-token", "refresh_token": "r1",
                                                 "user": {"email": creds["email"]}})
            return httpx.Response(400, json={"error": "invalid_grant"})
        if path == "/auth/v1/token" and "grant_type=refresh_token" in query:
            if json.loads(request.content)["refresh_token"] == "r1":
                return httpx.Response(200, json={"access_token": "new-token", "refresh_token": "r2",
                                                 "user": USERS["new-token"]})
            return httpx.Response(400, json={})
        if path == "/auth/v1/user":
            token = request.headers["authorization"].removeprefix("Bearer ")
            return httpx.Response(200, json=USERS[token]) if token in USERS else httpx.Response(401, json={})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def hosted(db, tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path / "uploads", raising=False)
    monkeypatch.setattr(main.settings, "auth_required", True, raising=False)
    monkeypatch.setattr(main.settings, "supabase_url", "https://x.supabase.co", raising=False)
    monkeypatch.setattr(main.settings, "supabase_anon_key", "anon-key", raising=False)
    monkeypatch.setattr(main.settings, "allowed_emails", frozenset(), raising=False)
    import app.auth as auth_mod
    real = auth_mod.SupabaseAuth
    monkeypatch.setattr(auth_mod, "SupabaseAuth", lambda url, key: real(url, key, client=fake_supabase()))
    from conftest import FakeJudge
    with TestClient(main.create_app(db=db, judge=FakeJudge()), follow_redirects=False) as c:
        yield c


def sign_in(c, email="ann@example.com", password="right"):
    return c.post("/api/login", json={"email": email, "password": password})


def test_pages_and_api_need_a_login(hosted):
    assert hosted.get("/").status_code == 303 and hosted.get("/").headers["location"] == "/login"
    assert hosted.get("/train").status_code == 303
    assert hosted.get("/help").status_code == 303 and hosted.get("/lessons").status_code == 303
    r = hosted.get("/api/sets")
    assert r.status_code == 401 and r.json()["detail"] == "Sign in required."
    assert hosted.get("/api/images/" + "a" * 64 + ".png").status_code == 401
    assert hosted.get("/docs").status_code == 303  # the API explorer is not public either


def test_public_paths_stay_open(hosted):
    assert hosted.get("/healthz").json() == {"ok": True}
    assert hosted.get("/login").status_code == 200
    assert hosted.get("/static/styles.css").status_code == 200
    assert hosted.get("/static/login.js").status_code == 200


def test_wrong_password_is_refused(hosted):
    r = sign_in(hosted, password="wrong")
    assert r.status_code == 401 and "Wrong email or password" in r.json()["detail"]
    assert "ij_at" not in hosted.cookies


def test_login_sets_httponly_cookies_and_unlocks_the_app(hosted):
    r = sign_in(hosted)
    assert r.status_code == 200
    set_cookie = " ".join(r.headers.get_list("set-cookie")).lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie
    assert hosted.get("/api/sets").status_code == 200
    assert hosted.get("/").status_code == 200
    assert hosted.get("/api/me").json() == {"email": "ann@example.com", "owner": True}


def test_logout_locks_it_again(hosted):
    sign_in(hosted)
    assert hosted.post("/api/logout").status_code == 200
    assert hosted.get("/api/sets").status_code == 401


def test_an_expired_session_is_renewed_with_the_refresh_cookie(hosted):
    sign_in(hosted)
    # Supabase rejects this token; the refresh token still works.
    hosted.cookies.set("ij_at", "expired-token", domain="testserver.local", path="/")
    r = hosted.get("/api/sets")
    assert r.status_code == 200
    assert hosted.cookies.get("ij_at") == "new-token" and hosted.cookies.get("ij_rt") == "r2"


def test_allowlist_blocks_other_accounts(hosted, monkeypatch):
    monkeypatch.setattr(main.settings, "allowed_emails", frozenset({"boss@example.com"}), raising=False)
    assert sign_in(hosted).status_code == 401  # a valid Supabase user, but not on the list


def test_cross_site_writes_are_refused(hosted):
    sign_in(hosted)
    r = hosted.post("/api/sets", json={"name": "x"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = hosted.post("/api/sets", json={"name": "x"}, headers={"origin": "http://testserver"})
    assert r.status_code == 200


def test_server_folder_imports_are_off_when_hosted(hosted):
    sign_in(hosted)
    sid = hosted.post("/api/sets", json={"name": "s"}).json()["id"]
    assert hosted.post(f"/api/sets/{sid}/import", json={"path": "C:/Windows"}).status_code == 403
    assert hosted.post("/api/benchmarks", json={"dataset": "C:/Windows"}).status_code == 403


def test_work_is_attributed_to_the_signed_in_person(hosted, db):
    sign_in(hosted)
    res = hosted.post("/api/sets/tasks", data={"prompt": "Make the circle green", "label": "B",
                                               "new_set_name": "mine"}, files=task_files(0))
    assert res.status_code == 200
    row = db._one("SELECT created_by, labelled_by FROM set_tasks")
    assert (row["created_by"], row["labelled_by"]) == ("ann@example.com", "ann@example.com")
    hosted.patch(f"/api/tasks/{res.json()['id']}", json={"label": "A"})
    assert db._one("SELECT labelled_by FROM set_tasks")["labelled_by"] == "ann@example.com"


def test_login_needs_supabase_settings(db, monkeypatch):
    monkeypatch.setattr(main.settings, "auth_required", True, raising=False)
    monkeypatch.setattr(main.settings, "supabase_url", "", raising=False)
    monkeypatch.setattr(main.settings, "supabase_anon_key", "", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        main.create_app(db=db)


# --- owner-only (labellers cannot change what applies to everyone) -----------------
@pytest.fixture
def labeller(hosted, monkeypatch):
    monkeypatch.setattr(main.settings, "admin_emails", frozenset({"boss@example.com"}), raising=False)
    sign_in(hosted)  # ann@example.com is signed in but is not an owner
    return hosted


def test_labellers_cannot_open_owner_pages(labeller):
    for page in ("/train", "/benchmark"):
        r = labeller.get(page)
        assert r.status_code == 303 and r.headers["location"] == "/"
    assert labeller.get("/").status_code == 200 and labeller.get("/lessons").status_code == 200
    assert labeller.get("/help").status_code == 200


def test_labellers_cannot_change_lessons_or_start_runs(labeller):
    assert labeller.post("/api/runs", json={"mode": "train", "set_id": 1}).status_code == 403
    assert labeller.post("/api/benchmarks", json={"dataset": "x"}).status_code == 403
    assert labeller.post("/api/knowledge/undo").status_code == 403
    assert labeller.post("/api/knowledge/gate").status_code == 403
    assert labeller.post("/api/knowledge/activate", json={"id": None}).status_code == 403
    assert labeller.post("/api/knowledge/1/reject").status_code == 403
    assert labeller.delete("/api/sets/1").status_code == 403
    assert labeller.delete("/api/tasks/1").status_code == 403
    assert labeller.post("/api/knowledge/gate").json()["detail"] == "Only the project owner can do this."


def test_labellers_can_do_their_own_work(labeller, db):
    assert labeller.get("/api/knowledge").status_code == 200     # read the lessons
    assert labeller.get("/api/answers/missing-reasons").status_code == 200
    res = labeller.post("/api/sets/tasks", data={"prompt": "Make the circle green", "label": "B",
                                                 "new_set_name": "My tasks"}, files=task_files(0))
    assert res.status_code == 200                                  # save a labelled task
    assert labeller.patch(f"/api/tasks/{res.json()['id']}", json={"label": "A"}).status_code == 200
    assert labeller.get("/api/me").json() == {"email": "ann@example.com", "owner": False}


def test_owners_can_do_everything(hosted, monkeypatch):
    monkeypatch.setattr(main.settings, "admin_emails", frozenset({"ann@example.com"}), raising=False)
    sign_in(hosted)
    assert hosted.get("/train").status_code == 200
    assert hosted.post("/api/knowledge/undo").status_code == 400   # allowed through (nothing to undo)
    assert hosted.get("/api/me").json()["owner"] is True


def test_with_no_owner_list_everyone_is_an_owner(hosted):
    sign_in(hosted)
    assert hosted.get("/train").status_code == 200
    assert hosted.get("/api/me").json()["owner"] is True


def test_background_judgments_are_attributed_and_need_a_login(hosted, db):
    import time
    assert hosted.post("/api/evaluate/start", data={"prompt": "x"}, files=task_files(0)).status_code == 401
    sign_in(hosted)
    job = hosted.post("/api/evaluate/start", data={"prompt": "Make the circle green"}, files=task_files(0)).json()["job"]
    end = time.time() + 20
    while time.time() < end and hosted.get(f"/api/evaluate/jobs/{job}").json()["status"] == "running":
        time.sleep(0.05)
    assert hosted.get(f"/api/evaluate/jobs/{job}").json()["status"] == "done"
    assert db._one("SELECT created_by FROM evaluations")["created_by"] == "ann@example.com"
