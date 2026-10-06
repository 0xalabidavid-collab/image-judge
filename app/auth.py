"""Login for the hosted app, using Supabase Auth (email + password).

The browser never holds a token in JavaScript. /api/login exchanges the password for a session at
Supabase and stores the tokens in HttpOnly cookies, so pages, API calls and <img> tags all carry it.
Invite people from the Supabase dashboard (Authentication > Users) and turn off public sign-ups there.
"""

from __future__ import annotations

import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from .config import Settings
from .identity import set_user

ACCESS_COOKIE = "ij_at"
REFRESH_COOKIE = "ij_rt"
PUBLIC_PATHS = {"/healthz", "/login", "/api/login"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
OWNER_PAGES = {"/train", "/benchmark"}
# Changing these affects everyone, so labellers (non-owners) may only read them.
OWNER_WRITE_PREFIXES = ("/api/runs", "/api/benchmarks", "/api/knowledge")


def owner_only(method: str, path: str) -> bool:
    """True if this request is something only an owner may do."""
    if method in ("GET", "HEAD") and path in OWNER_PAGES:
        return True
    if method not in SAFE_METHODS and path.startswith(OWNER_WRITE_PREFIXES):
        return True
    # Deleting or renaming whole task sets / single tasks (labelling a task is fine).
    if method == "DELETE" and path.startswith(("/api/sets", "/api/tasks")):
        return True
    if method == "PATCH" and path.startswith("/api/sets"):
        return True
    return method == "POST" and path.startswith("/api/sets/") and path.endswith("/import")
USER_CACHE_SECONDS = 60


class LoginBody(BaseModel):
    email: str
    password: str


class SupabaseAuth:
    def __init__(self, url: str, anon_key: str, client: Optional[httpx.AsyncClient] = None):
        self.url = url.rstrip("/")
        self.anon_key = anon_key
        self._client = client
        self._users: dict[str, tuple[float, dict]] = {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20)
        return self._client

    def _headers(self, token: Optional[str] = None) -> dict:
        return {"apikey": self.anon_key, "Authorization": f"Bearer {token or self.anon_key}"}

    async def sign_in(self, email: str, password: str) -> Optional[dict]:
        r = await self._http().post(f"{self.url}/auth/v1/token?grant_type=password", headers=self._headers(),
                                    json={"email": email, "password": password})
        if r.status_code in (400, 401, 422):
            return None  # wrong email or password
        if r.status_code != 200:
            raise HTTPException(502, "The login service is not available. Try again in a minute.")
        return r.json()

    async def refresh(self, refresh_token: str) -> Optional[dict]:
        r = await self._http().post(f"{self.url}/auth/v1/token?grant_type=refresh_token", headers=self._headers(),
                                    json={"refresh_token": refresh_token})
        return r.json() if r.status_code == 200 else None

    async def user_for(self, access_token: str) -> Optional[dict]:
        cached = self._users.get(access_token)
        if cached and cached[0] > time.time():
            return cached[1]
        r = await self._http().get(f"{self.url}/auth/v1/user", headers=self._headers(access_token))
        if r.status_code != 200:
            self._users.pop(access_token, None)
            return None
        user = r.json()
        if len(self._users) > 500:
            self._users.clear()
        self._users[access_token] = (time.time() + USER_CACHE_SECONDS, user)
        return user

    def forget(self, access_token: Optional[str]) -> None:
        if access_token:
            self._users.pop(access_token, None)


def _set_session_cookies(response, session: dict, secure: bool) -> None:
    opts = dict(httponly=True, samesite="lax", secure=secure, path="/")
    response.set_cookie(ACCESS_COOKIE, session["access_token"], max_age=60 * 60 * 24, **opts)
    response.set_cookie(REFRESH_COOKIE, session["refresh_token"], max_age=60 * 60 * 24 * 30, **opts)


def _clear_session_cookies(response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/")


def _same_origin(request: Request) -> bool:
    """Reject state-changing requests that a different website triggered in the user's browser."""
    origin = request.headers.get("origin")
    if not origin:
        return True  # not a browser cross-site request
    return urlparse(origin).netloc == request.headers.get("host", "")


def install_auth(app: FastAPI, cfg: Settings, auth: Optional[SupabaseAuth] = None) -> None:
    """Protect the whole app. Call once, after the app is created and before it serves requests."""
    if not (cfg.supabase_url and cfg.supabase_anon_key):
        raise RuntimeError("Login is on (IMAGE_JUDGE_AUTH) but SUPABASE_URL and SUPABASE_ANON_KEY are not both set.")
    auth = auth or SupabaseAuth(cfg.supabase_url, cfg.supabase_anon_key)
    app.state.auth = auth
    login_page = __import__("pathlib").Path(__file__).parent / "static" / "login.html"

    def allowed(user: dict) -> bool:
        email = (user.get("email") or "").lower()
        return bool(email) and (not cfg.allowed_emails or email in cfg.allowed_emails)

    @app.get("/login", include_in_schema=False)
    def login_get():
        return FileResponse(login_page)

    @app.post("/api/login")
    async def login(body: LoginBody, request: Request):
        session = await auth.sign_in(body.email.strip(), body.password)
        if not session or not allowed(session.get("user") or {}):
            raise HTTPException(401, "Wrong email or password.")
        response = JSONResponse({"email": session["user"]["email"]})
        _set_session_cookies(response, session, request.url.scheme == "https")
        return response

    @app.post("/api/logout")
    async def logout(request: Request):
        auth.forget(request.cookies.get(ACCESS_COOKIE))
        response = JSONResponse({"ok": True})
        _clear_session_cookies(response)
        return response

    def is_owner(email: str) -> bool:
        return not cfg.admin_emails or email.lower() in cfg.admin_emails

    @app.get("/api/me")
    async def me(request: Request):
        from .identity import who
        return {"email": who(), "owner": is_owner(who())}

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)
        if request.method not in SAFE_METHODS and not _same_origin(request):
            return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)

        user, new_session = None, None
        token = request.cookies.get(ACCESS_COOKIE)
        if token:
            user = await auth.user_for(token)
        if not user and request.cookies.get(REFRESH_COOKIE):
            new_session = await auth.refresh(request.cookies[REFRESH_COOKIE])
            if new_session:
                user = new_session.get("user") or await auth.user_for(new_session["access_token"])
        if not user or not allowed(user):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Sign in required."}, status_code=401)
            return RedirectResponse("/login", status_code=303)

        set_user(user["email"].lower())
        if owner_only(request.method, path) and not is_owner(user["email"]):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Only the project owner can do this."}, status_code=403)
            return RedirectResponse("/", status_code=303)
        response = await call_next(request)
        if new_session:
            _set_session_cookies(response, new_session, request.url.scheme == "https")
        return response
