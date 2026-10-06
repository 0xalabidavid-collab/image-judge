"""Who is making the current request (set by the login middleware; '' when login is off)."""

from contextvars import ContextVar

_current_user: ContextVar[str] = ContextVar("current_user", default="")


def set_user(email: str):
    return _current_user.set(email)


def who() -> str:
    return _current_user.get()
