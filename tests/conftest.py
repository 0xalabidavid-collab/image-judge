import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def png_bytes(color=(40, 90, 200), size=(200, 150), shape="circle", text=None) -> bytes:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    box = (50, 30, 150, 120)
    if shape == "circle":
        d.ellipse(box, fill=color)
    else:
        d.rectangle(box, fill=color)
    if text:
        d.text((10, 5), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_judgment(verdict="A", confidence="high"):
    return {
        "originals_summary": "a blue circle",
        "visible_differences": ["colour"],
        "requirements": [{
            "id": "c1", "category": "change", "severity": "critical", "description": "circle is red",
            "a": {"status": "pass" if verdict == "A" else "fail", "evidence": "A: red"},
            "b": {"status": "pass" if verdict == "B" else "fail", "evidence": "B: green"},
        }],
        "decisive_difference": f"Only {verdict} has a red circle.",
        "reasoning": "colour is the core instruction",
        "verdict": verdict,
        "confidence": confidence,
    }


class FakeJudge:
    """Returns scripted verdicts. `script` maps (swapped) -> verdict, or is a list consumed in order."""

    def __init__(self, verdict_for=None, error=None):
        self.verdict_for = verdict_for or (lambda task, swapped: ("A", "high"))
        self.error = error
        self.calls = []

    async def judge_once(self, task, config, swapped, notes):
        self.calls.append({"swapped": swapped, "notes": notes, "prompt": task.prompt})
        if self.error:
            raise self.error
        verdict, conf = self.verdict_for(task, swapped)
        return {"judgment": make_judgment(verdict, conf), "model": "fake",
                "usage": {"input_tokens": 100, "output_tokens": 50}}


@pytest.fixture
def fake_judge():
    return FakeJudge()


@pytest.fixture
def db(tmp_path):
    from app.db import DB
    return DB(tmp_path / "test.sqlite3")
