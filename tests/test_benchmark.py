import asyncio
import csv

import pytest

from app import benchmark as bm
from app.config import JudgeConfig
from conftest import FakeJudge, png_bytes

CFG = JudgeConfig(model="fake", runs=4, rubric_version="v1", effort="high")


def write_csv_dataset(root, rows):
    (root / "img").mkdir(parents=True, exist_ok=True)
    colors = {"orig": (40, 90, 200), "red": (220, 30, 30), "green": (30, 160, 60)}
    for name, col in colors.items():
        (root / "img" / f"{name}.png").write_bytes(png_bytes(col))
    with (root / "tasks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "originals", "result_a", "result_b", "correct_label"])
        w.writeheader()
        w.writerows(rows)


def row(i, label, a="img/red.png", b="img/green.png"):
    return {"id": f"t{i}", "prompt": "Make the circle red", "originals": "img/orig.png",
            "result_a": a, "result_b": b, "correct_label": label}


def test_wilson_lower_bound():
    assert bm.wilson_lower(0, 0) is None
    assert bm.wilson_lower(50, 50) == pytest.approx(0.9287, abs=1e-3)
    # 98/100 is not enough evidence for ">= 98%".
    assert bm.wilson_lower(98, 100) < 0.95


def test_load_csv_dataset(tmp_path):
    write_csv_dataset(tmp_path, [row(1, "A"), row(2, "b")])
    tasks = bm.load_dataset(tmp_path)
    assert [t.id for t in tasks] == ["t1", "t2"]
    assert tasks[1].label == "B"
    assert tasks[0].originals[0].name == "orig.png"


def test_load_dataset_reports_problems(tmp_path):
    write_csv_dataset(tmp_path, [row(1, "C"), row(2, "A", a="img/missing.png")])
    with pytest.raises(bm.DatasetError) as exc:
        bm.load_dataset(tmp_path)
    assert "correct_label" in str(exc.value)
    assert "missing.png" in str(exc.value)


def test_load_folder_dataset(tmp_path):
    t = tmp_path / "task1"
    t.mkdir()
    (t / "prompt.txt").write_text("Make the circle red")
    (t / "label.txt").write_text("b\n")
    (t / "original_1.png").write_bytes(png_bytes())
    (t / "original_2.png").write_bytes(png_bytes((1, 1, 1)))
    (t / "a.png").write_bytes(png_bytes((30, 160, 60)))
    (t / "result_b.jpg").write_bytes(png_bytes((220, 30, 30)))
    [task] = bm.load_dataset(tmp_path)
    assert task.label == "B" and len(task.originals) == 2 and task.b.name == "result_b.jpg"


def item(status, verdict, label, runs=()):
    return {"task_id": "x", "status": status, "verdict": verdict, "label": label,
            "correct": None if verdict is None else verdict == label, "runs": list(runs)}


def r(i, v, conf="high"):
    return {"index": i, "swapped": i % 2 == 1, "verdict": v, "confidence": conf, "error": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_compute_metrics_definitions():
    items = [
        item("confident", "A", "A", [r(0, "A"), r(1, "A")]),
        item("confident", "A", "A", [r(0, "A"), r(1, "A")]),
        item("confident", "B", "A", [r(0, "B"), r(1, "B")]),   # confident and wrong
        item("review", "B", "B", [r(0, "B"), r(1, "B", "low")]),
        item("unclear", None, "A", [r(0, "A"), r(1, "B")]),
    ]
    m = bm.compute_metrics(items)
    assert m["counts"] == {"confident": 3, "review": 1, "unclear": 1, "error": 0}
    assert m["confident_accuracy"] == pytest.approx(2 / 3)
    assert m["coverage"] == pytest.approx(3 / 5)
    assert m["overall_accuracy"] == pytest.approx(3 / 5)
    assert m["answered_accuracy"] == pytest.approx(3 / 4)
    assert m["single_run_accuracy"] == pytest.approx(7 / 10)
    assert m["api_tokens"] == {"input": 100, "output": 50}


def test_run_count_curve_reaggregates_prefixes():
    items = [
        item("review", "A", "A", [r(0, "A"), r(1, "A"), r(2, "B"), r(3, "A")]),
        item("confident", "B", "B", [r(0, "B"), r(1, "B"), r(2, "B"), r(3, "B")]),
    ]
    curve = {c["runs"]: c for c in bm.run_count_curve(items)}
    assert curve[1]["coverage"] == 0  # one run is never confident
    assert curve[2]["coverage"] == 1.0
    assert curve[4]["coverage"] == 0.5


def test_failure_list_puts_confident_wrong_first():
    items = [item("unclear", None, "A"), item("confident", "B", "A"), item("confident", "A", "A"),
             item("review", "B", "A")]
    assert [f["status"] for f in bm.failure_list(items)] == ["confident", "review", "unclear"]


def test_run_benchmark_end_to_end(tmp_path, db, monkeypatch):
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    write_csv_dataset(tmp_path / "ds", [row(1, "A"), row(2, "A"), row(3, "B")])
    judge = FakeJudge()  # always answers A
    out = asyncio.run(bm.run_benchmark(tmp_path / "ds", CFG, judge, db))
    assert out.metrics["confident_accuracy"] == pytest.approx(2 / 3)
    assert out.metrics["confident_wrong"] == 1
    assert out.results_file.exists()
    stored = db.get_benchmark(out.run_id)
    assert stored["status"] == "done" and len(stored["items"]) == 3
    report = bm.format_report(out, CFG)
    assert "Accuracy on confident verdicts" in report and "t3" in report


def test_run_benchmark_survives_broken_image(tmp_path, db, monkeypatch):
    monkeypatch.setattr(bm.settings, "results_dir", tmp_path / "results", raising=False)
    write_csv_dataset(tmp_path / "ds", [row(1, "A"), row(2, "A", b="img/broken.png")])
    (tmp_path / "ds" / "img" / "broken.png").write_bytes(b"not an image")
    out = asyncio.run(bm.run_benchmark(tmp_path / "ds", CFG, FakeJudge(), db))
    assert out.metrics["counts"]["error"] == 1
    assert out.metrics["counts"]["confident"] == 1
