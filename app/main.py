"""FastAPI app: the judge UI, training/testing on task sets, the benchmark UI, and their JSON APIs."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .benchmark import DatasetError, load_dataset, run_benchmark
from .config import has_key, provider_of, settings
from .db import DB, open_db
from .auth import install_auth
from .autotrain import MY_TASKS_SET, gate_candidate, pending_corrections
from .storage import StorageError, get_store
from .images import ImageError, prepare_image
from .judge import JudgeError, Task, evaluate
from .learn import (SetError, compare_with_baseline, learn_from_correction, refresh_labels, run_set,
                    train)
from .providers import RoutingJudge
from .rubric import RUBRICS

STATIC = Path(__file__).parent / "static"
UPLOAD_NAME = re.compile(r"^[0-9a-f]{64}\.[a-z0-9]{1,5}$")


# Request bodies live at module level: with postponed annotations FastAPI can't
# resolve classes defined inside create_app().
class ReasonBody(BaseModel):
    reason: str
    learn: bool = True  # also learn from this answer now that it is explained


class Feedback(BaseModel):
    # Right/Wrong on a verdict, or (when there was no verdict) which result was correct.
    verdict_correct: Optional[bool] = None
    true_label: Optional[Literal["A", "B"]] = None
    reason: str = ""      # why the correct result is correct (the judge learns from this)
    learn: bool = True    # learn straight away when it was wrong or unsure




class BenchmarkRequest(BaseModel):
    dataset: str
    runs: Optional[int] = None
    model: Optional[str] = None
    rubric_version: Optional[str] = None
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    limit: Optional[int] = None
    use_cache: bool = True
    note: str = ""


class SetBody(BaseModel):
    name: str


class LabelBody(BaseModel):
    label: Optional[Literal["A", "B"]] = None


class ImportBody(BaseModel):
    path: str


class KnowledgeBody(BaseModel):
    guidelines: str = ""
    lessons: list[str] = []
    parent_id: Optional[int] = None


class ActivateBody(BaseModel):
    id: Optional[int] = None  # None = judge without guidelines or lessons


class RunRequest(BaseModel):
    mode: Literal["train", "test", "judge"]
    set_id: int
    # "active" (default), a knowledge version id, or "none" for no guidelines/lessons.
    knowledge: Union[Literal["active", "none"], int] = "active"
    runs: Optional[int] = None
    model: Optional[str] = None
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None
    compare_baseline: bool = False  # test/judge: also run without knowledge, to see what training changed
    max_cases: int = 15             # train: most mistakes to write lessons from
    use_cache: bool = True
    note: str = ""


def create_app(db: Optional[DB] = None, judge=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs left 'running' by a previous server process can never finish.
        app.state.db.mark_interrupted_runs()
        yield

    app = FastAPI(title="Image Judge", lifespan=lifespan)
    app.state.db = db or open_db(settings)
    app.state.judge = judge
    app.state.jobs = {}
    app.state.learning = {}               # evaluation id -> {"status": ..., "result"/"error": ...}
    app.state.learn_lock = asyncio.Lock()  # one merge at a time, or two corrections overwrite each other
    app.state.gate = {"status": "idle", "knowledge_id": None, "error": None}  # the auto-training test
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    if settings.auth_required:
        install_auth(app, settings)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"ok": True}

    def refuse_server_paths() -> None:
        """Importing a dataset reads a path on the server's own disk. Fine on your PC, a file leak when hosted."""
        if settings.auth_required:
            raise HTTPException(403, "Importing from a server folder is turned off in the hosted app. "
                                     "Upload tasks from the browser instead.")

    def get_judge():
        if app.state.judge is None:
            app.state.judge = RoutingJudge()
        return app.state.judge

    def db() -> DB:
        return app.state.db

    def current_config():
        """Default judge settings plus whichever knowledge version is active."""
        return settings.judge.with_knowledge(db().active_knowledge())

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/train", include_in_schema=False)
    def train_page():
        return FileResponse(STATIC / "train.html")

    @app.get("/help", include_in_schema=False)
    def help_page():
        return FileResponse(STATIC / "help.html")

    @app.get("/lessons", include_in_schema=False)
    def lessons_page():
        return FileResponse(STATIC / "lessons.html")

    @app.get("/benchmark", include_in_schema=False)
    def benchmark_page():
        return FileResponse(STATIC / "benchmark.html")

    @app.get("/api/config")
    def config():
        active = db().active_knowledge()
        return {
            "defaults": settings.judge.as_dict(),
            "rubric_versions": sorted(RUBRICS),
            "provider": provider_of(settings.judge.model),
            "api_key_configured": has_key(provider_of(settings.judge.model)),
            "active_knowledge": None if not active else {"id": active["id"], "lessons": len(active["lessons"])},
        }

    # --- uploads ---------------------------------------------------------
    async def _read(upload: UploadFile, what: str) -> bytes:
        raw = await upload.read()
        if len(raw) > settings.max_upload_bytes:
            raise HTTPException(413, f"{what} is larger than {settings.max_upload_bytes // 2**20} MB")
        if not raw:
            raise HTTPException(400, f"{what} is empty")
        return raw

    def _save_upload(raw: bytes, sha: str, filename: str | None) -> str:
        ext = (Path(filename or "").suffix.lower() or ".png")[:6]
        if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
            ext = ".png"
        name = f"{sha}{ext}"
        # Rewritten if missing or damaged (e.g. left empty when the disk was full); the store writes a
        # temp file first so a failed write never leaves a broken image, then mirrors it to the bucket.
        try:
            get_store().save(name, raw)
        except OSError as exc:
            raise HTTPException(507, f"Could not save the image ({exc.strerror}). Is the disk full?") from exc
        except StorageError as exc:
            raise HTTPException(502, f"Could not save the image to storage: {exc}") from exc
        return name

    async def _read_task_images(originals, result_a, result_b):
        if not 1 <= len(originals) <= 10:
            raise HTTPException(400, "Add between 1 and 10 original images")
        try:
            raw_originals = [await _read(f, f"Original {i}") for i, f in enumerate(originals, 1)]
            raw_a, raw_b = await _read(result_a, "Result A"), await _read(result_b, "Result B")
            prepared = await asyncio.to_thread(lambda: [prepare_image(r) for r in [*raw_originals, raw_a, raw_b]])
        except ImageError as exc:
            raise HTTPException(400, str(exc)) from exc
        images = {
            "originals": [_save_upload(r, p.sha256, f.filename) for r, p, f in zip(raw_originals, prepared, originals)],
            "a": _save_upload(raw_a, prepared[-2].sha256, result_a.filename),
            "b": _save_upload(raw_b, prepared[-1].sha256, result_b.filename),
        }
        return prepared, images

    @app.get("/api/images/{name}", include_in_schema=False)
    def image(name: str):
        # Only content-addressed files from the upload folder; no paths.
        if not UPLOAD_NAME.match(name):
            raise HTTPException(404, "No such image")
        try:
            return FileResponse(get_store().ensure(name))
        except FileNotFoundError as exc:
            raise HTTPException(404, "No such image") from exc
        except StorageError as exc:
            raise HTTPException(502, f"Image storage error: {exc}") from exc

    # --- single evaluation -----------------------------------------------
    @app.post("/api/evaluate")
    async def evaluate_endpoint(
        prompt: str = Form(...),
        originals: list[UploadFile] = File(...),
        result_a: UploadFile = File(...),
        result_b: UploadFile = File(...),
        fresh: bool = Form(False),
    ):
        if not prompt.strip():
            raise HTTPException(400, "Prompt is empty")
        prepared, images = await _read_task_images(originals, result_a, result_b)
        task = Task(prompt=prompt, originals=prepared[:-2], a=prepared[-2], b=prepared[-1])
        cfg = current_config()
        result = await evaluate(task, cfg, get_judge(), db=db(), use_cache=not fresh)
        if result["aggregate"]["status"] == "error":
            raise HTTPException(502, result["aggregate"]["explanation"])
        eval_id = db().add_evaluation(prompt, images, cfg.as_dict(), result)
        return {"id": eval_id, **result}

    @app.post("/api/evaluations/{eval_id}/feedback")
    async def feedback(eval_id: int, body: Feedback):
        ev = db().get_evaluation(eval_id)
        if not ev:
            raise HTTPException(404, "No such evaluation")
        verdict = ev["verdict"]
        if body.verdict_correct is not None:
            if verdict is None:
                raise HTTPException(400, "This evaluation had no verdict; send true_label instead")
            true_label = verdict if body.verdict_correct else ("B" if verdict == "A" else "A")
            verdict_correct = body.verdict_correct
        elif body.true_label:
            true_label = body.true_label
            verdict_correct = None if verdict is None else verdict == true_label
        else:
            raise HTTPException(400, "Send verdict_correct or true_label")
        db().set_feedback(eval_id, verdict_correct, true_label, body.reason)

        # File it under "My tasks" with the right answer (once per evaluation), for later tests.
        task_id = db().evaluation_task_id(eval_id)
        if task_id and db().get_set_task(task_id):
            db().set_task_label(task_id, true_label)
        else:
            set_id = db().get_or_create_set(MY_TASKS_SET)
            task_id = db().add_set_task(set_id, ev["prompt"], ev["images"], true_label)
            db().link_evaluation_task(eval_id, task_id)

        out = {"ok": True, "verdict_correct": verdict_correct, "true_label": true_label, "learning": None}
        unsure = ev["status"] != "confident"
        if body.learn and (verdict_correct is not True or unsure):
            start_learning(eval_id, ev, true_label, body.reason)  # in the background: you can move on
            out["learning"] = "started"
        return out

    # --- learning from corrections, gated by a held-out test ---------------
    def gate_busy() -> bool:
        return app.state.gate["status"] == "running"

    def start_gate(candidate_id: int) -> None:
        app.state.gate = {"status": "running", "knowledge_id": candidate_id, "error": None}

        async def gate_job():
            try:
                result = await gate_candidate(db(), candidate_id, settings.judge, get_judge())
            except Exception as exc:
                app.state.gate = {"status": "failed", "knowledge_id": candidate_id,
                                  "error": f"{exc}" if isinstance(exc, JudgeError) else f"{type(exc).__name__}: {exc}"}
            else:
                app.state.gate = {"status": "done", "knowledge_id": candidate_id, "error": None, "result": result}

        app.state.jobs[f"gate-{candidate_id}"] = asyncio.create_task(gate_job())

    def maybe_start_gate() -> None:
        """Test the waiting candidate once enough new corrections have piled up behind it."""
        cand = db().pending_candidate()
        if cand and not gate_busy() and pending_corrections(cand, db().active_knowledge()) >= settings.gate_every:
            start_gate(cand["id"])

    def start_learning(eval_id: int, ev: dict, true_label: str, reason: str) -> None:
        app.state.learning[eval_id] = {"status": "running"}
        task_id = db().evaluation_task_id(eval_id)

        async def learn_job():
            async with app.state.learn_lock:
                try:
                    learned = await learn_from_correction(db(), ev, true_label, reason, current_config(), get_judge(),
                                                          gated=settings.auto_gate, task_id=task_id)
                except Exception as exc:
                    app.state.learning[eval_id] = {
                        "status": "failed",
                        "error": f"{exc}" if isinstance(exc, JudgeError) else f"{type(exc).__name__}: {exc}"}
                else:
                    app.state.learning[eval_id] = {"status": "done", "result": {
                        "new_lessons": learned["new_lessons"], "knowledge_id": learned["knowledge_id"],
                        "total_lessons": len(learned["lessons"]), "changes": learned["changes"],
                        "what_judge_missed": learned["case"]["what_judge_missed"], "status": learned["status"]}}
                    if learned["status"] == "candidate":
                        maybe_start_gate()

        app.state.jobs[f"learn-{eval_id}"] = asyncio.create_task(learn_job())

    @app.get("/api/answers/missing-reasons")
    def missing_reasons(limit: int = 100):
        return {"items": db().answers_missing_reason(min(max(limit, 1), 500)), "coverage": db().reason_coverage()}

    @app.post("/api/evaluations/{eval_id}/reason")
    async def add_reason(eval_id: int, body: ReasonBody):
        ev = db().get_evaluation(eval_id)
        if not ev or not ev["feedback"]:
            raise HTTPException(404, "There is no saved answer for this evaluation")
        if len(body.reason.strip()) < 8:
            raise HTTPException(400, "Write a short reason (a sentence) saying why that result is the better one.")
        db().set_feedback_reason(eval_id, body.reason)
        started = False
        if body.learn:
            start_learning(eval_id, ev, ev["feedback"]["true_label"], body.reason)
            started = True
        return {"ok": True, "learning": "started" if started else None}

    @app.get("/api/evaluations/{eval_id}/learning")
    def learning_status(eval_id: int):
        return app.state.learning.get(eval_id, {"status": "none"})

    @app.get("/api/learning")
    def learning_overview():
        running = sum(1 for v in app.state.learning.values() if v["status"] == "running")
        return {"running": running}

    @app.get("/api/evaluations")
    def evaluations(limit: int = 50):
        return {"items": db().list_evaluations(limit), "feedback_stats": db().feedback_stats()}

    @app.get("/api/evaluations/{eval_id}")
    def evaluation(eval_id: int):
        ev = db().get_evaluation(eval_id)
        if not ev:
            raise HTTPException(404, "No such evaluation")
        return ev

    # --- task sets -------------------------------------------------------
    @app.get("/api/sets")
    def sets():
        return {"items": db().list_sets()}

    @app.post("/api/sets")
    def create_set(body: SetBody):
        if not body.name.strip():
            raise HTTPException(400, "Give the set a name")
        if any(s["name"] == body.name.strip() for s in db().list_sets()):
            raise HTTPException(400, f"A set called '{body.name.strip()}' already exists")
        return {"id": db().create_set(body.name)}

    @app.get("/api/sets/{set_id}")
    def get_set(set_id: int):
        s = db().get_set(set_id)
        if not s:
            raise HTTPException(404, "No such set")
        s["runs"] = db().runs_for_set(set_id)
        return s

    @app.patch("/api/sets/{set_id}")
    def rename_set(set_id: int, body: SetBody):
        if not db().get_set(set_id):
            raise HTTPException(404, "No such set")
        db().rename_set(set_id, body.name)
        return {"ok": True}

    @app.delete("/api/sets/{set_id}")
    def delete_set(set_id: int):
        db().delete_set(set_id)
        return {"ok": True}

    @app.post("/api/sets/tasks")
    async def add_task(
        prompt: str = Form(...),
        originals: list[UploadFile] = File(...),
        result_a: UploadFile = File(...),
        result_b: UploadFile = File(...),
        set_id: Optional[int] = Form(None),
        new_set_name: str = Form(""),
        label: str = Form(""),
    ):
        """Save a task (prompt + images + the correct answer if known) into a set, without judging it."""
        if not prompt.strip():
            raise HTTPException(400, "Prompt is empty")
        if label not in ("", "A", "B"):
            raise HTTPException(400, "label must be A, B or empty")
        if set_id is None:
            if not new_set_name.strip():
                raise HTTPException(400, "Choose a set or name a new one")
            set_id = db().get_or_create_set(new_set_name)
        elif not db().get_set(set_id):
            raise HTTPException(404, "No such set")
        _, images = await _read_task_images(originals, result_a, result_b)
        task_id = db().add_set_task(set_id, prompt, images, label or None)
        s = next(x for x in db().list_sets() if x["id"] == set_id)
        return {"id": task_id, "set": s}

    @app.patch("/api/tasks/{task_id}")
    def label_task(task_id: int, body: LabelBody):
        if not db().get_set_task(task_id):
            raise HTTPException(404, "No such task")
        db().set_task_label(task_id, body.label)
        return {"ok": True}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: int):
        db().delete_set_task(task_id)
        return {"ok": True}

    @app.post("/api/sets/{set_id}/import")
    def import_tasks(set_id: int, body: ImportBody):
        """Copy a dataset folder/CSV (see README) into a set."""
        refuse_server_paths()
        if not db().get_set(set_id):
            raise HTTPException(404, "No such set")
        try:
            tasks = load_dataset(body.path)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from exc
        import hashlib
        for t in tasks:
            def keep(p: Path) -> str:
                raw = p.read_bytes()
                return _save_upload(raw, hashlib.sha256(raw).hexdigest(), p.name)
            images = {"originals": [keep(p) for p in t.originals], "a": keep(t.a), "b": keep(t.b)}
            db().add_set_task(set_id, t.prompt, images, t.label if t.label in ("A", "B") else None)
        return {"imported": len(tasks)}

    # --- knowledge (guidelines + lessons) --------------------------------
    @app.get("/api/knowledge")
    def knowledge():
        items = db().list_knowledge()
        by_id = {k["id"]: k for k in items}
        for k in items:
            parent = by_id.get(k["parent_id"])
            before = parent["lessons"] if parent else []
            k["added"] = [l for l in k["lessons"] if l not in before]
            k["removed"] = [l for l in before if l not in k["lessons"]]
            new_ids = set(k["learned_from"]) - set(parent["learned_from"] if parent else [])
            k["new_tasks"] = len(new_ids)
            k["taught_by"] = sorted({t["labelled_by"] for tid in new_ids
                                     if (t := db().get_set_task(tid)) and t["labelled_by"]})
        return {"items": items, "reasons": db().reason_coverage(),
                "gate": {"enabled": settings.auto_gate, "every": settings.gate_every,
                         "min_pool": settings.gate_min_pool, "max_tasks": settings.gate_max_tasks,
                         "state": app.state.gate}}

    @app.post("/api/knowledge/gate")
    def test_candidate_now():
        cand = db().pending_candidate()
        if not cand:
            raise HTTPException(400, "No new lessons are waiting for a test.")
        if gate_busy():
            raise HTTPException(409, "A test is already running.")
        start_gate(cand["id"])
        return {"started": cand["id"]}

    @app.post("/api/knowledge/{knowledge_id}/reject")
    def reject_knowledge(knowledge_id: int):
        k = db().get_knowledge(knowledge_id)
        if not k:
            raise HTTPException(404, "No such lessons version")
        if k["active"]:
            raise HTTPException(400, "These lessons are in use. Go back to another version first.")
        db().set_knowledge_status(knowledge_id, "rejected", k["gate_result"])
        return {"ok": True}

    @app.post("/api/knowledge/undo")
    def undo_knowledge():
        """Go back to the version the active one was built from (skipping ones that never passed a test)."""
        active = db().active_knowledge()
        if not active:
            raise HTTPException(400, "No lessons are in use.")
        target = db().get_knowledge(active["parent_id"]) if active["parent_id"] else None
        while target and target["status"] != "accepted":
            target = db().get_knowledge(target["parent_id"]) if target["parent_id"] else None
        if not target:
            raise HTTPException(400, "There is no earlier version to go back to. "
                                     "Use 'Switch off all lessons' to judge with none.")
        db().activate_knowledge(target["id"])
        return {"active": target["id"]}

    @app.post("/api/knowledge")
    def save_knowledge(body: KnowledgeBody):
        """Edited guidelines/lessons are saved as a new version and made active."""
        lessons = [l.strip() for l in body.lessons if l.strip()]
        kid = db().add_knowledge(body.guidelines.strip(), lessons, "Edited by hand", parent_id=body.parent_id)
        return {"id": kid}

    @app.post("/api/knowledge/activate")
    def activate(body: ActivateBody):
        if body.id is not None and not db().get_knowledge(body.id):
            raise HTTPException(404, "No such knowledge version")
        db().activate_knowledge(body.id)
        return {"ok": True}

    # --- train / test / judge runs ---------------------------------------
    def _fail(run_id: Optional[int], exc: Exception) -> None:
        if run_id is None:
            return
        run = db().get_benchmark(run_id, include_items=False)
        if run and run["status"] in ("running", "failed"):
            metrics = dict(run["metrics"] or {})
            metrics["error"] = str(exc)
            db().finish_benchmark(run_id, metrics, status="failed")

    def _start(run_id: int, coro) -> None:
        async def job():
            try:
                await coro
            except Exception as exc:  # the run row carries the error for the UI
                _fail(run_id, exc)
            finally:
                app.state.jobs.pop(run_id, None)
        app.state.jobs[run_id] = asyncio.create_task(job())

    @app.post("/api/runs")
    async def start_run(body: RunRequest):
        if body.runs is not None and not 1 <= body.runs <= 16:
            raise HTTPException(400, "runs must be between 1 and 16")
        s = db().get_set(body.set_id)
        if not s:
            raise HTTPException(404, "No such set")
        if body.knowledge == "active":
            know = db().active_knowledge()
        elif body.knowledge == "none":
            know = None
        else:
            know = db().get_knowledge(body.knowledge)
            if not know:
                raise HTTPException(404, "No such knowledge version")
        cfg = settings.judge.with_overrides(runs=body.runs, model=body.model or None,
                                            effort=body.effort).with_knowledge(know)
        labeled = sum(1 for t in s["tasks"] if t["label"] in ("A", "B"))
        if body.mode == "train":
            if not labeled:
                raise HTTPException(400, "Training needs tasks with the correct answer marked")
            total = labeled
        else:
            if not s["tasks"]:
                raise HTTPException(400, "This set has no tasks")
            total = len(s["tasks"])

        run_id = db().start_benchmark(f"set: {s['name']}", cfg.as_dict(), total, body.note,
                                      mode=body.mode, set_id=body.set_id)
        baseline_id = None
        if body.mode == "train":
            coro = train(db(), body.set_id, cfg, get_judge(), use_cache=body.use_cache, max_cases=body.max_cases,
                         note=body.note, run_id=run_id)
        else:
            if body.compare_baseline and know:
                baseline_id = db().start_benchmark(
                    f"set: {s['name']}", cfg.with_knowledge(None).as_dict(), total,
                    f"untrained comparison for run #{run_id}", mode=body.mode, set_id=body.set_id,
                    baseline_of=run_id)
            coro = run_set(db(), body.set_id, body.mode, cfg, get_judge(), use_cache=body.use_cache,
                           compare_baseline=baseline_id is not None, note=body.note, run_id=run_id,
                           baseline_run_id=baseline_id)

        async def guarded():
            try:
                await coro
            except Exception as exc:
                _fail(baseline_id, exc)
                raise
        _start(run_id, guarded())
        return {"id": run_id, "baseline_id": baseline_id, "total": total}

    # --- benchmark runs (datasets on disk) -------------------------------
    @app.post("/api/benchmarks")
    async def start_benchmark(body: BenchmarkRequest):
        refuse_server_paths()
        if body.rubric_version and body.rubric_version not in RUBRICS:
            raise HTTPException(400, f"Unknown rubric version {body.rubric_version}")
        if body.runs is not None and not 1 <= body.runs <= 16:
            raise HTTPException(400, "runs must be between 1 and 16")
        try:
            tasks = load_dataset(body.dataset)
        except DatasetError as exc:
            raise HTTPException(400, str(exc)) from exc
        cfg = current_config().with_overrides(  # blank fields mean "use the default"
            runs=body.runs, model=body.model or None, rubric_version=body.rubric_version or None,
            effort=body.effort)
        total = min(len(tasks), body.limit) if body.limit else len(tasks)
        run_id = db().start_benchmark(body.dataset, cfg.as_dict(), total, body.note)
        _start(run_id, run_benchmark(body.dataset, cfg, get_judge(), db(), limit=body.limit,
                                     use_cache=body.use_cache, note=body.note, run_id=run_id))
        return {"id": run_id, "total": total}

    @app.get("/api/benchmarks")
    def benchmarks(mode: Optional[str] = None):
        return {"items": db().list_benchmarks(mode=mode)}

    @app.get("/api/benchmarks/{run_id}")
    def benchmark(run_id: int):
        run = db().get_benchmark(run_id)
        if not run:
            raise HTTPException(404, "No such benchmark run")
        run = refresh_labels(db(), run)
        baseline = next((r for r in db().list_benchmarks(limit=500) if r.get("baseline_of") == run_id), None)
        if baseline:
            baseline = refresh_labels(db(), db().get_benchmark(baseline["id"]))
            run["baseline"] = {k: baseline[k] for k in ("id", "status", "completed", "total", "metrics")}
            run["baseline"]["items"] = [{"task_id": i["task_id"], "verdict": i["verdict"], "status": i["status"],
                                         "correct": i["correct"]} for i in baseline["items"]]
            if baseline["status"] == "done":
                run["comparison"] = compare_with_baseline(run["items"], baseline["items"])
        if run.get("learned_knowledge_id"):
            run["learned_knowledge"] = db().get_knowledge(run["learned_knowledge_id"])
        return run

    return app


app = create_app()
