# Image Judge

Decides which of two result images is the correct output of an image
generation/editing prompt, and measures how often it's right.

- **Judge page** (`/`): paste a prompt, 1–10 originals, Result A and Result B → verdict,
  confidence status, the decisive difference, and a requirement-by-requirement
  pass/fail table. Mark the verdict Right/Wrong; feedback is stored.
- **Benchmark page** (`/benchmark`) and **CLI**: run the pipeline over a labelled
  dataset and report accuracy on confident verdicts, coverage, overall accuracy
  and every failure with the model's reasoning.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # macOS/Linux: .venv/bin/pip
cp .env.example .env                             # or use the local Codex/Claude CLI login
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Train, test, judge (the main workflow)

Page: **Train & Test** (`/train`). Three modes, pick one per run:

| mode | needs answers? | what it does |
|---|---|---|
| **Train** | yes | Judges the set, studies every task it got wrong or wasn't sure about (it sees the images, the correct answer and its own reasoning), writes general lessons, merges them into one list (max 20), and saves a new **lessons version** that becomes active. |
| **Test (score it)** | before or after | Judges the set without seeing the answers and scores it. Tick "also run without lessons" to see what training changed (fixed / broke per task). |
| **Judge only** | no | Verdicts for every task. Mark answers afterwards if you want a score. |

Typical round: save 40 labelled tasks into "Train 1" (Judge page → pick the correct answer →
**Save task to set**), run **Train** on it, save 30 new tasks into "Test 1", run **Test**. Then train
again on another 40 ("Train 2"): lessons build on the previous version. Use **Judge only** for real work.

What "training" is and isn't: the model's weights never change. Training writes rules into the
judge's instructions, like notes a new labeller keeps. That's why a **test set must be tasks it was
never trained on**: lessons are only ever written from training sets, so a test score is an honest,
held-out number. With 30 test tasks one or two tasks is ~3-7%, so treat small differences as noise.

Also on that page:
- **Lessons & guidelines**: every version, where it came from, its lessons; make any version active,
  judge without lessons, or edit and save a new version. Paste your project's own labelling
  guidelines here - they go to the judge with every task and override its general rubric.
- If the lesson writer thinks a marked answer is wrong, it says so and doesn't learn from that task.
- The Judge page and benchmarks always use the active lessons.

CLI equivalents:

```bash
python -m app.cli train --set "Train 1"
python -m app.cli test --set "Test 1" --compare
python -m app.cli judge --set "Batch 3"
```

## How a verdict is produced

1. Images are decoded, EXIF-rotated, and downscaled to at most 1568 px on the long edge / 1.15 MP
   (what the vision model actually uses), then sent as PNG (JPEG if large and opaque).
2. Cheap pixel checks run first. If A and B are identical, the answer is "unclear" with no API call.
   If a result is identical to an original ("edit not applied"), that fact is given to the model.
3. **N independent judge calls** (default 4) run concurrently with the rubric in
   [`app/rubric.py`](app/rubric.py). Odd-numbered runs show B before A, so position bias shows
   up as disagreement instead of a wrong answer. Labels always stay the true ones, so the
   model's evidence text never needs relabelling.
4. Each call returns schema-constrained JSON: originals summary → visible differences →
   change / preserve / quality requirements with pass/fail/unsure + evidence for A and B →
   decisive difference → reasoning → verdict → confidence (evidence is written before the verdict).
5. Aggregation ([`app/aggregate.py`](app/aggregate.py)):
   - **confident**: every planned run succeeded, all agree, none low-confidence, ≥2 runs
   - **review**: a strict majority of planned runs agree (shown as "Leaning A/B")
   - **unclear**: anything else — no verdict is forced

### Providers and models

The provider is chosen from the model name, so you can switch with `IMAGE_JUDGE_MODEL` or
`--model` and even mix providers in one sweep:

| model | provider | key |
|---|---|---|
| codex:gpt-6-astra (or codex) | your local Codex CLI | none: uses your Codex login and plan |
| `claude-code:opus` (or `claude-code:fable`, `claude-code`) | your local Claude Code CLI | none: uses your `claude` login and plan |
| `gemini-3.1-pro-preview` (default when only a Gemini key is set) | Google | `GEMINI_API_KEY` |
| `gemini-3.8-flash` (cheaper/faster; try it if Pro isn't on your quota) | Google | `GEMINI_API_KEY` |
| `claude-fable-5-1` (default otherwise), `claude-opus-5-5` | Anthropic | `ANTHROPIC_API_KEY` |

### MWAPI / Anthropic-compatible gateway

Private provider settings live in .env, which is excluded by .gitignore. Change the key there and
restart the local server:

    ANTHROPIC_BASE_URL=https://api.mwapi.dev
    ANTHROPIC_API_KEY=replace-with-your-key
    IMAGE_JUDGE_MODEL=claude-opus-5
    IMAGE_JUDGE_FALLBACKS=0
    IMAGE_JUDGE_MAX_CONCURRENCY=1

The SDK adds /v1/messages, so keep /v1 off ANTHROPIC_BASE_URL. Available MWAPI model values
for this setup are claude-sonnet-5, claude-haiku-4-5-20251001, claude-opus-4-8, and
claude-opus-5.

**Claude Code (no API key):** each judge call runs `claude -p` on this machine with the images
piped in as JSON, all tools disabled, `--safe-mode` (no CLAUDE.md/hooks/MCP), and
`--json-schema` holding the output to the rubric schema. Measured here: about 35 s per call.
It uses your plan's usage limits (Fable needs usage credits on some plans), so keep
`IMAGE_JUDGE_MAX_CONCURRENCY` low. Hitting a usage limit stops the run instead of failing
every task. It only works where `claude` is installed and logged in; set
`IMAGE_JUDGE_CLAUDE_BIN` if it isn't on PATH. For an app other people use, use an API key.

**Codex (no API key):** each call runs codex exec ephemerally with the prepared images and the
rubric's JSON schema. It uses your existing Codex login and plan, ignores repository/user rules,
and receives an empty read-only temporary workspace. Use codex for the CLI's configured default
model or codex:<model-id> for an exact model. Set IMAGE_JUDGE_CODEX_BIN if codex is not on
PATH. Keep IMAGE_JUDGE_MAX_CONCURRENCY=1 unless your plan comfortably supports parallel calls.

Get a Gemini key at https://aistudio.google.com/apikey. Gemini runs use JSON-schema output,
high media resolution (small details decide verdicts), and `effort` mapped to Gemini's
thinking level (`low`/`medium`/`high`; `xhigh`/`max` map to `high`). On the free tier, set
`IMAGE_JUDGE_MAX_CONCURRENCY=1` or `2` to stay under the per-minute limits.

Rate limits and transient errors: the SDK retries 429/5xx/connection errors with exponential
backoff (`IMAGE_JUDGE_API_RETRIES`, default 5), and a process-wide semaphore caps concurrent
calls (`IMAGE_JUDGE_MAX_CONCURRENCY`, default 4). A bad key or unknown model stops the run
immediately instead of failing every task.

## Benchmark datasets

Either a CSV (or a folder containing `tasks.csv`):

```csv
id,prompt,originals,result_a,result_b,correct_label
t001,"Make the car blue",img/t001_orig.png,img/t001_a.png,img/t001_b.png,B
t002,"Put the dog from image 1 on the sofa in image 2",img/t002_o1.png;img/t002_o2.png,img/t002_a.png,img/t002_b.png,A
```

or one folder per task: `prompt.txt`, `label.txt` (A/B), `original*.png` (0–4), `a.png`, `b.png`.
Paths are relative to the CSV / dataset folder.

**Grow the dataset from real use:** every Right/Wrong click records the true label, and
`python -m app.cli export-feedback --out datasets/from_feedback` writes those evaluations
(with their images) as a CSV dataset.

## Running benchmarks

```bash
python -m app.cli benchmark --dataset datasets/mine
python -m app.cli benchmark --dataset datasets/mine --runs 8 --rubric v2 --effort xhigh --note "try v2"
python -m app.cli sweep --dataset datasets/mine --config rubric=v1 --config rubric=v2
python -m app.cli sweep --dataset datasets/mine --config model=gemini-3.1-pro-preview --config model=gemini-3.8-flash
```

The report gives: accuracy on confident verdicts with a 95% Wilson lower bound, coverage,
overall accuracy (unclear counts as wrong), single-run accuracy split by image order
(position bias), a **run-count curve** (what N=1…max would have given, computed from the same
calls), and the failures, confident-and-wrong first. Full results go to
`benchmarks/results/run_NNNN.json`, and each run appends a row to [EXPERIMENTS.md](EXPERIMENTS.md).

Every judge call is cached in SQLite by (model, rubric, effort, images, prompt, run index), so:
- running `--runs 8` once and then `--runs 4` costs nothing extra (runs 0–3 are reused);
- changing the rubric, model or effort invalidates the cache, as it should;
- `--no-cache` forces fresh samples (use it to measure run-to-run variance).

**Reading the numbers honestly:** 98% on 50 confident verdicts has a 95% lower bound of ~89%.
To show ≥98% with confidence you need several hundred confident verdicts with almost no
errors. The synthetic dataset (`scripts/make_synthetic_dataset.py`) is for checking the
plumbing only; its tasks are much easier than real outputs.

## Tests

```bash
.venv/Scripts/python -m pytest
```

Covers aggregation rules, swap handling, caching, the run-count curve, metrics, dataset
loading, the HTTP API, image preparation, and the exact requests sent to the Gemini and
Anthropic APIs and the Claude Code CLI (against fakes).

## Layout

```
app/
  config.py      env-driven settings (model, runs, rubric, effort, concurrency)
  images.py      decode, orient, resize, encode; near-identical pixel check
  rubric.py      versioned rubric prompts + JSON schema
  judge.py       shared types, the Anthropic judge, and the N-run pipeline (evaluate)
  gemini_judge.py  the Gemini judge (google-genai SDK)
  claude_code_judge.py  the Claude Code CLI judge (no API key)
  providers.py   routes each call to Codex, Claude Code, Gemini or Anthropic by model name
  aggregate.py   pure voting logic -> confident / review / unclear
  benchmark.py   dataset loading, benchmark/run engine, metrics, report
  learn.py       train (lessons from mistakes) / test / judge-only runs over task sets
  db.py          SQLite: evaluations, feedback, judge cache, benchmark runs
  main.py        FastAPI routes
  cli.py         benchmark / sweep / export-feedback
  static/        judge, train & test, and benchmark pages (vanilla JS, no build step)
scripts/make_synthetic_dataset.py
tests/
```
