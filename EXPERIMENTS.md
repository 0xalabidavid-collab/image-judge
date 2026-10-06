# Experiments

Every `python -m app.cli benchmark` / `sweep` run appends one row to the table below
(pass `--no-log` to skip). Only measured results go here.

## Planned iterations

Run these on a real labelled dataset (≥200 tasks for numbers you can trust), in order,
keeping whichever change raises the 95% lower bound on confident accuracy without
cutting coverage too much:

1. Baseline: `--rubric v1 --runs 8` on `gemini-3.1-pro-preview` (or `claude-fable-5-1`), effort `high`. Read the run-count
   curve to pick N (the cache makes N=2/4/6 free once N=8 has run).
2. Rubric: `sweep --config rubric=v1 --config rubric=v2`. v2 adds a re-check of the decisive
   region before committing.
3. Effort: `sweep --config effort=medium --config effort=high` (on Gemini, xhigh/max are the same as high).
4. Model: `sweep --config model=gemini-3.1-pro-preview --config model=gemini-3.8-flash`
   (Flash is cheaper; if its confident accuracy matches, it can be used with more runs).
5. Read the confident-and-wrong failures. Each one is either a rubric gap (add guidance as
   a new rubric version), a labelling error (fix the dataset), or ambiguous (drop the task).

## Results

| date | run | dataset | tasks | model | rubric | runs | effort | confident acc. | 95% lower | coverage | overall | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | #6 | synthetic | 35 | claude-code:opus | v1 | 4 | high | 100.0% (35/35) | 90.1% | 100.0% | 100.0% | claude-code:opus (Opus 4.8); synthetic; first full run |
