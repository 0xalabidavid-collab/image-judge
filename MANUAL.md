# Image Judge — User Manual

Image Judge looks at an image prompt, the original image(s), and two results (A and B), and
decides which result is the correct one. You can train it on tasks where you know the answer,
test how well it does, and then let it judge new tasks on its own.

---

## 0. The simple way (start here)

Everything happens on the **Judge** page:

1. Paste the **prompt**, the **original image(s)**, **Result A** and **Result B**.
2. Click **Evaluate**. It says which result is correct.
3. Tell it if it was right:
   - **✓ Correct** — done. (If it was right but *unsure*, it asks why, and learns to be surer.)
   - **✗ Wrong — B is correct** — a box opens: write **why the other result is correct**
     (e.g. *"A misspells the word as ASLE; B spells SALE like the prompt asks"*). Click
     **Save & learn**. In about a minute it shows the lesson it learned. From then on, every task
     uses that lesson.
   - **? I'm not sure** — nothing is saved or learned.
   - **↻ Look again** — it checks the same task again from scratch (useful after it learned
     something, or when you doubt the answer).
4. If it has no answer ("Unclear"), click **A is correct** or **B is correct** and write why.

After **Save & learn** it learns **in the background** — you can paste the next task straight
away. The lesson appears under the verdict when it's ready (about a minute), and the line above
the Evaluate button shows "Learning from N correction(s)…" while it works.

The more specific your reason, the better the lesson. Every task you mark is also kept in a set
called **My tasks** with the correct answer, so you can test it on them later if you want.

**Recent evaluations** (bottom of the page): thumbnails of every task. Click the number to open
any past evaluation again — you can see the full result and change your answer there.

### How it judges
It follows your project's labelling guidelines (rubric v3): start with prompt compliance but never
stop there; weigh each flaw by how serious and how visible it is; trust the first impression;
anatomy and structure flaws usually decide; then artifacts, cohesion and style match; finally,
pick the image the requester would be happier to receive. For local edits it also gets a
**difference heat map** of each result against the original (like the Difference block), plus the
percentage of the image that changed. Each evaluation takes under a minute (4 checks at once).

### Lessons stay few and calibrated
Lessons are capped at 12, may not contradict the guidelines or each other, and may not claim one
criterion "always wins". If a correction is too specific to one task, it learns nothing from it on
purpose — piling up one-task rules made the old version worse over time.

You never have to use the rest of this manual — sections 4–9 are for checking, in bulk, how well
it has learned.

---

## 1. Starting and stopping

Open a terminal in the `image-judge` folder and run:

```bash
.venv/Scripts/python -m uvicorn app.main:app
```

Then open **http://127.0.0.1:8000** in Chrome or Edge. Leave the terminal open while you work;
press **Ctrl+C** in it to stop the app. Your tasks, results and lessons are saved and will be
there next time.

It judges using the Claude Code app on this computer (your Claude plan), so you don't need an
API key. You must be logged in to Claude Code.

---

## 2. The three pages

| Page | What it's for |
|---|---|
| **Judge** (`/`) | Enter a task. Either judge it right away, or save it into a task set. |
| **Train & Test** (`/train`) | Your task sets, the Train / Test / Judge-only runs, and the lessons it has learned. |
| **Benchmark** (`/benchmark`) | Advanced: run it on a folder of tasks on disk. You can ignore this page. |

---

## 3. Entering a task (Judge page)

A task is: **a prompt + 1 to 4 original images + Result A + Result B**.

1. **Prompt** — paste the instruction text into box 1.
2. **Images** — for each image, either:
   - copy it and press **Ctrl+V** — it goes into the box highlighted "Paste target"
     (originals first, then A, then B — the highlight moves on by itself), or
   - drag the file onto a box, or
   - click a box and pick the file.

   Click a box first if you want to paste into that specific box. Use the **×** on a picture to remove it.
3. Then do **one** of these:
   - **Save it to a task set** (for training/testing) — see section 4.
   - **Evaluate** — judge it now. You get the verdict, how sure it is, the deciding difference,
     and a requirement-by-requirement table. Click **Right** or **Wrong** afterwards to record
     whether it was correct.

**Clear all** empties the form.

---

## 4. Saving tasks into a set

A **task set** is a named group of tasks, e.g. "Train 1", "Test 1", "Batch 3".

1. Enter the task (prompt + images) as above.
2. Under **3. Save to a task set**:
   - **Task set**: pick an existing set, or **+ New set…** and type a name.
   - **Correct answer**: **A**, **B**, or **Not known yet**.
3. Click **Save task to set**.

The form clears so you can paste the next task. The set you chose stays selected. The
message under the button shows how many tasks are in the set and how many have an answer.

You can add or fix answers later on the Train & Test page (open the set, click **A** or **B**).

---

## 5. The three modes (Train & Test page)

At the top of the Train & Test page, pick a mode, a task set, and press **Start**.

### Train
*Needs: a set where you marked the correct answers.*

It judges every task, then **studies the ones it got wrong or wasn't sure about**, and writes
lessons from them. The lessons are saved as a new **lessons version** and switched on, so every
judgment after that uses them. (See section 6 for exactly how.)

### Test (score it)
*Needs: answers — either marked before the run, or added after.*

It judges every task **without seeing your answers**, then scores itself against them:
"27/30 correct". If you didn't mark answers beforehand, click **A** or **B** next to each task
in the results and the score updates as you go.

Tick **"Also run without lessons"** to judge the same tasks a second time with no lessons, so you
can see whether training helped ("Lessons fixed 3 tasks and broke 0"). This doubles the time.

### Judge only
*Needs: nothing.*

It just gives a verdict for every task. This is for real work. You can still mark answers
afterwards if you want a score.

### Other options
- **Lessons** — which lessons to use: the active ones (normal), none, or an older version.
- **Runs per task** — how many independent times it judges each task (default 4). More runs =
  more reliable but slower.
- **Learn from up to N mistakes** (Train only) — caps how many mistakes it studies (default 15).

Before you press Start, the page shows how many model calls it will make and roughly how long it
will take. While it runs you see a progress bar; you can leave the page and come back.

---

## 6. How training works

Nothing about the model itself changes — that isn't possible with this kind of tool. Instead it
works the way a new labeller keeps notes:

1. **Judge.** It judges every task in the training set using its current lessons. This gives you
   the "score before learning".
2. **Pick the mistakes.** It takes the tasks it got **wrong**, then the ones it got right but
   **wasn't sure** about. (Tasks it got right with confidence are skipped — nothing to learn.)
3. **Study each mistake.** For each one it is shown the images again, **your correct answer**, and
   **its own wrong reasoning**, and asked:
   - What did I miss? (e.g. *"Result A says ASLE, not SALE; I compared font size and never read the letters."*)
   - What general rule would have got this right — and would help on other tasks too?
     (e.g. *"When the prompt asks for exact text, read the letters one by one in both results first;
     a misspelling outweighs font size or styling."*)

   Rules must be general — never "the red car on the left".
4. **Check your answer.** If, looking at the images, it believes your marked answer is wrong or the
   task is ambiguous, it says so and **does not** learn from that task. Those tasks are flagged in
   the results so you can double-check them.
5. **Merge.** New rules are merged with the existing lessons: duplicates combined, contradictions
   resolved, at most 20 rules, most important first.
6. **Save.** The result is saved as a new lessons version and made active.

Each training round builds on the previous version, so lessons accumulate over rounds.

**Why the test set must be different tasks:** lessons are written from the training tasks, so it
will naturally do well on those. Only tasks it has never trained on tell you how good it really is.

---

## 7. Lessons & guidelines

At the bottom of the Train & Test page:

- **Versions list** — every lessons version, where it came from ("Trained on 'Train 1' (run #3)…"),
  and its lessons (click to expand).
- **Make active** — switch to an older version (e.g. if a newer one made things worse).
- **Judge without lessons** — turn lessons off.
- **Edit guidelines and lessons** — change or delete lessons by hand, and paste your **project's
  labelling guidelines** (the instructions you follow when you choose A or B). Saving creates a new
  version and makes it active. Guidelines are given to the judge with every task and take priority
  over its built-in rules — this is often the single biggest improvement you can make.

The Judge page always shows which lessons version it is using.

### Automatic training (the Lessons page)

When you mark an answer wrong or unsure, it writes a lesson from it, but **does not use it yet**. The
lesson is a *candidate*. Once 5 new corrections are waiting (`IMAGE_JUDGE_GATE_EVERY`), the candidate
and the lessons in use are both run on up to 20 answered tasks the candidate was **not learned from**
(at least 12 needed, `IMAGE_JUDGE_GATE_MIN_POOL`). The candidate is switched on only if it gets
clearly more right (at least 10% of the tasks, minimum 1 - so 2 of 20) **and** makes no more confident mistakes.
Otherwise it is thrown away and nothing changes. A test costs roughly 2 × the number of tasks × runs
judge calls; results are cached, so repeats are cheaper.

The **Lessons** page shows the lessons in use, the test status, and every version with what it added
or removed, who taught it, and its test result. From there you can **Undo the last change**, **Use** any
older version, **Throw away** a candidate, **Test now**, or switch all lessons off.

**Reasons.** Marking a result wrong now asks for a one-sentence reason (you can choose "Save without a
reason"). The Lessons page lists earlier answers that have none, so you can fill them in; **Save &
learn** also writes a lesson from it. A good reason is specific: "A still shows the old logo; B replaced
it as asked".

Set `IMAGE_JUDGE_AUTO_GATE=0` to go back to switching lessons on immediately. Running **Train** on the
Train & Test page is a deliberate action and still switches its lessons on at once.

---

## 8. Reading the results

### Status of each verdict
| Status | Meaning |
|---|---|
| **Confident** | Every run agreed, with the images shown in both orders, and none was unsure. |
| **Review** ("Leaning A/B") | Most runs agreed but not all, or one was unsure. Check it yourself. |
| **Unclear** | The runs disagreed. It gives no answer rather than guess. |

### Numbers on a Test or Train run
- **Score** — correct verdicts out of tasks with an answer. Unclear counts as wrong.
- **Accuracy when confident** — how often it was right when it said "confident". This is the number
  that matters most if you plan to trust confident verdicts without checking.
- **Confident on (%)** — how many tasks it was confident about.
- **Not sure** — how many were review or unclear (these are the ones to check by hand).

### What it learned (Train runs)
- How many mistakes it studied and how many lessons it wrote.
- The new lessons list.
- **Each mistake**: what it missed and the rules it took from it.
- A warning if it thinks some of your answers are wrong.

### Did the lessons really help? (Test with "Also run without lessons")
The comparison pairs the two runs task by task and shows how many tasks the lessons **fixed**
and **broke**, plus the **chance of a split like that by pure luck**, with a plain verdict:

| Verdict | Means |
|---|---|
| **Helped** | Chance of luck 5% or less: the lessons very likely help. |
| **Maybe** | Between 5% and 20%: could be luck — test more tasks. |
| **Can't tell** | Over 20%: normal noise. |
| **Made it worse** | Broke clearly more than it fixed: make the previous lessons version active again. |

Examples: fixed 6, broke 0 → about 1 in 64 (Helped). Fixed 6, broke 1 → about 1 in 16 (Maybe).
Fixed 3, broke 2 → high (Can't tell).

### Small numbers
With 30 test tasks, each task is about 3%. A change of one or two tasks between rounds can be luck.
More test tasks give more trustworthy numbers.

---

## 9. A recommended routine

1. **Round 1** — Save 40 tasks with answers into **Train 1** → run **Train**.
2. Save 30 *different* tasks into **Test 1** → run **Test**, tick "Also run without lessons".
   Read the failures: was it the judge, or a wrong/unclear answer on your side?
3. **Round 2** — Save 40 new tasks into **Train 2** → **Train** (it starts from round 1's lessons).
4. Save 30 new tasks into **Test 2** → **Test**. Compare with Test 1.
5. When the test scores are good enough for you, use **Judge only** for real batches. Spot-check
   the **Review** and **Unclear** ones by hand, and keep an eye on confident ones too.

Tips:
- Paste your guidelines (section 7) before the first training round.
- Make training sets varied — the kinds of tasks you'll actually see.
- If it flags one of your answers as wrong, look again; a wrong answer teaches a wrong lesson.
- If a new lessons version scores worse on a test set, make the previous version active again.

---

## 10. Time and usage

Each model call takes about 35 seconds; it runs 2 at a time.

| Run | Calls | Time (roughly) |
|---|---|---|
| Train, 40 tasks, 4 runs each | ~175 | ~25 min |
| Test, 30 tasks, 4 runs each | ~120 | ~18 min (double with "without lessons") |
| Judge one task on the Judge page | 4 | ~1 min |

This uses your Claude plan's usage. If you hit your plan's limit, the run stops and tells you why.
Results that already finished are kept, so re-running the same thing later skips them.

---

## 11. Problems

| You see | What to do |
|---|---|
| "The selected model isn't usable…" banner | Make sure Claude Code is installed and you're logged in, then restart the app. |
| "Claude usage limit reached" | Wait for your plan's limit to reset, then start the run again. |
| Run says **interrupted** | The app was closed during the run. Start it again — finished parts are reused. |
| "does not support this model; version … required" | Run `claude update` in a terminal. |
| Paste doesn't work | Click the box you want first, then Ctrl+V. Or drag the file in. |
| Page won't open | The app isn't running — start it (section 1). |

---

## 12. For advanced use

Command line (from the `image-judge` folder):

```bash
.venv/Scripts/python -m app.cli train --set "Train 1"
.venv/Scripts/python -m app.cli test --set "Test 1" --compare
.venv/Scripts/python -m app.cli judge --set "Batch 3"
```

Settings live in the `.env` file (model, runs per task, how many calls at once). Your data is in
`data/` (database and uploaded images). See `README.md` for technical details.
