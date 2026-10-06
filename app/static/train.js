"use strict";

const $ = sel => document.querySelector(sel);
const st = { config: null, sets: [], knowledge: [], openSet: null, pollTimer: null };
// Measured on this machine with Claude Code: about 0.6 min per task at 4 runs => ~9 s per judge call.
const SECONDS_PER_CALL = 9;

function imgUrl(path) {
  const name = String(path).split(/[\\/]/).pop();
  return `/api/images/${encodeURIComponent(name)}`;
}

function mode() { return document.querySelector('input[name="mode"]:checked').value; }

function fmtDate(t) { return new Date(t * 1000).toLocaleString(); }

// ---------- loading ----------

async function loadAll() {
  const [cfg, sets, know] = await Promise.all([
    fetchJSON("/api/config"), fetchJSON("/api/sets"), fetchJSON("/api/knowledge"),
  ]);
  st.config = cfg; st.sets = sets.items; st.knowledge = know.items;
  $("#key-warning").hidden = cfg.api_key_configured;
  renderSets();
  renderKnowledge();
  fillRunForm();
}

async function reloadSets() {
  st.sets = (await fetchJSON("/api/sets")).items;
  renderSets();
  fillRunForm(true);
}

async function reloadKnowledge() {
  st.knowledge = (await fetchJSON("/api/knowledge")).items;
  renderKnowledge();
  fillRunForm(true);
}

// ---------- run form ----------

function fillRunForm(keep) {
  const prevSet = $("#run-set").value, prevK = $("#run-knowledge").value;
  $("#run-set").innerHTML = st.sets.length
    ? st.sets.map(s => `<option value="${s.id}">${esc(s.name)} — ${s.tasks} tasks, ${s.labeled} with answer</option>`).join("")
    : `<option value="">No sets yet — create one below</option>`;
  const active = st.knowledge.find(k => k.active);
  $("#run-knowledge").innerHTML =
    `<option value="active">Active lessons${active ? ` (version ${active.id}, ${active.lessons.length} lessons)` : " (none yet)"}</option>` +
    `<option value="none">No lessons (untrained)</option>` +
    st.knowledge.map(k => `<option value="${k.id}">Version ${k.id} — ${k.lessons.length} lessons</option>`).join("");
  if (keep) {
    if ([...$("#run-set").options].some(o => o.value === prevSet)) $("#run-set").value = prevSet;
    if ([...$("#run-knowledge").options].some(o => o.value === prevK)) $("#run-knowledge").value = prevK;
  } else {
    $("#run-runs").value = st.config.defaults.runs;
    $("#run-model").value = st.config.defaults.model;
  }
  updateRunForm();
}

function updateRunForm() {
  const m = mode();
  $("#compare-wrap").hidden = m === "train";
  $("#cases-wrap").hidden = m !== "train";
  const set = st.sets.find(s => String(s.id) === $("#run-set").value);
  const runs = Number($("#run-runs").value) || 1;
  let n = 0, info = "";
  if (set) {
    n = m === "train" ? set.labeled : set.tasks;
    if (m === "train") info = set.labeled < set.tasks ? `${set.tasks - set.labeled} tasks without an answer will be skipped.` : "";
    if (m === "test" && set.labeled < set.tasks) info = `${set.tasks - set.labeled} tasks have no answer yet; mark them after the run to include them in the score.`;
  }
  $("#run-set-info").textContent = info;
  let calls = n * runs;
  if (m === "train") calls += Math.min(n, Number($("#run-cases").value) || 15) + 1;
  if (m !== "train" && $("#run-compare").checked) calls *= 2;
  $("#run-estimate").textContent = n
    ? `${calls} model calls, roughly ${Math.max(1, Math.round(calls * SECONDS_PER_CALL / 60))} min at the speed measured here.`
    : "";
  $("#run-btn").disabled = !n;
  $("#run-btn").textContent = { train: "Start training", test: "Start test", judge: "Start judging" }[m];
}

document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener("change", updateRunForm));
["#run-set", "#run-runs", "#run-compare", "#run-cases"].forEach(s => $(s).addEventListener("input", updateRunForm));

$("#run-form").addEventListener("submit", async e => {
  e.preventDefault();
  const k = $("#run-knowledge").value;
  const body = {
    mode: mode(),
    set_id: Number($("#run-set").value),
    knowledge: k === "active" || k === "none" ? k : Number(k),
    runs: Number($("#run-runs").value) || null,
    model: $("#run-model").value.trim() || null,
    compare_baseline: $("#run-compare").checked,
    max_cases: Number($("#run-cases").value) || 15,
  };
  try {
    const res = await fetchJSON("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("#run-status").textContent = `Started run #${res.id} on ${res.total} tasks.`;
    history.replaceState(null, "", `?run=${res.id}`);
    showRun(res.id, true);
  } catch (err) {
    $("#run-status").textContent = `Could not start: ${err.message}`;
  }
});

// ---------- run results ----------

async function showRun(id, focus) {
  clearTimeout(st.pollTimer);
  let run;
  try { run = await fetchJSON(`/api/benchmarks/${id}`); }
  catch (err) { $("#run-status").textContent = err.message; return; }
  $("#current").hidden = false;
  const name = { train: "Training", test: "Test", judge: "Judging", benchmark: "Benchmark" }[run.mode] || "Run";
  $("#current-title").textContent = `${name} run #${run.id} — ${run.dataset.replace(/^set: /, "")} — ${run.status}`;
  $("#current-body").innerHTML = renderRun(run);
  wireRunLabels(run);
  if (focus) $("#current").focus();
  const waiting = run.status === "running" || (run.baseline && run.baseline.status === "running");
  if (waiting) st.pollTimer = setTimeout(() => showRun(id), 3000);
  else { reloadSets(); if (run.mode === "train") reloadKnowledge(); }
}

function scoreLine(m) {
  if (!m || !m.total) return null;
  const correct = Math.round((m.overall_accuracy || 0) * m.total);
  return { correct, total: m.total, pct: correct / m.total };
}

function metric(value, label) {
  return `<div class="metric"><div class="v">${value}</div><div class="l">${esc(label)}</div></div>`;
}

function renderRun(run) {
  const c = run.config;
  let html = `<p class="hint">${esc(c.model)} · ${c.runs} runs per task · ${
    c.knowledge_id ? `lessons version ${c.knowledge_id} (${c.lessons} lessons)` : "no lessons"}</p>`;
  if (run.status === "running") {
    const learning = run.mode === "train" && run.completed >= run.total;
    html += learning
      ? `<p><strong>Judging done. Now studying the mistakes and writing lessons…</strong></p>`
      : `<label class="field" for="prog">Judged ${run.completed} of ${run.total} tasks</label>
         <progress id="prog" max="${run.total}" value="${run.completed}"></progress>`;
  }
  const m = run.metrics;
  if (m && m.error) html += `<div class="banner error">Stopped: ${esc(m.error)}</div>`;
  if (run.status === "interrupted") html += `<div class="banner warn">This run was interrupted (the server restarted). Start it again; finished judgments are reused from the cache.</div>`;

  const sc = scoreLine(m);
  if (m && m.tasks !== undefined) {
    if (sc) {
      html += `<div class="metrics" style="margin-top:12px">
        ${metric(`${sc.correct}/${sc.total}`, `Score: correct verdicts (${pct(sc.pct)}). Unclear counts as wrong.`)}
        ${metric(pct(m.confident_accuracy), `Accuracy when confident (${m.confident_correct}/${m.counts.confident})`)}
        ${metric(pct(m.coverage), "Confident on (% of tasks)")}
        ${metric(String(m.counts.review + m.counts.unclear), "Not sure (review or unclear)")}
      </div>`;
      if (run.mode === "train") html += `<p class="hint">This is the score <em>before</em> learning, using the lessons it had when the run started.</p>`;
    } else if (run.status !== "running") {
      html += `<p>No answers marked yet. Mark the correct answer for each task below to see a score.</p>`;
    }
    if (m.unlabeled && sc) html += `<p class="hint">${m.unlabeled} tasks have no answer yet and aren't in the score.</p>`;
  }

  if (run.baseline) html += renderBaseline(run);
  if (run.mode === "train" && m && m.training) html += renderTraining(run);
  html += renderItems(run);
  return html;
}

function renderBaseline(run) {
  const a = scoreLine(run.metrics), b = scoreLine(run.baseline.metrics);
  let html = `<h3>With lessons vs without</h3>`;
  if (run.baseline.status === "running") {
    return html + `<p class="hint">The untrained comparison run is going: ${run.baseline.completed} of ${run.baseline.total}.</p>`;
  }
  if (!a || !b) return html + `<p class="hint">Mark the answers to compare.</p>`;
  html += `<div class="compare">
    ${metric(`${a.correct}/${a.total}`, `With lessons (${pct(a.pct)})`)}
    ${metric(`${b.correct}/${b.total}`, `Without lessons (${pct(b.pct)})`)}
  </div>`;
  const cmp = run.comparison;
  if (!cmp) return html;
  const cls = { helped: "confident", hurt: "error", maybe: "review", noise: "unclear", no_change: "unclear" }[cmp.verdict];
  const label = { helped: "Helped", hurt: "Made it worse", maybe: "Maybe", noise: "Can't tell", no_change: "No change" }[cmp.verdict];
  const ids = list => list.map(t => "#" + Number(t)).join(", ");
  const odds = cmp.fixed.length + cmp.broke.length
    ? ` Chance of a split like this by pure luck: <strong>${cmp.luck_chance >= 0.5 ? "high" : `about 1 in ${cmp.luck_one_in}`}</strong> (${pct(cmp.luck_chance)}).`
    : "";
  html += `<p><span class="status ${cls}">${label}</span> Lessons fixed <strong>${cmp.fixed.length}</strong> task(s)
      and broke <strong>${cmp.broke.length}</strong>.${odds}</p>
    <p>${esc(cmp.summary)}</p>
    ${cmp.fixed.length ? `<p class="hint">Fixed: ${ids(cmp.fixed)}</p>` : ""}
    ${cmp.broke.length ? `<p class="hint">Broke: ${ids(cmp.broke)}</p>` : ""}
    <p class="hint">Only tasks where the two runs disagree count. Unclear counts as not right.</p>`;
  return html;
}

function renderTraining(run) {
  const t = run.metrics.training;
  let html = `<h3>What it learned</h3>`;
  if (!t.cases_reviewed) return html + `<p>It got every task right with confidence — nothing to learn from this set.</p>`;
  html += `<p>Studied ${t.cases_reviewed} task(s) it got wrong or wasn't sure about and wrote ${t.new_lessons} new lesson(s).`;
  html += t.knowledge_after && t.knowledge_after !== t.knowledge_before
    ? ` Saved as <strong>lessons version ${t.knowledge_after}</strong> (${t.lessons_after} lessons), now active.</p>`
    : `</p>`;
  if (t.changes) html += `<p class="hint">${esc(t.changes)}</p>`;
  if (t.flagged_labels && t.flagged_labels.length) {
    html += `<div class="banner warn">It thinks the marked answer may be wrong for task(s) ${t.flagged_labels.map(x => "#" + Number(x)).join(", ")}.
      Those weren't used for lessons; check them below.</div>`;
  }
  if (run.learned_knowledge) {
    html += `<details open><summary>All lessons in version ${run.learned_knowledge.id}</summary>
      <ol class="lessons-list">${run.learned_knowledge.lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ol></details>`;
  }
  html += `<details><summary>Each mistake and what it learned from it</summary>${t.cases.map(cs => `
    <div class="run" style="margin-top:8px"><strong>Task #${Number(cs.task_id)}</strong> — correct ${esc(cs.label)}, it said ${esc(cs.verdict || "nothing")} (${esc(cs.status)})
      <div class="hint">${esc(cs.prompt.slice(0, 160))}</div>
      <p><strong>What it missed:</strong> ${esc(cs.what_judge_missed)}</p>
      ${cs.label_seems_wrong ? `<p class="bad-text">Thinks the marked answer is wrong; no lesson taken.</p>` :
        `<ul class="lessons-list">${cs.lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ul>`}
    </div>`).join("")}</details>`;
  if (t.case_errors && t.case_errors.length) html += `<p class="hint">${t.case_errors.length} lesson call(s) failed: ${esc(t.case_errors[0])}</p>`;
  return html;
}

function renderItems(run) {
  const items = run.items || [];
  if (!items.length) return "";
  const canLabel = run.set_id != null;
  const rows = items.map(i => {
    const verdict = i.verdict ? `${i.status === "review" ? "leaning " : ""}${esc(i.verdict)}` : "no verdict";
    const mark = i.correct === true ? `<span class="ok-text">✓</span>` : i.correct === false ? `<span class="bad-text">✗</span>` :
      (i.label ? `<span class="bad-text">✗</span>` : "—");
    const picker = canLabel && i.set_task_id ? `<span class="label-pick" role="group" aria-label="Correct answer for task ${Number(i.task_id)}">
      ${["A", "B"].map(l => `<button type="button" data-task="${i.set_task_id}" data-label="${l}" aria-pressed="${i.label === l}">${l}</button>`).join("")}
      </span>` : esc(i.label || "—");
    return `<tr><td>#${Number(i.task_id) || esc(i.task_id)}</td><td>${esc(i.prompt.slice(0, 90))}</td>
      <td><span class="status ${esc(i.status)}">${esc(i.status)}</span> ${verdict}
        ${i.decisive_difference ? `<span class="ev">${esc(i.decisive_difference.slice(0, 200))}</span>` : ""}</td>
      <td>${picker}</td><td>${mark}</td></tr>`;
  }).join("");
  return `<h3>Tasks</h3>
    <p class="hint">${canLabel ? "Click A or B to mark the correct answer; the score updates." : ""}</p>
    <div class="table-wrap"><table>
      <thead><tr><th scope="col">Task</th><th scope="col">Prompt</th><th scope="col">Verdict</th>
        <th scope="col">Correct answer</th><th scope="col">Right?</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

function wireRunLabels(run) {
  document.querySelectorAll("#current-body [data-task]").forEach(btn => btn.addEventListener("click", async () => {
    const pressed = btn.getAttribute("aria-pressed") === "true";
    try {
      await fetchJSON(`/api/tasks/${btn.dataset.task}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: pressed ? null : btn.dataset.label }),
      });
      showRun(run.id);
      if (st.openSet) openSet(st.openSet, false);
    } catch (err) { $("#run-status").textContent = `Could not save: ${err.message}`; }
  }));
}

// ---------- sets ----------

function renderSets() {
  if (!st.sets.length) { $("#sets-list").innerHTML = `<p class="hint">No sets yet.</p>`; return; }
  $("#sets-list").innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th scope="col">Set</th><th scope="col">Tasks</th><th scope="col">With answer</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
    <tbody>${st.sets.map(s => `<tr><td>${esc(s.name)}</td><td>${s.tasks}</td><td>${s.labeled}</td>
      <td><button type="button" class="secondary" data-open="${s.id}">Open</button></td></tr>`).join("")}</tbody></table></div>`;
  document.querySelectorAll("[data-open]").forEach(b => b.addEventListener("click", () => openSet(Number(b.dataset.open), true)));
}

$("#new-set-form").addEventListener("submit", async e => {
  e.preventDefault();
  const name = $("#new-set").value.trim();
  if (!name) return;
  try {
    const res = await fetchJSON("/api/sets", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    $("#new-set").value = "";
    await reloadSets();
    openSet(res.id, true);
  } catch (err) { alert(err.message); }
});

async function openSet(id, focus) {
  st.openSet = id;
  const s = await fetchJSON(`/api/sets/${id}`);
  $("#set-detail").hidden = false;
  $("#set-title").textContent = `Set: ${s.name}`;
  const rows = s.tasks.map(t => `<tr>
    <td>#${t.id}</td>
    <td><div class="task-thumbs">${t.images.originals.map((p, i) => `<img src="${imgUrl(p)}" alt="Original ${i + 1}" loading="lazy">`).join("")}
      <span class="lab">A</span><img src="${imgUrl(t.images.a)}" alt="Result A" loading="lazy">
      <span class="lab">B</span><img src="${imgUrl(t.images.b)}" alt="Result B" loading="lazy"></div></td>
    <td>${esc(t.prompt.slice(0, 120))}</td>
    <td><span class="label-pick" role="group" aria-label="Correct answer for task ${t.id}">
      ${["A", "B"].map(l => `<button type="button" data-settask="${t.id}" data-label="${l}" aria-pressed="${t.label === l}">${l}</button>`).join("")}
    </span></td>
    <td><button type="button" class="secondary" data-del="${t.id}" aria-label="Delete task ${t.id}">Delete</button></td></tr>`).join("");
  const runs = s.runs.filter(r => !r.baseline_of).reverse().map(r => {
    const sc = scoreLine(r.metrics);
    return `<li><a href="?run=${r.id}" data-run="${r.id}">#${r.id} ${esc(r.mode)}</a> — ${esc(r.status)}${sc ? `, ${sc.correct}/${sc.total}` : ""} · ${fmtDate(r.created_at)}</li>`;
  }).join("");
  $("#set-body").innerHTML = `
    ${s.tasks.length ? `<div class="table-wrap"><table>
      <thead><tr><th scope="col">Task</th><th scope="col">Images</th><th scope="col">Prompt</th>
        <th scope="col">Correct answer</th><th scope="col"><span class="sr-only">Delete</span></th></tr></thead>
      <tbody>${rows}</tbody></table></div>` : `<p class="hint">No tasks yet. Add them from the <a href="/">Judge page</a>.</p>`}
    ${runs ? `<h3>Runs on this set</h3><ul>${runs}</ul>` : ""}
    <details style="margin-top:12px"><summary>Import tasks from a folder or CSV on this computer</summary>
      <form id="import-form" class="actions" style="margin-top:8px">
        <label class="sr-only" for="import-path">Folder or CSV path</label>
        <input type="text" id="import-path" placeholder="C:\\path\\to\\dataset" style="max-width:420px">
        <button type="submit" class="secondary">Import</button>
        <span class="hint" id="import-status" role="status"></span>
      </form>
      <p class="hint">Same format as the Benchmark page (see README): a folder with tasks.csv, or one folder per task.</p>
    </details>
    <div class="actions" style="margin-top:16px">
      <button type="button" class="secondary" id="delete-set">Delete this set</button>
    </div>`;
  document.querySelectorAll("[data-settask]").forEach(btn => btn.addEventListener("click", async () => {
    const pressed = btn.getAttribute("aria-pressed") === "true";
    await fetchJSON(`/api/tasks/${btn.dataset.settask}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: pressed ? null : btn.dataset.label }),
    });
    openSet(id, false); reloadSets();
  }));
  document.querySelectorAll("[data-del]").forEach(btn => btn.addEventListener("click", async () => {
    if (!confirm(`Delete task #${btn.dataset.del} from this set?`)) return;
    await fetchJSON(`/api/tasks/${btn.dataset.del}`, { method: "DELETE" });
    openSet(id, false); reloadSets();
  }));
  document.querySelectorAll("#set-body [data-run]").forEach(a => a.addEventListener("click", e => {
    e.preventDefault(); history.replaceState(null, "", `?run=${a.dataset.run}`); showRun(Number(a.dataset.run), true);
  }));
  $("#import-form").addEventListener("submit", async e => {
    e.preventDefault();
    try {
      const res = await fetchJSON(`/api/sets/${id}/import`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: $("#import-path").value.trim() }),
      });
      $("#import-status").textContent = `Imported ${res.imported} tasks.`;
      openSet(id, false); reloadSets();
    } catch (err) { $("#import-status").textContent = err.message; }
  });
  $("#delete-set").addEventListener("click", async () => {
    if (!confirm(`Delete the set "${s.name}" and its ${s.tasks.length} tasks? Past runs stay.`)) return;
    await fetchJSON(`/api/sets/${id}`, { method: "DELETE" });
    $("#set-detail").hidden = true; st.openSet = null; reloadSets();
  });
  if (focus) $("#set-detail").focus();
}

// ---------- knowledge ----------

function renderKnowledge() {
  const active = st.knowledge.find(k => k.active);
  let html = `<p>${active ? `Active: <strong>version ${active.id}</strong> — ${active.lessons.length} lessons${active.guidelines ? " + your guidelines" : ""}.`
    : "No lessons active: the judge uses only its built-in rubric."}</p>`;
  if (st.knowledge.length) {
    html += `<div class="table-wrap"><table>
      <thead><tr><th scope="col">Version</th><th scope="col">Where it came from</th><th scope="col">Lessons</th><th scope="col"><span class="sr-only">Actions</span></th></tr></thead>
      <tbody>${st.knowledge.map(k => `<tr>
        <td>${k.id}${k.active ? ` <span class="status confident">active</span>` : ""}<div class="hint">${fmtDate(k.created_at)}</div></td>
        <td>${esc(k.source)}${k.guidelines ? `<div class="hint">has guidelines</div>` : ""}</td>
        <td><details><summary>${k.lessons.length} lessons</summary><ol class="lessons-list">${k.lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ol>
          ${k.guidelines ? `<p class="hint"><strong>Guidelines:</strong> ${esc(k.guidelines)}</p>` : ""}</details></td>
        <td>${k.active ? "" : `<button type="button" class="secondary" data-activate="${k.id}">Make active</button>`}</td>
      </tr>`).join("")}</tbody></table></div>`;
    if (active) html += `<div class="actions" style="margin-top:8px"><button type="button" class="secondary" data-activate="none">Judge without lessons</button></div>`;
  }
  $("#knowledge-list").innerHTML = html;
  document.querySelectorAll("[data-activate]").forEach(b => b.addEventListener("click", async () => {
    const id = b.dataset.activate === "none" ? null : Number(b.dataset.activate);
    await fetchJSON("/api/knowledge/activate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) });
    reloadKnowledge();
  }));
  if (!$("#knowledge-edit").open) {
    $("#k-guidelines").value = active ? active.guidelines : "";
    $("#k-lessons").value = active ? active.lessons.join("\n") : "";
  }
}

$("#knowledge-form").addEventListener("submit", async e => {
  e.preventDefault();
  const active = st.knowledge.find(k => k.active);
  try {
    const res = await fetchJSON("/api/knowledge", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ guidelines: $("#k-guidelines").value, lessons: $("#k-lessons").value.split("\n"),
                             parent_id: active ? active.id : null }),
    });
    $("#k-status").textContent = `Saved as version ${res.id} and made active.`;
    $("#knowledge-edit").open = false;
    reloadKnowledge();
  } catch (err) { $("#k-status").textContent = `Could not save: ${err.message}`; }
});

loadAll().then(() => {
  const id = new URLSearchParams(location.search).get("run");
  if (id) showRun(Number(id), false);
}).catch(err => { $("#run-status").textContent = err.message; });
