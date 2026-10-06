"use strict";

const $ = sel => document.querySelector(sel);
const state = { items: [], gate: null, reasons: null, timer: null };

const POST = { method: "POST", headers: { "Content-Type": "application/json" } };

function when(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function showError(err) {
  const box = $("#error");
  box.textContent = err ? (err.message || String(err)) : "";
  box.hidden = !err;
}

function imgUrl(path) {
  return `/api/images/${encodeURIComponent(String(path).split(/[\\/]/).pop())}`;
}

function statusBadge(k) {
  if (k.active) return `<span class="badge active">In use</span>`;
  const label = { candidate: "Waiting for a test", rejected: "Failed its test", superseded: "Replaced by a newer one",
    accepted: "Older version" }[k.status] || k.status;
  const cls = k.status === "candidate" ? "candidate" : k.status === "rejected" ? "rejected" : "muted";
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function gateLine(g) {
  if (!g) return "";
  if (g.decision === "waiting") return `<p class="gate-line">⏳ ${esc(g.reason)}</p>`;
  const b = g.baseline, c = g.candidate;
  const icon = g.decision === "accepted" ? "✓" : "✗";
  return `<p class="gate-line">${icon} Tested on ${g.pool} tasks it was not learned from:
    ${c.right} right with these lessons vs ${b.right} with the ones before; confident mistakes ${c.confident_wrong} vs
    ${b.confident_wrong}. ${esc(g.reason)}</p>`;
}

function versionHtml(k) {
  const changes = [
    ...k.added.map(l => `<li class="added">${esc(l)}</li>`),
    ...k.removed.map(l => `<li class="removed">${esc(l)}</li>`),
  ].join("");
  const by = k.taught_by.length ? ` · taught by ${esc(k.taught_by.join(", "))}` : "";
  const actions = [];
  if (!k.active) actions.push(`<button type="button" class="secondary" data-act="use" data-id="${k.id}">Use this version</button>`);
  if (!k.active && k.status === "candidate") {
    actions.push(`<button type="button" class="secondary" data-act="reject" data-id="${k.id}">Throw away</button>`);
  }
  return `<article class="version">
    <header><strong>Version ${k.id}</strong> ${statusBadge(k)}
      <span class="meta">${esc(when(k.created_at))} · ${k.lessons.length} lessons${by}</span></header>
    <p class="meta tight">${esc(k.source)}${k.new_tasks ? ` · learned from ${k.new_tasks} new answer${k.new_tasks === 1 ? "" : "s"}` : ""}</p>
    ${changes ? `<ul>${changes}</ul>` : `<p class="meta tight">No change to the lessons.</p>`}
    ${gateLine(k.gate_result)}
    ${actions.length ? `<div class="row-actions">${actions.join("")}</div>` : ""}
  </article>`;
}

function renderNow() {
  const active = state.items.find(k => k.active);
  $("#now").innerHTML = active
    ? `Version <strong>${active.id}</strong> — ${active.lessons.length} lessons.
       <details><summary>Show the lessons</summary><ol class="lessons-list">${active.lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ol></details>`
    : "No lessons are in use. The judge is using only its built-in instructions.";
  $("#undo").disabled = !active || !active.parent_id;
  $("#off").disabled = !active;
}

function renderGate() {
  const g = state.gate;
  const cand = state.items.find(k => k.status === "candidate");
  $("#gate-explain").textContent = g.enabled
    ? `Every correction writes new lessons, but they are not used yet. After ${g.every} new corrections they are tested on up to
       ${g.max_tasks} answered tasks they were not learned from (at least ${g.min_pool} needed), and switched on only if they get clearly
       more right without more confident mistakes. Otherwise they are thrown away and nothing changes.`
    : "Automatic testing is off: every correction switches its lessons on straight away. Set IMAGE_JUDGE_AUTO_GATE=1 to turn it on.";
  const s = g.state;
  let msg = "Idle.";
  if (s.status === "running") msg = `⏳ Testing version ${s.knowledge_id} on tasks it was not learned from… this takes a few minutes.`;
  else if (s.status === "failed") msg = `The last test failed: ${esc(s.error || "unknown error")}`;
  else if (s.status === "done" && s.result) {
    msg = s.result.decision === "accepted" ? `✓ Version ${s.knowledge_id} passed and is now in use.`
      : s.result.decision === "rejected" ? `✗ Version ${s.knowledge_id} did not pass and was thrown away.`
      : `⏳ ${esc(s.result.reason)}`;
  } else if (cand) msg = `Version ${cand.id} is waiting: ${cand.new_tasks} new correction${cand.new_tasks === 1 ? "" : "s"} so far.`;
  $("#gate-status").innerHTML = msg;
  $("#test-now").hidden = !cand || s.status === "running";
  return s.status === "running";
}

function renderReasons(missing) {
  const r = state.reasons;
  $("#reasons-summary").innerHTML = `<strong>${r.with_reason}</strong> of ${r.answers} answers have a reason written.`;
  $("#missing").innerHTML = missing.length ? missing.map(a => `<div class="answer" data-id="${a.id}">
      <div><strong>${esc(a.prompt.slice(0, 160))}</strong></div>
      <div class="meta">The judge said ${esc(a.verdict || "nothing")} (${esc(a.status)}); the right answer is <strong>${esc(a.true_label)}</strong>.
        ${a.labelled_by ? `Marked by ${esc(a.labelled_by)}.` : ""}</div>
      <div class="thumbs">
        <figure><img src="${imgUrl(a.images.a)}" alt="Result A" loading="lazy">A</figure>
        <figure><img src="${imgUrl(a.images.b)}" alt="Result B" loading="lazy">B</figure>
      </div>
      <label class="field" for="r${a.id}">Why is Result ${esc(a.true_label)} the better response to the request?</label>
      <textarea id="r${a.id}" placeholder="One specific sentence, e.g. A still shows the old logo; B replaced it as asked."></textarea>
      <div class="row-actions">
        <button type="button" class="primary" data-act="learn" data-id="${a.id}">Save &amp; learn</button>
        <button type="button" class="secondary" data-act="save" data-id="${a.id}">Save only</button>
        <span class="meta" role="status" data-note="${a.id}"></span>
      </div></div>`).join("")
    : `<p>Nothing missing. Every wrong or unsure answer has a reason.</p>`;
}

async function load() {
  try {
    const [know, missing] = await Promise.all([fetchJSON("/api/knowledge"), fetchJSON("/api/answers/missing-reasons?limit=30")]);
    state.items = know.items; state.gate = know.gate; state.reasons = know.reasons;
    showError(null);
    renderNow();
    const running = renderGate();
    renderReasons(missing.items);
    $("#history").innerHTML = state.items.length ? state.items.map(versionHtml).join("")
      : `<p>No lessons yet. They appear here as you mark answers wrong or unsure.</p>`;
    clearTimeout(state.timer);
    if (running) state.timer = setTimeout(load, 5000);
  } catch (err) { showError(err); }
}

async function act(path, okMessage) {
  try { await fetchJSON(path, POST); showError(null); } catch (err) { showError(err); }
  await // Labellers can read everything here and add reasons, but only the owner can change which lessons are used.
fetch("/api/me").then(r => (r.ok ? r.json() : null)).then(me => {
  if (me && me.owner === false) {
    document.body.classList.add("read-only");
    for (const id of ["#undo", "#off", "#test-now"]) $(id).remove();
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Only the project owner can switch lessons on or off. You can add reasons below.";
    $("#now").after(note);
  }
}).catch(() => {});

load();
}

document.addEventListener("click", async e => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const id = btn.dataset.id;
  switch (btn.dataset.act) {
    case "use":
      if (confirm(`Switch the judge to version ${id}? It skips the test.`)) {
        try { await fetchJSON("/api/knowledge/activate", { ...POST, body: JSON.stringify({ id: Number(id) }) }); } catch (err) { showError(err); }
        await load();
      }
      break;
    case "reject": await act(`/api/knowledge/${id}/reject`); break;
    case "learn": case "save": {
      const text = $(`#r${id}`).value.trim();
      const note = $(`[data-note="${id}"]`);
      if (text.length < 8) { note.textContent = "Write a short sentence first."; return; }
      btn.disabled = true; note.textContent = "Saving…";
      try {
        await fetchJSON(`/api/evaluations/${id}/reason`, { ...POST, body: JSON.stringify({ reason: text, learn: btn.dataset.act === "learn" }) });
        await load();
      } catch (err) { note.textContent = err.message; btn.disabled = false; }
      break;
    }
  }
});

$("#undo").addEventListener("click", async () => {
  if (confirm("Go back to the previous version of the lessons?")) await act("/api/knowledge/undo");
});
$("#off").addEventListener("click", async () => {
  if (!confirm("Judge with no lessons at all? You can switch a version back on from the history.")) return;
  try { await fetchJSON("/api/knowledge/activate", { ...POST, body: JSON.stringify({ id: null }) }); } catch (err) { showError(err); }
  await load();
});
$("#test-now").addEventListener("click", async () => { await act("/api/knowledge/gate"); });

load();
