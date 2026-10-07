"use strict";

const MAX_ORIGINALS = 10;
const SLOT_NAMES = { originals: "Originals", a: "Result A", b: "Result B" };
const state = { originals: [], a: null, b: null, target: "originals", targetPinned: false };

const $ = sel => document.querySelector(sel);
const zones = {
  originals: $("#zone-originals"),
  a: $("#zone-a"),
  b: $("#zone-b"),
};

// ---------- image slots ----------

function makeItem(file) {
  return { file, url: URL.createObjectURL(file), name: file.name || "pasted image" };
}

function addFiles(slot, files) {
  const images = [...files].filter(f => f.type.startsWith("image/"));
  if (!images.length) { status("That isn't an image."); return; }
  if (slot === "originals") {
    const room = MAX_ORIGINALS - state.originals.length;
    if (room <= 0) { status(`Already ${MAX_ORIGINALS} originals — remove one first.`); return; }
    if (images.length > room) status(`Only ${room} more original(s) allowed; extra images ignored.`);
    state.originals.push(...images.slice(0, room).map(makeItem));
  } else {
    if (state[slot]) URL.revokeObjectURL(state[slot].url);
    state[slot] = makeItem(images[0]);
  }
  status(`Added to ${SLOT_NAMES[slot]}.`);
  if (!state.targetPinned) state.target = nextEmptySlot();
  render();
}

function removeItem(slot, index) {
  if (slot === "originals") {
    URL.revokeObjectURL(state.originals[index].url);
    state.originals.splice(index, 1);
  } else {
    URL.revokeObjectURL(state[slot].url);
    state[slot] = null;
  }
  if (!state.targetPinned) state.target = nextEmptySlot();
  render();
}

function nextEmptySlot() {
  if (!state.originals.length) return "originals";
  if (!state.a) return "a";
  if (!state.b) return "b";
  return state.originals.length < MAX_ORIGINALS ? "originals" : "b";
}

function render() {
  for (const [slot, zone] of Object.entries(zones)) {
    const items = slot === "originals" ? state.originals : (state[slot] ? [state[slot]] : []);
    zone.querySelector(".zone-empty").hidden = items.length > 0;
    zone.querySelector(".thumbs").innerHTML = items.map((it, i) => `
      <div class="thumb">
        <img src="${it.url}" alt="${esc(SLOT_NAMES[slot])}${slot === "originals" ? " " + (i + 1) : ""}: ${esc(it.name)}">
        <button type="button" class="remove" data-slot="${slot}" data-index="${i}"
                aria-label="Remove ${esc(SLOT_NAMES[slot])}${slot === "originals" ? " " + (i + 1) : ""}">×</button>
        ${slot === "originals" ? `<div class="cap">Original ${i + 1}</div>` : ""}
      </div>`).join("");
    const isTarget = state.target === slot;
    zone.classList.toggle("target", isTarget);
    zone.querySelector(".zone-badge").textContent = isTarget ? "Paste target" : "";
  }
  $("#paste-target-name").textContent = SLOT_NAMES[state.target];
  $("#evaluate-btn").disabled = !formReady();
  const setChosen = $("#save-set").value && ($("#save-set").value !== "new" || $("#new-set-name").value.trim());
  $("#save-btn").disabled = !(formReady() && setChosen);
}

function formReady() {
  return $("#prompt").value.trim() && state.originals.length >= 1 && state.a && state.b;
}

function status(msg) { $("#form-status").textContent = msg; }

for (const [slot, zone] of Object.entries(zones)) {
  const input = zone.querySelector("input[type=file]");
  const pick = () => input.click();
  zone.addEventListener("click", e => {
    if (e.target.closest("button.remove")) return;
    state.target = slot; state.targetPinned = true; render();
    pick();
  });
  zone.addEventListener("focus", () => { state.target = slot; state.targetPinned = true; render(); });
  zone.addEventListener("keydown", e => {
    if ((e.key === "Enter" || e.key === " ") && e.target === zone) { e.preventDefault(); pick(); }
  });
  input.addEventListener("change", () => { addFiles(slot, input.files); input.value = ""; });
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("dragover");
    addFiles(slot, e.dataTransfer.files);
  });
}

document.addEventListener("click", e => {
  const btn = e.target.closest("button.remove");
  if (btn) { e.stopPropagation(); removeItem(btn.dataset.slot, Number(btn.dataset.index)); zones[btn.dataset.slot].focus(); }
});

document.addEventListener("paste", e => {
  const files = [...(e.clipboardData?.items || [])]
    .filter(i => i.kind === "file" && i.type.startsWith("image/"))
    .map(i => i.getAsFile());
  if (!files.length) return; // let normal text paste into the prompt happen
  e.preventDefault();
  // Several images pasted at once fill the slots in order.
  for (const f of files) {
    addFiles(state.target, [f]);
    state.targetPinned = false;
    state.target = nextEmptySlot();
  }
  render();
});

$("#prompt").addEventListener("input", render);

function clearTask() {
  [...state.originals, state.a, state.b].filter(Boolean).forEach(it => URL.revokeObjectURL(it.url));
  Object.assign(state, { originals: [], a: null, b: null, target: "originals", targetPinned: false });
  $("#prompt").value = "";
  $("#result").hidden = true;
  render();
}

$("#reset-btn").addEventListener("click", () => { clearTask(); status("Cleared."); });

// ---------- evaluate ----------

$("#judge-form").addEventListener("submit", async e => {
  e.preventDefault();
  if (!formReady()) return;
  const fd = new FormData();
  fd.append("prompt", $("#prompt").value);
  state.originals.forEach(it => fd.append("originals", it.file, it.name));
  fd.append("result_a", state.a.file, state.a.name);
  fd.append("result_b", state.b.file, state.b.name);
  fd.append("fresh", $("#fresh").checked ? "true" : "false");

  const btn = $("#evaluate-btn");
  btn.disabled = true; btn.textContent = "Evaluating…";
  status("Running independent judge passes. This can take a minute or two.");
  try {
    const data = await evaluateInBackground(fd);
    showResult(data);
    status("");
    loadHistory();
  } catch (err) {
    $("#result").hidden = false;
    $("#result-body").innerHTML = `<div class="banner error">${esc(err.message)}</div>`;
    status("Evaluation failed.");
  } finally {
    btn.textContent = "Evaluate"; render();
  }
});

// Starts the judgment on the server and checks on it every few seconds until it is done.
async function evaluateInBackground(formData) {
  const { job } = await fetchJSON("/api/evaluate/start", { method: "POST", body: formData });
  const started = Date.now();
  let hiccups = 0;
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 3000));
    let st;
    try {
      st = await fetchJSON(`/api/evaluate/jobs/${job}`);
      hiccups = 0;
    } catch (err) {
      if (++hiccups >= 5) throw err;  // a few dropped checks are fine; keep waiting
      continue;
    }
    if (st.status === "done") return st.result;
    if (st.status === "failed") throw new Error(st.error);
    const minutes = Math.floor((Date.now() - started) / 60000);
    if (minutes >= 30) throw new Error("This is taking over 30 minutes. If it finishes, it will appear in the history list.");
    status(minutes >= 1 ? `Still judging (${minutes} min so far). Slow servers can take a few minutes, please wait.`
                        : "Running independent judge passes. This can take a minute or two.");
  }
}

function showResult(data, opts = {}) {
  const agg = data.aggregate;
  const past = opts.fromHistory;
  const rep = data.runs.find(r => r.index === agg.representative);
  const j = rep && rep.judgment;
  const verdictText = agg.verdict ? (agg.status === "review" ? `Leaning ${agg.verdict}` : `Result ${agg.verdict}`) : "No verdict";

  let html = past ? `<p class="hint">Evaluation #${past.id} from ${new Date(past.created_at * 1000).toLocaleString()}</p>
    <p><strong>Prompt:</strong> ${esc(past.prompt)}</p>${thumbs(past.images)}
    ${past.feedback ? `<p class="hint">You marked it: ${past.feedback.verdict_correct === null ? past.feedback.true_label + " is correct"
      : past.feedback.verdict_correct ? "Correct" : "Wrong (" + past.feedback.true_label + " is correct)"}. You can change it below.</p>` : ""}` : "";
  html += `
    <div class="verdict-row">
      <div class="verdict-big">${esc(verdictText)}</div>
      <div>
        <span class="status ${esc(agg.status)}">${esc(agg.status)}</span>
        <p class="hint">${esc(STATUS_HELP[agg.status] || "")}</p>
      </div>
    </div>
    <p>${esc(agg.explanation)} Votes: A ${agg.votes.A ?? 0}, B ${agg.votes.B ?? 0}.</p>`;
  html += feedbackBlock(agg, !!past).replace("__ID__", String(data.id));

  if (data.notes && data.notes.length) {
    html += `<p class="hint">Pixel checks: ${data.notes.map(esc).join(" ")}</p>`;
  }
  if (j && j.first_impression) {
    html += `<h3>First impression</h3>
      <p class="hint">${j.edit_type ? `Edit type: <strong>${esc(j.edit_type)}</strong>. ` : ""}</p>
      <ul class="lessons-list"><li><strong>A:</strong> ${esc(j.first_impression.a)}</li>
        <li><strong>B:</strong> ${esc(j.first_impression.b)}</li></ul>`;
  }
  if (j) {
    html += `<h3>Decisive difference</h3><p class="decisive">${esc(j.decisive_difference)}</p>
      <p class="hint">${esc(j.reasoning)}</p>
      <h3>Requirements</h3>${requirementsTable(j.requirements)}`;
  }
  html += `<details><summary>All ${data.runs.length} runs</summary><div class="runs">${data.runs.map(runSummary).join("")}</div></details>`;

  $("#result-body").innerHTML = html;
  $("#result").hidden = false;
  $("#result").focus();
  wireFeedback(data.id, agg);
}

function requirementsTable(reqs) {
  const rows = (reqs || []).map(r => `
    <tr class="${r.a.status !== r.b.status ? "differs" : ""}">
      <td>${esc(r.description)} <span class="tag">${esc(r.category)}</span> <span class="tag">${esc(r.severity)}</span></td>
      <td>${pf(r.a.status)}<span class="ev">${esc(r.a.evidence)}</span></td>
      <td>${pf(r.b.status)}<span class="ev">${esc(r.b.evidence)}</span></td>
    </tr>`).join("");
  return `<div class="table-wrap"><table>
    <caption class="sr-only">Requirement checks for Result A and Result B. Highlighted rows differ.</caption>
    <thead><tr><th scope="col">Requirement</th><th scope="col">Result A</th><th scope="col">Result B</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function runSummary(r) {
  const order = r.swapped ? "B shown first" : "A shown first";
  if (r.error) return `<div class="run"><strong>Run ${r.index + 1}</strong> (${order}): error — ${esc(r.error)}</div>`;
  const j = r.judgment || {};
  return `<div class="run"><strong>Run ${r.index + 1}</strong> (${order}${r.cached ? ", cached" : ""}):
    chose <strong>${esc(r.verdict)}</strong>, ${esc(r.confidence)} confidence.
    <div class="hint">${esc(j.decisive_difference)}</div>
    <details><summary>Requirements for this run</summary>${requirementsTable(j.requirements)}</details></div>`;
}

function feedbackBlock(agg, fromHistory) {
  const v = agg.verdict;
  const other = v === "A" ? "B" : "A";
  const buttons = v
    ? `<button type="button" class="secondary" data-fb="correct">✓ Correct</button>
       <button type="button" class="secondary" data-fb="wrong">✗ Wrong — ${other} is correct</button>`
    : `<button type="button" class="secondary" data-fb="A">A is correct</button>
       <button type="button" class="secondary" data-fb="B">B is correct</button>`;
  return `<section class="decisive" style="margin:16px 0" aria-labelledby="fb-title">
      <h3 id="fb-title" style="margin-top:0">${v ? "Is this right?" : "Which one is correct?"}</h3>
      <div class="feedback" role="group" aria-labelledby="fb-title">
        ${buttons}
        <button type="button" class="secondary" data-fb="unsure">? I'm not sure</button>
        ${fromHistory ? "" : `<button type="button" class="secondary" id="look-again">↻ Look again</button>`}
      </div>
      <div id="fb-reason" hidden style="margin-top:12px">
        <label class="field" for="fb-reason-text" id="fb-reason-label"></label>
        <textarea id="fb-reason-text" placeholder="e.g. A misspells the word as ASLE; B spells SALE exactly like the prompt asks."></textarea>
        <div class="actions" style="margin-top:8px">
          <button type="button" class="primary" id="fb-save">Save &amp; learn</button>
          <button type="button" class="secondary" id="fb-skip">Save without a reason</button>
        </div>
      </div>
      <p id="fb-status" data-eval="__ID__" role="status" aria-live="polite" style="margin-bottom:0"></p>
    </section>`;
}

function wireFeedback(id, agg) {
  const v = agg.verdict;
  const unsure = agg.status !== "confident";
  let pending = null;  // {verdict_correct} or {true_label}
  let reasonRequired = false;
  const fbStatus = msg => { $("#fb-status").innerHTML = msg; };
  const press = btn => document.querySelectorAll("[data-fb]").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));

  function askReason(label, required) {
    $("#fb-reason").hidden = false;
    $("#fb-reason-label").textContent = label;
    reasonRequired = required;
    $("#fb-skip").hidden = !required;  // an optional reason needs no "skip" button
    $("#fb-save").textContent = "Save & learn";
    $("#fb-reason-text").focus();
    fbStatus("");
  }

  async function send(body) {
    $("#fb-save").disabled = true;
    $("#fb-skip").disabled = true;
    fbStatus("Saving…");
    try {
      const res = await fetchJSON(`/api/evaluations/${id}/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      $("#fb-reason").hidden = true;
      loadHistory();
      if (res.learning === "started") {
        fbStatus(`Saved as ${esc(res.true_label)}. <strong>Learning from it in the background</strong> (about a minute) —
          you can start the next task now.`);
        watchLearning(id, fbStatus);
      } else {
        fbStatus(`Saved. It was right and sure, so there was nothing new to learn.`);
      }
    } catch (err) {
      fbStatus(`Could not save: ${esc(err.message)}`);
    } finally {
      $("#fb-save").disabled = false;
      $("#fb-skip").disabled = false;
    }
  }

  document.querySelectorAll("[data-fb]").forEach(btn => btn.addEventListener("click", () => {
    press(btn);
    const choice = btn.dataset.fb;
    if (choice === "unsure") {
      $("#fb-reason").hidden = true;
      pending = null;
      fbStatus("OK, nothing saved and nothing learned. Try <strong>Look again</strong> for a fresh check, or decide later.");
      return;
    }
    if (choice === "correct") {
      pending = { verdict_correct: true };
      if (!unsure) { send(pending); return; }  // right and sure: nothing to learn
      askReason(`It was right, but not sure. Why is ${v} correct? (optional, helps it be sure next time)`, false);
      return;
    }
    const truth = choice === "wrong" ? (v === "A" ? "B" : "A") : choice;
    pending = choice === "wrong" ? { verdict_correct: false } : { true_label: truth };
    askReason(`Why is Result ${truth} the correct one? It learns from what you write here, so be specific.`, true);
  }));

  $("#fb-save").addEventListener("click", () => {
    if (!pending) return;
    const text = $("#fb-reason-text").value.trim();
    if (reasonRequired && text.length < 8) {
      fbStatus("Please write one sentence on why that result is better. It is what teaches the judge. " +
        "If you really cannot say, use <strong>Save without a reason</strong>.");
      $("#fb-reason-text").focus();
      return;
    }
    send({ ...pending, reason: text });
  });
  $("#fb-skip").addEventListener("click", () => { if (pending) send({ ...pending, reason: "" }); });

  if ($("#look-again")) $("#look-again").addEventListener("click", () => {
    if (!formReady()) { fbStatus("The images were cleared; add them again to re-check."); return; }
    $("#fresh").checked = true;
    $("#judge-form").requestSubmit();
    $("#fresh").checked = false;
  });
}

// Learning runs on the server after you save; poll until it finishes. The message goes to the
// feedback area if that evaluation is still on screen, and the lessons line is updated either way.
const learningWatch = new Set();
function watchLearning(id, show) {
  learningWatch.add(id);
  refreshKnowledgeLine();
  const tick = async () => {
    let st;
    try { st = await fetchJSON(`/api/evaluations/${id}/learning`); } catch (_) { setTimeout(tick, 4000); return; }
    if (st.status === "running") { setTimeout(tick, 3000); return; }
    learningWatch.delete(id);
    refreshKnowledgeLine();
    const onScreen = () => document.querySelector(`#fb-status[data-eval="${id}"]`);
    let msg;
    if (st.status === "done" && st.result.new_lessons.length && st.result.status === "candidate") {
      msg = `<strong>Wrote a lesson from evaluation #${id}.</strong> ${esc(st.result.what_judge_missed)}
        <ul class="lessons-list">${st.result.new_lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ul>
        <span class="hint">It is <strong>not in use yet</strong>: after a few more corrections it is tested on tasks it was not
        learned from, and switched on only if it helps. <a href="/lessons">See Lessons</a></span>`;
    } else if (st.status === "done" && st.result.new_lessons.length) {
      msg = `<strong>Learned from evaluation #${id}.</strong> ${esc(st.result.what_judge_missed)}
        <ul class="lessons-list">${st.result.new_lessons.map(l => `<li>${esc(l)}</li>`).join("")}</ul>
        <span class="hint">Now using lessons version ${st.result.knowledge_id} (${st.result.total_lessons} lessons).
        <a href="/lessons">See all lessons</a></span>`;
    } else if (st.status === "done") {
      msg = `Evaluation #${id}: no new lesson — ${esc(st.result.what_judge_missed || "its instructions already cover this case.")}`;
    } else {
      msg = `Evaluation #${id}: your answer is saved, but learning failed: ${esc(st.error || "unknown error")}`;
    }
    if (onScreen()) show(msg);
    else status(msg.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
  };
  setTimeout(tick, 3000);
}

function refreshKnowledgeLine() {
  fetchJSON("/api/config").then(c => {
    const k = c.active_knowledge;
    const busy = learningWatch.size ? ` <strong>Learning from ${learningWatch.size} correction(s)…</strong>` : "";
    $("#knowledge-line").innerHTML = (k
      ? `Judging with lessons version ${k.id} (${k.lessons} lessons). <a href="/lessons">See lessons</a>`
      : `No learned lessons yet. Mark verdicts Correct or Wrong and it learns from your corrections.`) + busy;
  }).catch(() => {});
}

// ---------- history ----------

function imgUrl(path) {
  return `/api/images/${encodeURIComponent(String(path).split(/[\\/]/).pop())}`;
}

function thumbs(images) {
  if (!images || !images.a) return "";
  return `<div class="task-thumbs">${(images.originals || []).map((p, i) =>
      `<img src="${imgUrl(p)}" alt="Original ${i + 1}" loading="lazy">`).join("")}
    <span class="lab">A</span><img src="${imgUrl(images.a)}" alt="Result A" loading="lazy">
    <span class="lab">B</span><img src="${imgUrl(images.b)}" alt="Result B" loading="lazy"></div>`;
}

let historyShown = 15;  // rows of the history list on screen; the rest load on request

async function loadHistory() {
  try {
    const data = await fetchJSON("/api/evaluations?limit=5000");
    if (!data.items.length) return;
    const s = data.feedback_stats.confident;
    const shown = Math.min(historyShown, data.items.length);
    const rows = data.items.slice(0, shown).map(ev => {
      const fb = ev.feedback;
      const fbText = !fb ? `<span class="hint">not marked yet</span>` : fb.verdict_correct === null ? `${fb.true_label} is correct` :
        fb.verdict_correct ? `<span class="ok-text">✓ Correct</span>` : `<span class="bad-text">✗ Wrong</span> (${fb.true_label})`;
      return `<tr>
        <td><button type="button" class="secondary" data-open-eval="${ev.id}" aria-label="Open evaluation ${ev.id}">#${ev.id}</button></td>
        <td>${thumbs(ev.images)}</td>
        <td>${esc(ev.prompt.slice(0, 90))}<div class="hint">${new Date(ev.created_at * 1000).toLocaleString()}</div></td>
        <td><span class="status ${esc(ev.status)}">${esc(ev.status)}</span> ${esc(ev.verdict ? "chose " + ev.verdict : "no verdict")}</td>
        <td>${fbText}</td></tr>`;
    }).join("");
    $("#history").innerHTML = `
      <p class="hint">Click a number to open that evaluation again (you can mark it there too).
        When the judge was confident and you marked it, it was right ${s.with_verdict ? `${s.correct} of ${s.with_verdict} times` : "(none marked yet)"}.</p>
      <div class="table-wrap"><table>
        <thead><tr><th scope="col">Open</th><th scope="col">Images</th><th scope="col">Prompt</th>
          <th scope="col">Judge's verdict</th><th scope="col">Your answer</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${shown < data.items.length
        ? `<p><button type="button" class="secondary" id="history-more">Show ${Math.min(15, data.items.length - shown)} more
             (${data.items.length - shown} older)</button></p>` : ""}`;
    const more = $("#history-more");
    if (more) more.addEventListener("click", () => { historyShown += 15; loadHistory(); });
    document.querySelectorAll("[data-open-eval]").forEach(b => b.addEventListener("click", () => openEvaluation(Number(b.dataset.openEval))));
  } catch (_) { /* history is non-essential */ }
}

async function openEvaluation(id) {
  try {
    const ev = await fetchJSON(`/api/evaluations/${id}`);
    showResult({ id: ev.id, ...ev.result }, { fromHistory: ev });
  } catch (err) { status(`Could not open #${id}: ${err.message}`); }
}

// ---------- save to a task set ----------

function storedSet() { try { return localStorage.getItem("ij-save-set"); } catch (_) { return null; } }
function storeSet(v) { try { localStorage.setItem("ij-save-set", v); } catch (_) { /* optional */ } }

async function loadSets(selectId) {
  try {
    const data = await fetchJSON("/api/sets");
    const want = String(selectId ?? storedSet() ?? "");
    const opts = data.items.map(s => `<option value="${s.id}" ${String(s.id) === want ? "selected" : ""}>
      ${esc(s.name)} (${s.tasks} tasks, ${s.labeled} with answer)</option>`);
    opts.unshift(`<option value="" ${data.items.length ? "" : "selected"} disabled>Choose a set…</option>`);
    opts.push(`<option value="new" ${data.items.length ? "" : "selected"}>+ New set…</option>`);
    $("#save-set").innerHTML = opts.join("");
    if (!data.items.some(s => String(s.id) === want) && data.items.length) $("#save-set").value = "";
  } catch (_) {
    $("#save-set").innerHTML = `<option value="new" selected>+ New set…</option>`;
  }
  $("#new-set-name").hidden = $("#save-set").value !== "new";
  render();
}

$("#save-set").addEventListener("change", () => {
  $("#new-set-name").hidden = $("#save-set").value !== "new";
  if ($("#save-set").value !== "new") storeSet($("#save-set").value);
  else $("#new-set-name").focus();
  render();
});
$("#new-set-name").addEventListener("input", render);

$("#save-btn").addEventListener("click", async () => {
  if (!formReady()) return;
  const fd = new FormData();
  fd.append("prompt", $("#prompt").value);
  state.originals.forEach(it => fd.append("originals", it.file, it.name));
  fd.append("result_a", state.a.file, state.a.name);
  fd.append("result_b", state.b.file, state.b.name);
  const label = document.querySelector('input[name="save-label"]:checked').value;
  fd.append("label", label);
  if ($("#save-set").value === "new") fd.append("new_set_name", $("#new-set-name").value.trim());
  else fd.append("set_id", $("#save-set").value);

  const btn = $("#save-btn");
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const res = await fetchJSON("/api/sets/tasks", { method: "POST", body: fd });
    storeSet(String(res.set.id));
    $("#new-set-name").value = "";
    await loadSets(res.set.id);
    clearTask();
    document.querySelector('input[name="save-label"][value=""]').checked = true;
    $("#save-status").textContent = `Saved to "${res.set.name}" — ${res.set.tasks} tasks, ${res.set.labeled} with answer` +
      (label ? "." : " (this one has no answer yet; add it on the Train & Test page).");
    $("#prompt").focus();
  } catch (err) {
    $("#save-status").textContent = `Could not save: ${err.message}`;
  } finally {
    btn.textContent = "Save task to set"; render();
  }
});

fetchJSON("/api/config").then(c => { $("#key-warning").hidden = c.api_key_configured; }).catch(() => {});
refreshKnowledgeLine();
loadSets();
render();
loadHistory();

// The "How this works" card stays closed once someone has hidden it.
(function rememberHowto() {
  const card = document.getElementById("howto");
  if (!card) return;
  try { if (localStorage.getItem("ij-howto-closed") === "1") card.open = false; } catch (_) { /* optional */ }
  card.addEventListener("toggle", () => {
    try { localStorage.setItem("ij-howto-closed", card.open ? "0" : "1"); } catch (_) { /* optional */ }
  });
})();
