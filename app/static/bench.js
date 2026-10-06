"use strict";

const $ = sel => document.querySelector(sel);
let pollTimer = null;

async function init() {
  const cfg = await fetchJSON("/api/config");
  $("#key-warning").hidden = cfg.api_key_configured;
  $("#rubric").innerHTML = cfg.rubric_versions.map(v => `<option ${v === cfg.defaults.rubric_version ? "selected" : ""}>${esc(v)}</option>`).join("");
  $("#runs").value = cfg.defaults.runs;
  $("#effort").value = cfg.defaults.effort;
  $("#model").value = cfg.defaults.model;
  $("#start-btn").disabled = false; // only once defaults are in the form
  await loadRuns();
  const id = new URLSearchParams(location.search).get("run");
  if (id) showRun(Number(id));
}

$("#bench-form").addEventListener("submit", async e => {
  e.preventDefault();
  const body = {
    dataset: $("#dataset").value.trim(),
    runs: Number($("#runs").value) || null,
    rubric_version: $("#rubric").value || null,
    effort: $("#effort").value || null,
    model: $("#model").value.trim() || null,
    limit: Number($("#limit").value) || null,
    use_cache: !$("#no-cache").checked,
    note: $("#note").value,
  };
  try {
    const res = await fetchJSON("/api/benchmarks", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("#bench-status").textContent = `Started run #${res.id} on ${res.total} tasks.`;
    history.replaceState(null, "", `?run=${res.id}`);
    showRun(res.id);
    loadRuns();
  } catch (err) {
    $("#bench-status").textContent = `Could not start: ${err.message}`;
  }
});

async function loadRuns() {
  const data = await fetchJSON("/api/benchmarks");
  if (!data.items.length) return;
  const rows = data.items.map(r => {
    const m = r.metrics;
    const c = r.config;
    return `<tr>
      <td><a href="?run=${r.id}" data-run="${r.id}">#${r.id}</a></td>
      <td>${esc(r.status)}${r.status === "running" ? ` ${r.completed}/${r.total}` : ""}</td>
      <td>${esc(c.model)} · ${esc(c.rubric_version)} · ${c.runs} runs · ${esc(c.effort)}</td>
      <td>${m ? `${pct(m.confident_accuracy)} <span class="hint">(≥${pct(m.confident_accuracy_lower95)})</span>` : "—"}</td>
      <td>${m ? pct(m.coverage) : "—"}</td>
      <td>${m ? pct(m.overall_accuracy) : "—"}</td>
      <td>${m ? m.total : r.total}</td>
      <td>${esc(r.note || "")}</td>
    </tr>`;
  }).join("");
  $("#runs-list").innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th scope="col">Run</th><th scope="col">Status</th><th scope="col">Config</th>
      <th scope="col">Confident acc. (95% low)</th><th scope="col">Coverage</th><th scope="col">Overall</th>
      <th scope="col">Tasks</th><th scope="col">Note</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
  document.querySelectorAll("[data-run]").forEach(a => a.addEventListener("click", e => {
    e.preventDefault();
    history.replaceState(null, "", `?run=${a.dataset.run}`);
    showRun(Number(a.dataset.run));
  }));
}

async function showRun(id) {
  clearTimeout(pollTimer);
  let run;
  try { run = await fetchJSON(`/api/benchmarks/${id}`); }
  catch (err) { $("#bench-status").textContent = err.message; return; }
  $("#current").hidden = false;
  $("#current-title").textContent = `Run #${run.id} — ${run.status}`;
  $("#current-body").innerHTML = renderRun(run);
  if (run.status === "running") {
    pollTimer = setTimeout(() => showRun(id), 3000);
  } else {
    loadRuns();
  }
}

function metric(value, label) {
  return `<div class="metric"><div class="v">${value}</div><div class="l">${esc(label)}</div></div>`;
}

function renderRun(run) {
  const c = run.config;
  let html = `<p class="hint">Dataset <code>${esc(run.dataset)}</code> · ${esc(c.model)} · rubric ${esc(c.rubric_version)} · ${c.runs} runs · effort ${esc(c.effort)}</p>`;
  if (run.status === "running") {
    html += `<label class="field" for="prog">Progress: ${run.completed} of ${run.total}</label>
      <progress id="prog" max="${run.total}" value="${run.completed}"></progress>`;
  }
  const m = run.metrics;
  if (m) {
    html += `<div class="metrics" style="margin-top:12px">
      ${metric(pct(m.confident_accuracy), `Accuracy on confident (${m.confident_correct}/${m.counts.confident})`)}
      ${metric(pct(m.confident_accuracy_lower95), "95% lower bound, confident")}
      ${metric(pct(m.coverage), "Coverage (% confident)")}
      ${metric(pct(m.overall_accuracy), "Overall accuracy")}
      ${metric(pct(m.answered_accuracy), "Accuracy when answered")}
      ${metric(pct(m.single_run_accuracy), "Single-run accuracy")}
    </div>
    <p class="hint">Counts — confident ${m.counts.confident}, review ${m.counts.review}, unclear ${m.counts.unclear}, error ${m.counts.error}.
      Single-run accuracy with A first ${pct(m.single_run_accuracy_unswapped)}, with B first ${pct(m.single_run_accuracy_swapped)} (a large gap means position bias).</p>
    <h3>If we used fewer runs</h3>
    <div class="table-wrap"><table>
      <thead><tr><th scope="col">Runs</th><th scope="col">Coverage</th><th scope="col">Confident accuracy</th><th scope="col">Confident &amp; wrong</th><th scope="col">Overall</th></tr></thead>
      <tbody>${m.run_count_curve.map(r => `<tr><td>${r.runs}</td><td>${pct(r.coverage)}</td><td>${pct(r.confident_accuracy)}</td><td>${r.confident_wrong}</td><td>${pct(r.overall_accuracy)}</td></tr>`).join("")}</tbody>
    </table></div>`;
  }
  const items = run.items || [];
  const order = { confident: 0, review: 1, unclear: 2, error: 3 };
  const fails = items.filter(i => !i.correct).sort((a, b) => order[a.status] - order[b.status]);
  html += `<h3>Failures (${fails.length}${run.status === "running" ? " so far" : ""}) — confident-and-wrong first</h3>`;
  if (!fails.length) html += `<p class="hint">None.</p>`;
  html += fails.map(f => `
    <details class="run">
      <summary><span class="status ${esc(f.status)}">${esc(f.status)}</span>
        ${esc(f.task_id)} — truth ${esc(f.label)}, verdict ${esc(f.verdict || "none")}, votes A${f.votes.A ?? 0}/B${f.votes.B ?? 0}</summary>
      <p><strong>Prompt:</strong> ${esc(f.prompt)}</p>
      <p><strong>Model's decisive difference:</strong> ${esc(f.decisive_difference || "—")}</p>
      <p class="hint">${esc(f.reasoning || f.explanation)}</p>
      ${f.runs.map(r => `<div class="hint">Run ${r.index + 1} (${r.swapped ? "B first" : "A first"}):
        ${r.error ? "error — " + esc(r.error) : `${esc(r.verdict)} / ${esc(r.confidence)} — ${esc(r.decisive_difference)}`}</div>`).join("")}
    </details>`).join("");
  return html;
}

init().catch(err => { $("#bench-status").textContent = err.message; });
