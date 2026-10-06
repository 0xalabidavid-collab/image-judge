"use strict";

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function pct(x) {
  return x === null || x === undefined ? "n/a" : (x * 100).toFixed(1) + "%";
}

// Hosted app: an expired session sends the person back to the sign-in page.
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (res.status === 401 && !location.pathname.startsWith("/login")) {
    location.href = "/login";
    throw new Error("Please sign in again.");
  }
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) {
    const detail = body && body.detail;
    const msg = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map(d => d.msg).join("; ")
      : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return body;
}

const STATUS_HELP = {
  confident: "Every run agreed, in both image orders, with no low-confidence run.",
  review: "Most runs agree, but not all — treat as a leaning and check it.",
  unclear: "The runs disagreed. No verdict is given rather than guessing.",
  error: "The judge could not run.",
};

function pf(status) {
  const label = { pass: "Pass", fail: "Fail", unsure: "Unsure" }[status] || esc(status);
  const icon = { pass: "✓", fail: "✗", unsure: "?" }[status] || "";
  return `<span class="pf ${esc(status)}">${icon} ${label}</span>`;
}

// Sign-out link, shown only when the hosted app has login on (/api/me answers; it is 404 locally).
(async function addSignOut() {
  const nav = document.querySelector("header.top nav");
  if (!nav) return;
  try {
    const res = await fetch("/api/me");
    if (!res.ok) return;
    const me = await res.json();
    window.ijMe = me;
    if (me.owner === false) {  // labellers do not see the owner-only pages
      nav.querySelectorAll('a[href="/train"], a[href="/benchmark"]').forEach(a => a.remove());
    }
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = `Sign out (${me.email})`;
    link.style.marginLeft = "auto";
    link.addEventListener("click", async (e) => {
      e.preventDefault();
      await fetch("/api/logout", { method: "POST" });
      location.href = "/login";
    });
    nav.appendChild(link);
  } catch (_) { /* offline or local mode: no sign-out link */ }
})();
