"use strict";

const form = document.getElementById("login-form");
const errorBox = document.getElementById("error");
const submit = document.getElementById("submit");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submit.disabled = true;
  submit.textContent = "Signing in…";
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: form.email.value, password: form.password.value }),
    });
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail; } catch (_) { /* not JSON */ }
      throw new Error(typeof detail === "string" && detail ? detail : "Could not sign in.");
    }
    window.location.href = "/";
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
    submit.disabled = false;
    submit.textContent = "Sign in";
  }
});
