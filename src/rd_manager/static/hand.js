(() => {
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "class") e.className = v; else if (k.startsWith("on")) e[k] = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v); }
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
async function api(path, opts) {
  const r = await fetch(path, opts); const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText || String(r.status));
  return d;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const err = (e) => handLog("error: " + (e && e.message ? e.message : e));
// The AmazingHand over UDP, through the manager (/api/hand*): the hand card of the Arm page (it was the Hand tab).
// Sliders are by PHYSICAL finger. The four curl channels the API takes are in servo order: channel 0 is the index
// (servos 1-2), 1 the middle finger (3-4), 2 the pinky (5-6), 3 the thumb (7-8) (checked on the hand with these
// sliders, Bryan, 2026-11-20). The channel order itself never changes: recorded datasets replay through it.
const HAND_FINGERS = [["thumb", 3], ["index", 0], ["middle", 1], ["pinky", 2]];   // [label, curl channel]
const HAND_CH = Object.fromEntries(HAND_FINGERS);                                // label -> channel
const handPoseOf = (open) => { const c = [1, 1, 1, 1]; for (const [f, v] of Object.entries(open)) c[HAND_CH[f]] = v; return c; };
const HAND_POSES = { open: [0, 0, 0, 0], half: [0.5, 0.5, 0.5, 0.5], close: [1, 1, 1, 1],
                     point: handPoseOf({ index: 0 }), pinch: handPoseOf({ index: 0.75, thumb: 0.8 }), thumbs: handPoseOf({ thumb: 0 }) };
const handS = { curls: [0, 0, 0, 0], abduct: [0, 0, 0, 0], timer: null, pending: null, lastSend: 0, gesturesBuilt: false };
const handHost = () => $("#hand-host").value.trim();
function handLog(line) {
  const pre = $("#hand-log");
  pre.textContent = [new Date().toLocaleTimeString() + "  " + line, ...pre.textContent.split("\n")].filter((x) => x).slice(0, 10).join("\n");
}
function handRow(name, idx) {
  // idx: the finger's curl channel (see HAND_FINGERS); null: the "all" row, which drives every finger at once
  const row = el("div", { class: "hand-r" + (idx === null ? " all" : "") });
  const curl = el("input", { type: "range", min: 0, max: 100, value: 0, title: "curl: 0 open … 1 closed" });
  const cval = el("span", { class: "val" }, "0.00");
  const abd = idx === null ? el("span", { class: "muted", style: "font-size: 11px" }, "—") : el("input", { type: "range", min: -100, max: 100, value: 0, title: "abduction: -1 … 1" });
  const aval = el("span", { class: "val" }, idx === null ? "" : "+0.00");
  curl.addEventListener("input", () => {
    const c = curl.value / 100;
    if (idx === null) handS.curls = [c, c, c, c]; else handS.curls[idx] = c;
    handSyncSliders(); handSendCurls();
  });
  curl.addEventListener("change", () => handSendCurls(true));
  if (idx !== null) {
    abd.addEventListener("input", () => { handS.abduct[idx] = abd.value / 100; handSyncSliders(); handSendCurls(); });
    abd.addEventListener("change", () => handSendCurls(true));
  }
  row.append(el("span", { class: "fname" }, name), curl, cval, abd, aval);
  row._set = (c, a) => {
    curl.value = Math.round(c * 100); cval.textContent = c.toFixed(2);
    if (idx !== null) { abd.value = Math.round(a * 100); aval.textContent = (a >= 0 ? "+" : "") + a.toFixed(2); }
  };
  return row;
}
function buildHandGrid() {
  const g = $("#hand-grid");
  if (g.childElementCount) return;
  g.append(el("div", { class: "hand-r hdr" }, el("span"), el("span", {}, "curl"), el("span"), el("span", {}, "abduction"), el("span")));
  handS.rows = [handRow("all", null), ...HAND_FINGERS.map(([f, ch]) => handRow(f, ch))];
  handS.rows.forEach((r) => g.append(r));
  handSyncSliders();
}
function handSyncSliders() {
  if (!handS.rows) return;
  const same = handS.curls.every((c) => Math.abs(c - handS.curls[0]) < 1e-6);
  handS.rows[0]._set(same ? handS.curls[0] : handS.curls.reduce((a, b) => a + b, 0) / 4, 0);
  HAND_FINGERS.forEach(([f, ch], i) => handS.rows[i + 1]._set(handS.curls[ch], handS.abduct[ch]));
}
async function handSendCurls(now = false) {
  // while dragging, at most ~14 lines/s (the teleop stream's order of magnitude); the last value always goes out
  const dt = performance.now() - handS.lastSend;
  if (!now && dt < 70) { clearTimeout(handS.pending); handS.pending = setTimeout(() => handSendCurls(true), 70 - dt); return; }
  clearTimeout(handS.pending); handS.lastSend = performance.now();
  const body = { curls: handS.curls.slice(), abduct: handS.abduct.slice(), speed: +$("#hand-speed").value || 200, host: handHost() || undefined };
  try { const r = await post("/api/hand/curls", body); handLog(r.line); } catch (e) { err(e); }
}
function handPose(name) { handS.curls = HAND_POSES[name].slice(); handS.abduct = [0, 0, 0, 0]; handSyncSliders(); handSendCurls(true); }
async function handGesture(c, label) {
  try { await post("/api/hand/gesture", { line: c, host: handHost() || undefined }); handLog(`${c}  (${label})`); } catch (e) { err(e); }
}
function handUptime(ms) { if (ms == null) return ""; const s = Math.floor(ms / 1000), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`; }
async function handStatus() {
  const st = $("#hand-status");
  let d; try { d = await api("/api/hand" + (handHost() ? "?host=" + encodeURIComponent(handHost()) : "")); } catch (e) { st.textContent = "○ " + e.message; st.className = "mono muted"; return; }
  $("#hand-console").href = d.console_url;
  $("#hand-host").placeholder = d.host;
  const recent = !d.alive && d.last_alive_s != null && d.last_alive_s < 90;   // one lost probe on a lossy link is not a dead hand
  st.textContent = d.alive ? `● ${d.host} · up ${handUptime(d.uptime_ms)} · ${d.state.sent} sent${d.state.errors ? ` · ${d.state.errors} failed` : ""}`
    : recent ? `◐ ${d.host}: probe lost (answered ${Math.round(d.last_alive_s)} s ago)` : `○ ${d.host}: ${d.error}`;
  st.className = d.alive ? "mono ok" : recent ? "mono warn" : "mono muted";
  if (!handS.gesturesBuilt) {
    handS.gesturesBuilt = true;
    $("#hand-gestures").replaceChildren(...Object.entries(d.gestures).map(([c, label]) => el("button", { class: "btn", title: `sends "${c}"`, onclick: () => handGesture(c, label) }, `${c} · ${label}`)));
    const sp = $("#hand-speed"); sp.min = d.speed_range[0]; sp.max = d.speed_range[1];
  }
  if (!handS.seeded && d.state.last) { handS.seeded = true; handS.curls = d.state.curls.slice(); handS.abduct = d.state.abduct.slice(); $("#hand-speed").value = d.state.speed; handSyncSliders(); }
}
async function loadHandView() {
  if (!$("#hand-host").value) $("#hand-host").value = localStorage.getItem("handHost") || "";
  buildHandGrid();
  await handStatus();
  clearInterval(handS.timer);
  handS.timer = setInterval(() => { if (!document.hidden) handStatus(); }, 5000);
}
$("#hand-host").addEventListener("change", () => { localStorage.setItem("handHost", handHost()); handStatus(); });
$("#btn-hand-probe").onclick = handStatus;
document.querySelectorAll(".hand-wrap [data-pose]").forEach((b) => b.addEventListener("click", () => handPose(b.dataset.pose)));
$("#btn-hand-send").onclick = async () => {
  const line = $("#hand-raw").value.trim(); if (!line) return;
  try { const r = await post("/api/hand/raw", { line, host: handHost() || undefined }); handLog(line + (r.reply ? "  →  " + r.reply : "")); }
  catch (e) { err(e); }
};
$("#hand-raw").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#btn-hand-send").click(); });
loadHandView();
})();
