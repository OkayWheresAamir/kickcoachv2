// KickCoach labelling page. It holds one clip's label file in memory and autosaves the whole
// file to tools/label_tool.py after every change. Definitions: docs/LABELLING_GUIDE.md.
"use strict";

const IDLE_MS = 30000; // heuristic: 30 s without input counts as a break, not labelling time
const HIT_PX = 10;
const ROLE_NAMES = { max_backswing: "max backswing", pre_contact: "contact − 1", contact: "contact" };
const REQUIRED_ROLES = ["max_backswing", "contact"]; // DEC-014; contact − 1 is optional
const isRequired = t => t.roles.some(r => REQUIRED_ROLES.includes(r));
const COLORS = { left: "#4fc3f7", right: "#ff8a65", ball: "#ffee58" };
const STATE_NAMES = { 2: "visible", 1: "occluded", 0: "unknown" };
const JOINT_EDGES = [["shoulder", "hip"], ["hip", "knee"], ["knee", "ankle"], ["ankle", "heel"],
  ["ankle", "big_toe"], ["heel", "big_toe"]];
const EDGES = [["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"],
  ...["left", "right"].flatMap(s => JOINT_EDGES.map(([a, b]) => [`${s}_${a}`, `${s}_${b}`]))];

const $ = id => document.getElementById(id);
const imgCanvas = $("img-canvas"), ov = $("ov-canvas");
const ictx = imgCanvas.getContext("2d"), octx = ov.getContext("2d");
const imgs = new Map();

const S = {
  mode: "label", clips: [], points: [], clip: null, label: null, qa: [],
  view: "events", cur: 0, task: null, sel: null, peek: 0,
  showLabels: true, showGhost: true, showSkel: false, hideImage: false,
  scale: 1, tx: 0, ty: 0, undo: [], drag: null, lastInput: Date.now(), ticks: 0,
  repeat: null, repeatIdx: 0, playing: null,
  dirty: false, saving: false, pendingSave: false, saveTimer: null,
};

const sideOf = n => n.startsWith("left_") ? "left" : n.startsWith("right_") ? "right" : "ball";
const short = n => n.replace("left_", "L ").replace("right_", "R ").replace("big_toe", "toe");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

async function api(method, url, body) {
  const r = await fetch(url, {
    method, headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function isoNow() {
  const d = new Date(), off = -d.getTimezoneOffset();
  const p = n => String(Math.floor(Math.abs(n))).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:` +
    `${p(d.getMinutes())}:${p(d.getSeconds())}${off >= 0 ? "+" : "-"}${p(off / 60)}:${p(off % 60)}`;
}

let toastTimer = null;
function toast(msg, ms = 3000) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

// ---------- images and view transform ----------

function img(cid, i) {
  const k = `${cid}/${i}`;
  let im = imgs.get(k);
  if (!im) {
    im = new Image();
    im.onload = () => { if (S.clip && S.clip.clip_id === cid) draw(); };
    im.src = `/frames/${cid}/${i}.jpg`;
    imgs.set(k, im);
  }
  return im;
}

function preload(cid, n) {
  for (const k of [...imgs.keys()]) if (!k.startsWith(cid + "/")) imgs.delete(k);
  let i = 0;
  const next = () => {
    if (i >= n || !S.clip || S.clip.clip_id !== cid) return;
    const im = img(cid, i++);
    if (im.complete) next();
    else { im.addEventListener("load", next, { once: true }); im.addEventListener("error", next, { once: true }); }
  };
  for (let k = 0; k < 6; k++) next();
}

const viewH = () => ov.clientHeight - 44; // keep the scrubber bar clear
const toScreen = (x, y) => [x * S.scale + S.tx, y * S.scale + S.ty];
function toImage(e) {
  const r = ov.getBoundingClientRect();
  return [(e.clientX - r.left - S.tx) / S.scale, (e.clientY - r.top - S.ty) / S.scale];
}

function fitRect(x0, y0, x1, y1) {
  const W = ov.clientWidth, H = viewH();
  const s = Math.min(W / (x1 - x0), H / (y1 - y0));
  S.scale = s;
  S.tx = (W - (x1 - x0) * s) / 2 - x0 * s;
  S.ty = (H - (y1 - y0) * s) / 2 - y0 * s;
}
const fitWhole = () => fitRect(0, 0, S.clip.width, S.clip.height);

function fitPlayer() {
  const pts = [];
  const add = p => { if (p && p.v > 0) pts.push([p.x, p.y]); };
  if (S.view === "frame") {
    const e = entry();
    Object.values(e.points).forEach(add);
    Object.values(e.prefill).forEach(p => p && pts.push([p[0], p[1]]));
  } else {
    Object.values(modelPoints(S.cur)).forEach(add);
  }
  if (!pts.length) return fitWhole();
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const pad = Math.max(x1 - x0, y1 - y0, Math.min(S.clip.width, S.clip.height) * 0.25) * 0.3;
  fitRect(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
}

function zoomAt(sx, sy, f) {
  const s = clamp(S.scale * f, 0.05, 80), k = s / S.scale;
  S.tx = sx - (sx - S.tx) * k;
  S.ty = sy - (sy - S.ty) * k;
  S.scale = s;
  draw();
}

// ---------- label data ----------

function frameRoles() { // same rule as kickcoach.labels.schema.frame_roles
  const c = S.label.clip, n = S.clip.n_frames, roles = {};
  const add = (i, r) => { if (i !== null && i >= 0 && i < n) (roles[i] = roles[i] || []).push(r); };
  add(c.max_backswing_frame, "max_backswing");
  add(c.contact_frame === null ? null : c.contact_frame - 1, "pre_contact");
  add(c.contact_frame, "contact");
  return roles;
}

function tasks() {
  if (S.mode === "repeat") return S.repeat.tasks.map((t, i) => ({ frame: t.frame, idx: i }));
  const roles = frameRoles();
  const frames = new Set([...Object.keys(roles), ...Object.keys(S.label.frames)].map(Number));
  return [...frames].sort((a, b) => a - b).map(f => ({ frame: f, roles: roles[f] || [] }));
}

function entry() {
  if (S.mode === "repeat") return S.repeat.tasks[S.repeatIdx];
  return S.label.frames[String(S.task.frame)];
}

function modelPoints(i) {
  const out = {};
  S.points.forEach((n, k) => {
    const p = S.clip.prefill[i][k];
    if (p) out[n] = { x: p[0], y: p[1], v: 2 };
  });
  return out;
}

function newEntry(row) {
  const e = { points: {}, prefill: {}, confirmed: false, confirmed_at: null, active_s: 0, wall_s: 0, edits: 0 };
  S.points.forEach((n, k) => {
    const p = row[k];
    e.prefill[n] = p;
    const inside = p && p[0] >= 0 && p[0] <= S.clip.width && p[1] >= 0 && p[1] <= S.clip.height;
    e.points[n] = inside ? { x: p[0], y: p[1], v: 2 } : { x: null, y: null, v: 0 };
  });
  return e;
}

// ---------- saving ----------

function setSaveState(s) { $("save-state").textContent = s; }

function changed(now = false) {
  S.dirty = true;
  setSaveState("unsaved");
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, now ? 0 : 600);
}

async function save() {
  if (S.saving) { S.pendingSave = true; return; }
  S.saving = true; S.dirty = false;
  try {
    if (S.mode === "repeat") {
      await api("PUT", "/api/repeat", S.repeat);
    } else {
      const r = await api("PUT", `/api/clip/${S.clip.clip_id}`, S.label);
      S.qa = r.qa;
      const c = S.clips.find(c => c.clip_id === S.clip.clip_id);
      if (c) { c.status = r.status; fillClipSelect(); }
    }
    setSaveState(S.dirty ? "unsaved" : "saved ✓");
  } catch (err) {
    S.dirty = true;
    setSaveState("SAVE FAILED");
    toast(`Save failed: ${err.message}`, 10000);
  } finally {
    S.saving = false;
    if (S.pendingSave) { S.pendingSave = false; save(); }
  }
  renderQA();
  if (S.view === "frame") renderTasks();
}

async function flushSave() {
  clearTimeout(S.saveTimer);
  if (S.dirty) await save();
  while (S.saving) await sleep(50);
}

// ---------- drawing ----------

function resize() {
  const dpr = devicePixelRatio || 1;
  for (const c of [imgCanvas, ov]) {
    c.width = Math.round(c.clientWidth * dpr);
    c.height = Math.round(c.clientHeight * dpr);
  }
}

const shownFrame = () => S.view === "frame"
  ? clamp(S.task.frame + S.peek, 0, S.clip.n_frames - 1) : S.cur;

function draw() {
  if (!S.clip) return;
  const dpr = devicePixelRatio || 1;
  if (ov.width !== Math.round(ov.clientWidth * dpr) || ov.height !== Math.round(ov.clientHeight * dpr)) resize();
  ictx.setTransform(1, 0, 0, 1, 0, 0);
  ictx.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
  ictx.setTransform(dpr * S.scale, 0, 0, dpr * S.scale, dpr * S.tx, dpr * S.ty);
  ictx.imageSmoothingEnabled = S.scale < 3;
  const im = img(S.clip.clip_id, shownFrame());
  if (!S.hideImage && im.complete && im.naturalWidth) ictx.drawImage(im, 0, 0, S.clip.width, S.clip.height);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, ov.clientWidth, ov.clientHeight);
  // With the image hidden the skeleton is all there is, so show it regardless of K.
  if (S.view === "events" && (S.showSkel || S.hideImage)) drawSkeleton(modelPoints(S.cur), true);
  if (S.view === "frame" && S.showLabels) {
    if (S.peek) drawSkeleton(entry().points, false, 0.35); // faint: they belong to another frame
    else {
      if (S.showGhost) drawGhosts();
      drawSkeleton(entry().points, false);
    }
  }
  updateFrameLabel();
}

function drawSkeleton(points, isModel, alpha = 1) {
  octx.lineWidth = isModel ? 1.5 : 2;
  octx.setLineDash(isModel ? [5, 4] : []);
  for (const [a, b] of EDGES) {
    const pa = points[a], pb = points[b];
    if (!pa || !pb || pa.v === 0 || pb.v === 0) continue;
    octx.strokeStyle = sideOf(a) === sideOf(b) ? COLORS[sideOf(a)] : "#cfd6df";
    octx.globalAlpha = (isModel ? 0.7 : 0.85) * alpha;
    octx.beginPath();
    octx.moveTo(...toScreen(pa.x, pa.y));
    octx.lineTo(...toScreen(pb.x, pb.y));
    octx.stroke();
  }
  octx.setLineDash([]);
  octx.globalAlpha = alpha;
  octx.font = "11px -apple-system, sans-serif";
  for (const n of S.points) {
    const p = points[n];
    if (!p || p.v === 0) continue;
    const [x, y] = toScreen(p.x, p.y), col = COLORS[sideOf(n)];
    const r = isModel ? 3 : (n === S.sel ? 6 : 4.5);
    octx.beginPath();
    octx.arc(x, y, r, 0, 2 * Math.PI);
    if (p.v === 2) { octx.fillStyle = col; octx.fill(); }
    else { octx.fillStyle = "rgba(0,0,0,0.55)"; octx.fill(); octx.strokeStyle = col; octx.lineWidth = 2; octx.stroke(); }
    if (isModel || alpha < 1) continue;
    if (n === S.sel) {
      octx.beginPath(); octx.arc(x, y, r + 4, 0, 2 * Math.PI);
      octx.strokeStyle = "#fff"; octx.lineWidth = 1.5; octx.stroke();
    }
    octx.lineWidth = 3; octx.strokeStyle = "rgba(0,0,0,0.8)";
    octx.strokeText(short(n), x + 8, y - 7);
    octx.fillStyle = col;
    octx.fillText(short(n), x + 8, y - 7);
  }
  octx.globalAlpha = 1;
  if (isModel) {
    octx.fillStyle = "#cfd6df";
    octx.fillText("model skeleton (not a label)", 10, 18);
  }
}

function drawGhosts() {
  const e = entry();
  octx.lineWidth = 1;
  for (const n of S.points) {
    const pre = e.prefill[n], p = e.points[n];
    if (!pre) continue;
    const [gx, gy] = toScreen(pre[0], pre[1]);
    const moved = p.v === 0 || Math.hypot(p.x - pre[0], p.y - pre[1]) > 1;
    if (!moved) continue;
    octx.strokeStyle = COLORS[sideOf(n)];
    octx.globalAlpha = 0.6;
    octx.beginPath();
    octx.moveTo(gx - 4, gy - 4); octx.lineTo(gx + 4, gy + 4);
    octx.moveTo(gx + 4, gy - 4); octx.lineTo(gx - 4, gy + 4);
    octx.stroke();
    if (p.v > 0) {
      octx.setLineDash([3, 3]);
      octx.beginPath(); octx.moveTo(gx, gy); octx.lineTo(...toScreen(p.x, p.y)); octx.stroke();
      octx.setLineDash([]);
    }
    octx.globalAlpha = 1;
  }
}

function updateFrameLabel() {
  if (S.view === "frame") {
    const roles = S.mode === "repeat" ? "" : (frameRoles()[S.task.frame] || []).map(r => ROLE_NAMES[r]).join(" + ");
    const peek = S.peek ? ` · showing frame ${shownFrame()} / ${S.clip.n_frames - 1}` : "";
    $("frame-label").textContent = `labelling frame ${S.task.frame}${roles ? " · " + roles : ""}${peek}`;
  } else {
    const c = S.label.clip, marks = [];
    if (c.max_backswing_frame === S.cur) marks.push("MAX BACKSWING");
    if (c.contact_frame === S.cur) marks.push("CONTACT");
    $("frame-label").textContent = `frame ${S.cur} / ${S.clip.n_frames - 1}` +
      (marks.length ? ` · ${marks.join(" + ")}` : "");
  }
}

// ---------- panels ----------

function statusText(st) {
  if (st.complete) return "✓ done";
  if (!st.events_done) return "events to do";
  return `${st.frames_confirmed}/${st.frames_needed} frames`;
}

function fillClipSelect() {
  const sel = $("clip-select"), cur = S.clip && S.clip.clip_id;
  sel.innerHTML = "";
  for (const c of S.clips) {
    const o = document.createElement("option");
    o.value = c.clip_id;
    o.textContent = `${c.clip_id} · ${c.purpose === "gate_only" ? "gate only" : "eval"}` +
      `${c.split === "heldout" ? " (held out)" : ""} · ${statusText(c.status)}`;
    sel.appendChild(o);
  }
  if (cur) sel.value = cur;
}

function renderEvents() {
  const c = S.label.clip;
  for (const b of document.querySelectorAll("#events-panel [data-field]")) {
    b.classList.toggle("on", c[b.dataset.field] === b.dataset.value);
  }
  $("bsw-val").textContent = c.max_backswing_frame ?? "–";
  $("contact-val").textContent = c.contact_frame ?? "–";
  $("between").checked = !!c.contact_between_frames;
  if (document.activeElement !== $("notes")) $("notes").value = c.notes || "";
  const gate = S.clip.purpose === "gate_only";
  for (const el of document.querySelectorAll(".eval-only")) el.hidden = gate;
  $("clip-meta").textContent = `${S.clip.width}×${S.clip.height}, ${S.clip.n_frames} frames` +
    (gate ? " · gate only: just View OK (and contact if visible)" : "");
}

function renderTasks() {
  const list = $("task-list");
  list.innerHTML = "";
  for (const t of tasks()) {
    const div = document.createElement("div");
    let e, label;
    if (S.mode === "repeat") {
      e = S.repeat.tasks[t.idx];
      label = `${t.idx + 1}. ${e.clip_id} frame ${t.frame}`;
    } else {
      e = S.label.frames[String(t.frame)];
      label = `frame ${t.frame} · ${t.roles.length ? t.roles.map(r => ROLE_NAMES[r]).join(" + ") : "extra frame (not backswing or contact)"}` +
        (t.roles.length && !isRequired(t) ? " (optional)" : "");
    }
    const cur = S.mode === "repeat" ? t.idx === S.repeatIdx : S.task && t.frame === S.task.frame;
    div.className = "task" + (cur ? " cur" : "");
    div.innerHTML = `<span class="tick">${e && e.confirmed ? "✓" : ""}</span><span>${label}</span>`;
    div.onclick = () => S.mode === "repeat" ? openRepeatTask(t.idx) : openTask(t.frame);
    list.appendChild(div);
  }
}

function renderPoints() {
  const list = $("point-list"), e = entry();
  list.innerHTML = "";
  for (const n of S.points) {
    const p = e.points[n], pre = e.prefill[n];
    let moved = "";
    if (p.v > 0 && pre) {
      const d = Math.hypot(p.x - pre[0], p.y - pre[1]);
      moved = d < 1 ? "= model" : `moved ${d.toFixed(0)} px`;
    } else if (p.v > 0) moved = "placed";
    const div = document.createElement("div");
    div.className = "pt" + (n === S.sel ? " sel" : "");
    div.innerHTML = `<span class="dot ${sideOf(n)}"></span><span class="name">${n.replaceAll("_", " ")}</span>` +
      `<span class="moved">${moved}</span><span class="chip v${p.v}">${STATE_NAMES[p.v]}</span>`;
    div.onclick = () => select(n);
    list.appendChild(div);
  }
  $("confirm").textContent = e.confirmed ? "Confirmed ✓ (Enter = next)" : "Confirm frame (Enter)";
}

function renderQA() {
  const list = $("qa-list");
  list.innerHTML = "";
  if (S.mode === "repeat") return;
  $("qa-count").textContent = S.qa.length ? `(${S.qa.length})` : "(none)";
  for (const f of S.qa) {
    const li = document.createElement("li");
    li.className = f.severity;
    li.textContent = (f.frame !== null ? `[frame ${f.frame}] ` : "") + f.message;
    li.onclick = () => {
      if (f.frame !== null && (S.label.frames[String(f.frame)] || frameRoles()[f.frame])) {
        openTask(f.frame);
        if (f.point) select(f.point);
      } else { showEvents(); }
    };
    list.appendChild(li);
  }
}

function renderBanner() {
  const b = $("banner");
  let msg = "";
  if (S.view === "frame" && S.peek) {
    msg = `Showing frame ${shownFrame()}; your points for frame ${S.task.frame} are drawn faint. ` +
      "Press / or Enter to go back and edit.";
  }
  else if (S.mode === "repeat") msg = "Blind relabel: your first labels are hidden. Label from scratch, as usual.";
  b.textContent = msg;
  b.hidden = !msg;
}

function renderTimer() {
  let t;
  if (S.view === "frame") t = entry();
  else if (S.mode === "label") t = S.label.clip_timing;
  if (!t) { $("timer").textContent = ""; return; }
  const what = S.view === "frame" ? "this frame" : "clip events";
  $("timer").textContent = `${what}: active ${fmt(t.active_s || 0)} (on screen ${fmt(t.wall_s || 0)})`;
}

function render() {
  $("events-panel").hidden = S.mode === "repeat" || S.view !== "events";
  $("frames-panel").hidden = S.view !== "frame";
  if (S.mode === "label") renderEvents();
  if (S.view === "frame") { renderTasks(); renderPoints(); }
  renderQA(); renderBanner(); renderTimer();
  $("scrub").value = shownFrame();
  draw();
}

// ---------- actions ----------

async function openClip(cid) {
  await flushSave();
  stopPlay();
  const p = await api("GET", `/api/clip/${cid}`);
  S.clip = p; S.label = p.label; S.qa = p.qa;
  S.cur = S.label.clip.contact_frame ?? 0;
  S.view = "events"; S.task = null; S.undo = []; S.sel = null;
  $("clip-select").value = cid;
  $("scrub").max = p.n_frames - 1;
  preload(cid, p.n_frames);
  fitWhole();
  render();
}

function showEvents() {
  if (S.mode === "repeat") return;
  stopPlay();
  if (S.view === "frame") S.cur = S.task.frame;
  S.view = "events"; S.peek = 0;
  render();
}

function openTask(frame) {
  stopPlay();
  const key = String(frame);
  if (!S.label.frames[key]) { S.label.frames[key] = newEntry(S.clip.prefill[frame]); changed(); }
  S.view = "frame"; S.task = { frame }; S.peek = 0; S.undo = []; S.sel = null;
  fitPlayer();
  render();
}

function openRepeatTask(i) {
  stopPlay();
  S.repeatIdx = i;
  const t = S.repeat.tasks[i];
  S.clip = { clip_id: t.clip_id, width: t.image_size[0], height: t.image_size[1], n_frames: t.n_frames, purpose: "eval" };
  if (!Object.keys(t.points).length) {
    const e = newEntry(t.prefill_row);
    t.points = e.points; t.prefill = e.prefill;
    changed();
  }
  S.view = "frame"; S.task = { frame: t.frame }; S.peek = 0; S.undo = []; S.sel = null;
  fitPlayer();
  render();
}

function toKeypoints() {
  if (S.clip.purpose === "gate_only") return toast("Gate-only clip: no keypoints needed.");
  const c = S.label.clip;
  if (c.contact_frame === null || c.max_backswing_frame === null) {
    return toast("Set the max backswing and contact frames first.");
  }
  const ts = tasks().filter(isRequired);
  const next = ts.find(t => !(S.label.frames[String(t.frame)] || {}).confirmed) || ts[0];
  openTask(next.frame);
}

function setField(field, value) {
  S.label.clip[field] = value;
  changed();
  render();
}

function setEventFrame(field, name) {
  const old = frameRoles();
  S.label.clip[field] = S.cur;
  const orphans = Object.keys(S.label.frames).map(Number).filter(f => old[f] && !frameRoles()[f]);
  changed();
  render();
  toast(`${name} = frame ${S.cur}` + (orphans.length
    ? `. Frame ${orphans.join(", ")} keeps its labels but is no longer a role frame.` : ""));
}

function select(n) { S.sel = n; renderPoints(); draw(); }

function cycle(dir) {
  const i = S.points.indexOf(S.sel);
  select(S.points[(i + dir + S.points.length) % S.points.length]);
}

function pushUndo() {
  S.undo.push(JSON.stringify(entry().points));
  if (S.undo.length > 200) S.undo.shift();
}

function edited() {
  const e = entry();
  if (e.confirmed) {
    e.confirmed = false; e.confirmed_at = null;
    toast("You edited a confirmed frame: press Enter to confirm it again.");
  }
  e.edits = (e.edits || 0) + 1;
  changed();
}

function undo() {
  if (!S.undo.length) return toast("Nothing to undo");
  entry().points = JSON.parse(S.undo.pop());
  edited();
  render();
}

function setState(v) {
  if (!S.sel) return toast("Select a point first (Tab, or click its row)");
  const p = entry().points[S.sel];
  if (v > 0 && p.v === 0) return toast("Click the image to place this point first");
  if (p.v === v) return;
  pushUndo();
  if (v === 0) { p.x = null; p.y = null; }
  p.v = v;
  edited();
  render();
}

function resetToModel() {
  if (!S.sel) return;
  const pre = entry().prefill[S.sel], p = entry().points[S.sel];
  if (!pre || pre[0] < 0 || pre[0] > S.clip.width || pre[1] < 0 || pre[1] > S.clip.height) {
    return toast("The model has no usable guess for this point");
  }
  pushUndo();
  Object.assign(p, { x: pre[0], y: pre[1], v: p.v || 2 });
  edited();
  render();
}

function confirmFrame() {
  const e = entry();
  if (!e.confirmed) {
    e.confirmed = true;
    e.confirmed_at = isoNow();
    changed(true);
  }
  if (S.mode === "repeat") {
    const i = S.repeat.tasks.findIndex(t => !t.confirmed);
    if (i >= 0) openRepeatTask(i); else { render(); toast("All relabel frames confirmed. Thank you!", 6000); }
    return;
  }
  const next = tasks().find(t => isRequired(t) && !(S.label.frames[String(t.frame)] || {}).confirmed);
  if (next) openTask(next.frame);
  else {
    render();
    toast("Backswing and contact are confirmed. Contact − 1 is optional (click it in the list); " +
      "otherwise pick the next clip.", 8000);
  }
}

function stepTask(dir) {
  if (S.mode === "repeat") {
    return openRepeatTask(clamp(S.repeatIdx + dir, 0, S.repeat.tasks.length - 1));
  }
  const ts = tasks(), i = ts.findIndex(t => t.frame === S.task.frame);
  openTask(ts[clamp(i + dir, 0, ts.length - 1)].frame);
}

function stopPlay() { if (S.playing) { clearInterval(S.playing); S.playing = null; } }
function togglePlay() {
  if (S.playing) return stopPlay();
  S.playing = setInterval(() => {
    if (shownFrame() >= S.clip.n_frames - 1) return stopPlay();
    if (S.view === "frame") S.peek++; else S.cur++;
    render();
  }, 1000 / 15);
}

// In the keypoint view, look at any frame of the clip without leaving the labelled one.
function peekTo(f) { S.peek = clamp(f, 0, S.clip.n_frames - 1) - S.task.frame; render(); }

function setCur(i) { S.cur = clamp(i, 0, S.clip.n_frames - 1); render(); }

// ---------- input ----------

function hitTest(e) {
  const r = ov.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  let best = null, bestD = HIT_PX;
  for (const n of S.points) {
    const p = entry().points[n];
    if (p.v === 0) continue;
    const [x, y] = toScreen(p.x, p.y), d = Math.hypot(x - sx, y - sy);
    if (d < bestD || (d === bestD && n === S.sel)) { best = n; bestD = d; }
  }
  return best;
}

const editable = () => S.view === "frame" && S.peek === 0 && S.showLabels;

ov.addEventListener("pointerdown", e => {
  S.lastInput = Date.now();
  ov.setPointerCapture(e.pointerId);
  if (editable()) {
    const [mx, my] = toImage(e);
    const hit = hitTest(e);
    if (hit) {
      const p = entry().points[hit];
      select(hit);
      pushUndo();
      S.drag = { kind: "point", name: hit, dx: p.x - mx, dy: p.y - my, moved: false };
      return;
    }
    if (S.sel && (entry().points[S.sel].v === 0 || e.shiftKey)) {
      pushUndo();
      const p = entry().points[S.sel];
      Object.assign(p, { x: clamp(mx, 0, S.clip.width), y: clamp(my, 0, S.clip.height), v: p.v || 2 });
      S.drag = { kind: "point", name: S.sel, dx: 0, dy: 0, moved: true };
      render();
      return;
    }
  }
  S.drag = { kind: "pan", sx: e.clientX, sy: e.clientY, tx: S.tx, ty: S.ty };
});

ov.addEventListener("pointermove", e => {
  const d = S.drag;
  if (!d) {
    if (editable()) ov.style.cursor = hitTest(e) ? "grab" : "crosshair";
    return;
  }
  S.lastInput = Date.now();
  if (d.kind === "pan") {
    S.tx = d.tx + e.clientX - d.sx;
    S.ty = d.ty + e.clientY - d.sy;
  } else {
    const [mx, my] = toImage(e), p = entry().points[d.name];
    p.x = clamp(mx + d.dx, 0, S.clip.width);
    p.y = clamp(my + d.dy, 0, S.clip.height);
    d.moved = true;
  }
  draw();
});

function endDrag() {
  const d = S.drag;
  S.drag = null;
  if (!d || d.kind !== "point") return;
  if (d.moved) { edited(); renderPoints(); } else S.undo.pop();
}
ov.addEventListener("pointerup", endDrag);
ov.addEventListener("pointercancel", endDrag);

ov.addEventListener("wheel", e => {
  e.preventDefault();
  S.lastInput = Date.now();
  const k = e.deltaMode === 1 ? 16 : 1, r = ov.getBoundingClientRect();
  if (e.ctrlKey || e.altKey) zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * k * 0.01));
  else { S.tx -= e.deltaX * k; S.ty -= e.deltaY * k; draw(); }
}, { passive: false });

document.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (b) b.blur(); // so Enter/Space go to the shortcuts, not to the last clicked button
});

document.addEventListener("keydown", e => {
  S.lastInput = Date.now();
  const tag = e.target.tagName;
  if (tag === "TEXTAREA") { if (e.key === "Escape") e.target.blur(); return; }
  if (tag === "SELECT") return;
  if (tag === "INPUT") e.target.blur();
  const k = e.key, lk = k.length === 1 ? k.toLowerCase() : k;
  if ((e.metaKey || e.ctrlKey) && lk === "z") { e.preventDefault(); if (editable()) undo(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey || !S.clip) return;
  let handled = true;
  if (lk === "f") { fitPlayer(); draw(); }
  else if (k === "0") { fitWhole(); draw(); }
  else if (k === "+" || k === "=") zoomAt(ov.clientWidth / 2, viewH() / 2, 1.25);
  else if (k === "-") zoomAt(ov.clientWidth / 2, viewH() / 2, 0.8);
  else if (lk === "i") setHideImage(!S.hideImage);
  else if (S.view === "events") {
    if (k === "ArrowRight") setCur(S.cur + (e.shiftKey ? 10 : 1));
    else if (k === "ArrowLeft") setCur(S.cur - (e.shiftKey ? 10 : 1));
    else if (k === " ") togglePlay();
    else if (lk === "c") setEventFrame("contact_frame", "Contact");
    else if (lk === "b" && S.clip.purpose !== "gate_only") setEventFrame("max_backswing_frame", "Max backswing");
    else if (lk === "k") { S.showSkel = !S.showSkel; draw(); }
    else if (k === "Enter") toKeypoints();
    else handled = false;
  } else {
    if (k === "Tab") cycle(e.shiftKey ? -1 : 1);
    else if (k === "Enter" && S.peek) { stopPlay(); S.peek = 0; render(); toast(`Back on frame ${S.task.frame}. Enter again confirms it.`); }
    else if (k === "Enter") confirmFrame();
    else if (k === "ArrowRight") peekTo(shownFrame() + (e.shiftKey ? 10 : 1));
    else if (k === "ArrowLeft") peekTo(shownFrame() - (e.shiftKey ? 10 : 1));
    else if (k === ",") peekTo(shownFrame() - 1);
    else if (k === ".") peekTo(shownFrame() + 1);
    else if (k === " ") togglePlay();
    else if (k === "/") { stopPlay(); S.peek = 0; render(); }
    else if (lk === "h") { S.showLabels = !S.showLabels; draw(); }
    else if (lk === "g") { S.showGhost = !S.showGhost; draw(); }
    else if (lk === "n") stepTask(1);
    else if (lk === "p") stepTask(-1);
    else if (lk === "e") showEvents();
    else if (k === "Escape") select(null);
    else if (!editable()) handled = false;
    else if (lk === "v") setState(2);
    else if (lk === "o") setState(1);
    else if (lk === "u") setState(0);
    else if (lk === "r") resetToModel();
    else if (lk === "b") select("ball");
    else handled = false;
  }
  if (handled) e.preventDefault();
});

$("scrub").addEventListener("input", e => {
  S.lastInput = Date.now();
  if (S.view === "frame") peekTo(Number(e.target.value)); else setCur(Number(e.target.value));
});
$("clip-select").addEventListener("change", e => { openClip(e.target.value); e.target.blur(); });
for (const b of document.querySelectorAll("#events-panel [data-field]")) {
  b.addEventListener("click", () => setField(b.dataset.field, b.dataset.value));
}
$("set-bsw").onclick = () => setEventFrame("max_backswing_frame", "Max backswing");
$("set-contact").onclick = () => setEventFrame("contact_frame", "Contact");
$("go-bsw").onclick = () => { if (S.label.clip.max_backswing_frame !== null) setCur(S.label.clip.max_backswing_frame); };
$("go-contact").onclick = () => { if (S.label.clip.contact_frame !== null) setCur(S.label.clip.contact_frame); };
$("between").addEventListener("change", e => { setField("contact_between_frames", e.target.checked); e.target.blur(); });
$("notes").addEventListener("input", e => { S.label.clip.notes = e.target.value; changed(); });
$("to-keypoints").onclick = toKeypoints;
$("confirm").onclick = confirmFrame;
$("back-events").onclick = showEvents;
function setHideImage(on) {
  S.hideImage = on;
  $("hide-img").checked = on;
  draw();
}
$("hide-img").addEventListener("change", e => { setHideImage(e.target.checked); e.target.blur(); });
const applyFilter = () => {
  imgCanvas.style.filter = `brightness(${$("bright").value}) contrast(${$("contrast").value})`;
};
$("bright").addEventListener("input", applyFilter);
$("contrast").addEventListener("input", applyFilter);
window.addEventListener("resize", () => { resize(); draw(); });
window.addEventListener("beforeunload", e => { if (S.dirty) e.preventDefault(); });

setInterval(() => {
  if (document.hidden || !S.clip) return;
  const t = S.view === "frame" ? entry() : (S.mode === "label" ? S.label.clip_timing : null);
  if (!t) return;
  t.wall_s = (t.wall_s || 0) + 1;
  if (Date.now() - S.lastInput < IDLE_MS) t.active_s = (t.active_s || 0) + 1;
  if (++S.ticks % 10 === 0 && !S.dirty) changed(); // persist the timing every 10 s
  renderTimer();
}, 1000);

async function init() {
  try {
    const st = await api("GET", "/api/state");
    S.mode = st.mode; S.points = st.points;
    if (S.mode === "repeat") {
      S.repeat = await api("GET", "/api/repeat");
      for (const id of ["clip-pick", "events-panel", "back-events", "qa-panel"]) $(id).hidden = true;
      $("frames-title").textContent = "Blind relabel";
      const i = S.repeat.tasks.findIndex(t => !t.confirmed);
      openRepeatTask(i >= 0 ? i : 0);
      for (const t of S.repeat.tasks) img(t.clip_id, t.frame);
      return;
    }
    S.clips = st.clips;
    fillClipSelect();
    const q = new URLSearchParams(location.search); // ?clip=c10&frame=60 opens a frame directly
    const first = S.clips.find(c => c.clip_id === q.get("clip")) ||
      S.clips.find(c => !c.status.complete) || S.clips[0];
    await openClip(first.clip_id);
    if (q.get("frame") !== null) openTask(Number(q.get("frame")));
  } catch (err) {
    toast(`Could not start: ${err.message}`, 60000);
  }
}
init();
