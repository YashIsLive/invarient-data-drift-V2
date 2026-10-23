// Monitor page: run a scenario batch by batch and investigate any batch.
import { api, esc, num, pct, capital } from "./api.js";
import { COLORS, drawChart, drawMap, purgeChart } from "./charts.js";

const $ = (id) => document.getElementById(id);

const LEGEND = [
  ["normal", "All normal"], ["shift", "Harmless shift"], ["broken", "Broken relationship"],
  ["unclear", "Unclear"], ["quality", "Data quality issue"], ["insufficient", "Not enough data"],
];

const state = {
  dataset: null,      // dataset name
  profile: null,      // profile summary from the server
  catalogue: null,    // presets, fault types, features
  run: null,          // current run state
  selected: null,     // index of the batch being investigated
  follow: true,       // keep selecting the newest batch while running
  playing: false,
  truth: null,
  cache: new Map(),   // batch index -> full evidence
  candidate: 0,       // which candidate the fingerprint chart shows
  scatterFeature: null,
};

let notify = () => {};

export function initMonitor({ notice }) {
  notify = notice;
  $("scenario").addEventListener("change", showSteps);
  $("start").addEventListener("click", start);
  $("next").addEventListener("click", () => advance(1));
  $("play").addEventListener("click", togglePlay);
  $("finish").addEventListener("click", runToEnd);
  $("reveal").addEventListener("click", reveal);
  $("inj-fault").addEventListener("change", updateAmountField);
  $("inject-form").addEventListener("submit", injectFault);
  $("inj-shift").addEventListener("click", injectShift);
  $("lane-sentinel").addEventListener("keydown", onTimelineKey);
  window.addEventListener("resize", () => state.run && renderErrorChart());
}

// ---------- dataset / scenarios ----------

export async function loadMonitor(dataset, profile) {
  stopPlaying();
  state.dataset = dataset;
  state.profile = profile;
  state.run = null;
  resetView();
  state.catalogue = await api(`/datasets/${dataset}/scenarios`);
  const select = $("scenario");
  select.innerHTML = state.catalogue.presets
    .map((s) => `<option value="${esc(s.id)}">${esc(s.title)} · ${s.batches} ${esc(profile.batches_word)}</option>`)
    .join("");
  // the main demo is the most useful starting point when a dataset has one
  const main = state.catalogue.presets.find((s) => /main demo/i.test(s.title));
  if (main) select.value = main.id;
  showSteps();

  $("inj-feature").innerHTML = state.catalogue.features
    .map((f) => `<option value="${esc(f.name)}">${esc(f.label)}${f.unit ? ` (${esc(f.unit)})` : ""}</option>`).join("");
  $("inj-fault").innerHTML = Object.entries(state.catalogue.faults)
    .map(([k, f]) => `<option value="${k}">${esc(f.name)}</option>`).join("");
  updateAmountField();
}

function showSteps() {
  const s = state.catalogue?.presets.find((p) => p.id === $("scenario").value);
  if (!s) return;
  $("scenario-steps").textContent = s.steps.length
    ? "What will happen: " + s.steps.join("; ") + "."
    : `What will happen: nothing is changed. Ordinary ${state.profile.batches_word}, to see how often the system raises a false alarm.`;
}

function resetView() {
  state.selected = null;
  state.follow = true;
  state.truth = null;
  state.cache.clear();
  $("timeline").hidden = true;
  $("inject").hidden = true;
  $("investigation").hidden = true;
  $("truth-lane").hidden = true;
  $("status-where").textContent = "";
  $("status-tag").innerHTML = "";
  const h = $("status-headline");
  h.className = "headline idle";
  h.textContent = "Choose a scenario and press Start.";
  for (const id of ["next", "play", "finish", "reveal"]) $(id).disabled = true;
  $("reveal").textContent = "Reveal what was injected";
}

// ---------- running ----------

async function start() {
  stopPlaying();
  resetView();
  $("start").disabled = true;
  try {
    state.run = await api("/runs", { method: "POST", body: { dataset: state.dataset, preset: $("scenario").value } });
    $("timeline").hidden = false;
    $("inject").hidden = false;
    renderLegend();
    renderTimeline();
    const h = $("status-headline");
    h.textContent = `Ready. Press Next to analyse the first ${state.profile.batch_word}.`;
    updateButtons();
    $("next").focus();
  } catch (e) {
    notify(e.message, true);
  } finally {
    $("start").disabled = false;
  }
}

async function advance(count) {
  if (!state.run || state.run.finished) return;
  try {
    state.run = await api(`/runs/${state.run.id}/next?count=${count}`, { method: "POST" });
  } catch (e) {
    stopPlaying();
    notify(e.message, true);
    return;
  }
  renderTimeline();
  updateButtons();
  if (state.follow) await select(state.run.done - 1, { follow: true });
  if (state.run.finished) stopPlaying();
}

function togglePlay() {
  if (state.playing) return stopPlaying();
  state.playing = true;
  state.follow = true;
  $("play").textContent = "Pause";
  (async function loop() {
    while (state.playing && state.run && !state.run.finished) {
      await advance(1);
      await new Promise((r) => setTimeout(r, 700));
    }
    stopPlaying();
  })();
}

function stopPlaying() {
  state.playing = false;
  $("play").textContent = "Play";
  updateButtons();
}

async function runToEnd() {
  stopPlaying();
  $("finish").disabled = true;
  await advance(state.run.batches);
}

function updateButtons() {
  const running = !!state.run && !state.run.finished;
  $("next").disabled = !running || state.playing;
  $("play").disabled = !running;
  $("finish").disabled = !running || state.playing;
  $("reveal").disabled = !state.run || state.run.done === 0;
}

// ---------- timeline ----------

function renderLegend() {
  $("legend").innerHTML = LEGEND.map(([v, t]) =>
    `<span><i style="background:${COLORS[v]}"></i>${t}</span>`).join("") +
    `<span><i style="background:${COLORS.ink2}"></i>Standard check: alarm</span>`;
}

function renderTimeline() {
  const run = state.run;
  const n = run.batches;
  const cols = `repeat(${n}, minmax(0, 1fr))`;
  const word = capital(run.batch_word);
  const s = [], b = [], t = [];
  for (let i = 0; i < n; i++) {
    const r = run.timeline[i];
    if (!r) {
      s.push(`<button class="block empty" disabled tabindex="-1" aria-label="${word} ${i + 1}: not analysed yet"></button>`);
      b.push(`<div class="block empty"></div>`);
    } else {
      const tip = `${word} ${i + 1}: ${r.title}${r.suspect ? ` (${r.suspect})` : ""}`;
      s.push(`<button class="block${i === state.selected ? " selected" : ""}" data-i="${i}" data-v="${r.verdict}"
        title="${esc(tip)}" aria-label="${esc(tip)}" tabindex="${i === state.selected ? 0 : -1}"></button>`);
      const btip = r.baseline ? `Standard check: alarm on ${r.baseline_moved.join(", ")}` : "Standard check: quiet";
      b.push(`<div class="block" data-b="${r.baseline ? 1 : 0}" title="${esc(`${word} ${i + 1}. ${btip}`)}"></div>`);
    }
    const every = n > 30 ? 5 : n > 12 ? 2 : 1;
    t.push(`<span>${(i + 1) % every === 0 || i === 0 ? i + 1 : ""}</span>`);
  }
  for (const [id, html] of [["lane-sentinel", s], ["lane-baseline", b], ["ticks", t]]) {
    $(id).style.gridTemplateColumns = cols;
    $(id).innerHTML = html.join("");
  }
  $("lane-sentinel").querySelectorAll("button[data-i]").forEach((el) =>
    el.addEventListener("click", () => select(Number(el.dataset.i))));
  if (state.truth) renderTruth();
  renderErrorChart();
}

function onTimelineKey(e) {
  if (!state.run || state.selected === null) return;
  const step = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity }[e.key];
  if (step === undefined) return;
  e.preventDefault();
  const i = Math.max(0, Math.min(state.run.done - 1, state.selected + step));
  select(i).then(() => $("lane-sentinel").querySelector(`[data-i="${i}"]`)?.focus());
}

function renderErrorChart() {
  const run = state.run;
  const el = $("err-chart");
  if (!run.timeline.length) { purgeChart(el); return; }
  const x = run.timeline.map((r) => r.index + 1);
  const ref = run.timeline[0].mae_ref;
  drawChart(el, [
    { x: [0.5, run.batches + 0.5], y: [ref, ref], mode: "lines", line: { color: COLORS.muted, dash: "dot", width: 1 },
      hoverinfo: "skip", name: "usual" },
    { x, y: run.timeline.map((r) => r.mae), mode: "lines+markers", line: { color: COLORS.ink, width: 1.5 },
      marker: { size: 4 }, name: "error", hovertemplate: `${capital(run.batch_word)} %{x}: %{y:,.3~r}<extra></extra>` },
  ], {
    margin: { l: 44, r: 4, t: 4, b: 4 }, showlegend: false, height: 70,
    xaxis: { range: [0.5, run.batches + 0.5], showticklabels: false, showgrid: false },
    yaxis: { rangemode: "tozero", nticks: 3, tickfont: { size: 10 } },
  }, { interactive: false });
}

async function reveal() {
  try {
    state.truth = await api(`/runs/${state.run.id}/truth`);
  } catch (e) { return notify(e.message, true); }
  $("truth-lane").hidden = false;
  renderTruth();
  $("reveal").textContent = "Injected changes shown";
}

function renderTruth() {
  const n = state.run.batches;
  const lane = $("lane-truth");
  if (!state.truth.events.length) {
    lane.innerHTML = `<span class="small muted">Nothing was injected. Every alarm in this run is a false alarm.</span>`;
    return;
  }
  // stack overlapping events on separate rows
  lane.style.height = `${1.3 * state.truth.events.length}rem`;
  lane.innerHTML = state.truth.events.map((e, k) => {
    const end = e.end === null ? n - 1 : Math.min(e.end, n - 1);
    const left = (e.start / n) * 100, width = ((end - e.start + 1) / n) * 100;
    const colour = e.kind === "shift" ? COLORS.shift : COLORS.ink;
    return `<div class="truth-bar" style="left:${left}%;width:${width}%;top:${0.2 + 1.3 * k}rem;border-color:${colour}"
      title="${esc(e.text)}">${esc(e.text)}</div>`;
  }).join("");
}

// ---------- selecting a batch ----------

async function select(i, { follow = false } = {}) {
  if (!state.run || i < 0 || i >= state.run.done) return;
  state.selected = i;
  state.follow = follow || i === state.run.done - 1;
  $("lane-sentinel").querySelectorAll(".block").forEach((el) => {
    const on = Number(el.dataset.i) === i;
    el.classList.toggle("selected", on);
    if (el.dataset.i !== undefined) el.tabIndex = on ? 0 : -1;
  });
  let ev = state.cache.get(i);
  if (!ev) {
    try {
      ev = await api(`/runs/${state.run.id}/batches/${i}`);
    } catch (e) { return notify(e.message, true); }
    state.cache.set(i, ev);
  }
  if (state.selected !== i) return; // the user moved on while this was loading
  const keepFeature = state.scatterFeature;
  state.candidate = 0;
  renderStatus(ev);
  renderInvestigation(ev, keepFeature);
}

function renderStatus(ev) {
  const word = capital(state.run.batch_word);
  $("status-where").textContent = `${word} ${ev.index + 1} of ${state.run.batches} · ${ev.rows} rows` +
    (state.follow && !state.run.finished ? " · following the newest" : "");
  const h = $("status-headline");
  h.className = "headline";
  h.textContent = ev.headline;
  $("status-tag").innerHTML = `<span class="verdict-tag"><i style="background:${COLORS[ev.verdict]}"></i>${esc(ev.title)}` +
    ` · confidence ${esc(ev.confidence)}${ev.baseline.alarm ? " · the standard check alarmed" : " · the standard check stayed quiet"}</span>`;
}

// ---------- investigation ----------

function renderInvestigation(ev, keepFeature) {
  const word = capital(state.run.batch_word);
  $("investigation").hidden = false;
  $("inv-title").textContent = `${word} ${ev.index + 1}: ${ev.title}`;
  $("inv-sub").textContent = "What happened → where → why → how strong → what to check";

  $("what").innerHTML = ev.paragraphs.map((p) => `<p>${esc(p)}</p>`).join("");
  const notes = $("notes");
  notes.hidden = !ev.notes.length;
  notes.innerHTML = ev.notes.length ? "Cleaned before checking: " + ev.notes.map(esc).join(" ") : "";

  renderMap(ev);
  renderWhy(ev);
  renderScatter(ev, keepFeature);
  renderStrength(ev);
  renderDrift(ev);
  renderImpact(ev);
  renderChecks(ev);
  renderBatchRows(ev);
}

function renderBatchRows(ev) {
  const columns = ev.record_columns || [];
  const records = ev.records || [];
  const implicated = new Set([
    ...ev.issues.map((issue) => issue.feature),
    ev.suspect,
    ...ev.candidates.map((candidate) => candidate.feature),
  ].filter(Boolean));
  const format = (value) => value === null || value === undefined
    ? "Missing"
    : typeof value === "number" ? String(Number(value.toPrecision(7))) : String(value);
  const header = columns.map((column) => {
    const label = `${column.label}${column.unit ? ` (${column.unit})` : ""}${column.target ? " · target" : ""}`;
    return `<th class="${implicated.has(column.name) ? "focus" : ""}">${esc(label)}</th>`;
  }).join("");
  const rows = records.map((record) => `<tr><th scope="row">${record.index}</th>${record.values.map((value, i) =>
    `<td class="${implicated.has(columns[i].name) ? "focus" : ""}${value === null ? " missing" : ""}">${esc(format(value))}</td>`).join("")}</tr>`).join("");
  $("batch-rows").innerHTML = `<p class="small muted">${records.length} rows</p><table class="data record-table">
    <thead><tr><th>Row</th>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderMap(ev) {
  const weak = state.profile.relationships.filter((r) => !r.usable && r.parents.length)
    .map((r) => ({ target: r.target, parents: r.parents, weak: true }));
  const rules = ev.rules.map((r) => ({ target: r.target, parents: r.parents, skipped: r.skipped,
    broken: r.shift_broken || r.spread_broken }));
  const moved = (ev.drift || []).filter((d) => d.moved).map((d) => d.feature);
  const candidates = ev.verdict === "unclear" ? ev.candidates.slice(0, 3).map((c) => c.feature) : [];
  drawMap($("map"), state.profile.features, [...weak, ...rules],
    { moved, candidates, masked: ev.masked, suspect: ev.suspect });
}

function renderWhy(ev) {
  const body = $("why-body");
  const hint = $("why-hint");
  const loc = ev.locate;
  if (!loc || !loc.candidates.length) {
    hint.textContent = "When relationships break, this panel compares the pattern of breaks with the pattern each feature's fault would leave.";
    const msg = {
      normal: "Nothing broke, so there is no pattern to match.",
      shift: "Values moved but every relationship held, so there is no fault pattern to match. That is exactly what makes it harmless.",
      quality: "The problem was found by the quality checks, before any relationship was tested.",
      insufficient: "Not enough valid rows to test relationships.",
      unclear: "No relationship broke. The feature that moved has no reliable relationship to check it with.",
    }[ev.verdict] || "No fingerprint to show.";
    body.innerHTML = `<p class="empty-panel">${esc(msg)}</p>`;
    const checked = ev.rules.filter((r) => !r.skipped);
    if (checked.length && ["normal", "shift", "unclear"].includes(ev.verdict)) {
      // still useful: show how close each relationship came to its limit
      hint.textContent = "No fault pattern to match. Instead: how far each relationship moved. Bars below the dashed limit mean the relationship still holds.";
      body.innerHTML += `<div class="chart-frame"><div class="chart" id="why-chart"></div></div>`;
      const zl = state.profile.thresholds.z_limit.value;
      const names = checked.map((r) => wrapRule(r.label));
      const sizes = checked.map((r) => Math.abs(r.z));
      const top = Math.max(zl * 1.2, ...sizes.map((v) => v * 1.1));
      drawChart(document.getElementById("why-chart"), [
        { type: "bar", x: names, y: sizes, name: "Shift score (size)",
          marker: { color: COLORS.normal }, hovertemplate: "%{x}<br>shift score %{y:.1f}<extra></extra>" },
      ], {
        showlegend: false,
        yaxis: { title: { text: "shift score", font: { size: 12 } }, range: [0, top] },
        xaxis: { tickfont: { size: 11 }, tickangle: 0 }, margin: { b: 64, t: 12 },
        // the limit is drawn as a shape across the whole plot, so it works on a category axis
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: zl, y1: zl,
          line: { color: COLORS.broken, dash: "dash", width: 1.5 } }],
        annotations: [{ xref: "paper", x: 1, y: zl, xanchor: "right", yanchor: "bottom", showarrow: false,
          text: `limit ${num(zl, 1)}`, font: { size: 11, color: COLORS.broken } }],
      });
    }
    return;
  }
  const spread = loc.mode === "spread";
  hint.textContent = spread
    ? "Bars: how much noisier each relationship became. Outline: the pattern extra noise on the chosen feature would cause. The better they match, the stronger the case."
    : "Bars: how far each relationship's errors moved. Outline: the pattern a fault on the chosen feature would cause. The better they match, the stronger the case.";
  const cands = loc.candidates.slice(0, 3);
  body.innerHTML = `<div class="segmented" role="group" aria-label="Candidate">${cands.map((c, k) =>
    `<button type="button" data-k="${k}" aria-pressed="${k === state.candidate}">${esc(c.label)} · ${pct(c.explained)}</button>`).join("")}</div>
    <div class="chart-frame"><div class="chart" id="why-chart"></div></div>`;
  body.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    state.candidate = Number(b.dataset.k);
    renderWhy(ev);
  }));
  // "Weight from Length" -> "Weight<br>from Length": short lines keep the axis readable
  const names = loc.rules.map((i) => wrapRule(ev.rules[i].label));
  const c = cands[state.candidate] || cands[0];
  const colour = state.candidate === 0 && ev.verdict === "broken" ? COLORS.broken : COLORS.unclear;
  drawChart(document.getElementById("why-chart"), [
    { type: "bar", x: names, y: loc.observed, name: "Observed", marker: { color: COLORS.ink2 },
      hovertemplate: "%{x}<br>observed %{y:.3~r}<extra></extra>" },
    { type: "bar", x: names, y: c.expected, name: `If ${c.label} were at fault`,
      marker: { color: "rgba(0,0,0,0)", line: { color: colour, width: 2.5 } },
      hovertemplate: "%{x}<br>expected %{y:.3~r}<extra></extra>" },
  ], {
    barmode: "group", bargap: 0.35,
    yaxis: { title: { text: spread ? "extra noise" : "shift in errors", font: { size: 12 } }, zeroline: true, zerolinecolor: COLORS.muted },
    xaxis: { tickfont: { size: 11 }, tickangle: 0 },
    margin: { b: 64 },
  });
}

function renderScatter(ev, keepFeature) {
  const body = $("scatter-body");
  const feats = Object.keys(ev.scatter || {});
  if (!feats.length) {
    body.innerHTML = `<p class="empty-panel">No relationship could be checked in this ${esc(state.run.batch_word)}.</p>`;
    return;
  }
  const preferred = [ev.suspect, ev.candidates[0]?.feature, keepFeature,
    ...(ev.drift || []).filter((d) => d.moved).map((d) => d.feature)];
  const f = preferred.find((x) => x && feats.includes(x)) || feats[0];
  state.scatterFeature = f;
  body.innerHTML = `<div class="field"><label class="small muted" for="scatter-feature">Feature </label>
    <select id="scatter-feature">${feats.map((x) =>
      `<option value="${esc(x)}"${x === f ? " selected" : ""}>${esc(ev.scatter[x].label)}</option>`).join("")}</select></div>
    <div class="chart-frame"><div class="chart" id="scatter-chart"></div></div>`;
  $("scatter-feature").addEventListener("change", (e) => {
    state.scatterFeature = e.target.value;
    drawScatter(ev, e.target.value);
  });
  drawScatter(ev, f);
}

function drawScatter(ev, f) {
  const s = ev.scatter[f];
  const unit = s.unit ? ` (${s.unit})` : "";
  const all = [...s.ref.x, ...s.ref.y, ...s.now.x, ...s.now.y].filter((v) => v !== null && (!s.log || v > 0));
  const lo = Math.min(...all), hi = Math.max(...all);
  const colour = COLORS[ev.verdict] === COLORS.normal ? COLORS.ink : COLORS[ev.verdict];
  const type = s.log ? "log" : "linear";
  // Plotly's automatic log ticks turn into a crowd of digits when the range is under one decade,
  // so log axes get five round, evenly spaced (on the log scale) ticks instead.
  const ticks = s.log ? logTicks(lo, hi) : {};
  drawChart($("scatter-chart"), [
    { x: s.ref.x, y: s.ref.y, mode: "markers", name: "Healthy reference",
      marker: { color: COLORS.normal, size: 4, opacity: 0.45 }, hoverinfo: "skip" },
    { x: [lo, hi], y: [lo, hi], mode: "lines", name: "Agrees exactly", line: { color: COLORS.muted, dash: "dash", width: 1 },
      hoverinfo: "skip" },
    { x: s.now.x, y: s.now.y, mode: "markers", name: `This ${state.run.batch_word}`,
      marker: { color: colour, size: 5, opacity: 0.85 },
      hovertemplate: `expected %{x:.3~r}<br>actual %{y:.3~r}<extra></extra>` },
  ], {
    xaxis: { title: { text: `${s.label} expected from its neighbours${unit}`, font: { size: 12 } }, type, ...ticks },
    yaxis: { title: { text: `${s.label} as recorded${unit}`, font: { size: 12 } }, type, ...ticks },
  });
}

// "Depth % from Table %, Depth and Width" -> one short line per name, so axis labels never collide
function wrapRule(label) {
  return label.replace(" from ", "<br>from ").replace(/, /g, ",<br>").replace(/ and /g, "<br>and ");
}

function logTicks(lo, hi) {
  const values = [];
  for (let k = 0; k <= 4; k++) {
    const v = Math.exp(Math.log(lo) + (k / 4) * (Math.log(hi) - Math.log(lo)));
    values.push(Number(v.toPrecision(v >= 1 ? 2 : 1)));
  }
  const unique = [...new Set(values)];
  return { tickvals: unique, ticktext: unique.map((v) => String(v)), minor: { showgrid: false } };
}

function stateCell(kind, text) {
  return `<span class="state ${kind}"><i></i>${esc(text)}</span>`;
}

function renderStrength(ev) {
  const t = state.profile.thresholds;
  const zl = t.z_limit.value, sl = t.spread_limit.value;
  const rows = ev.rules.map((r) => {
    const status = r.skipped ? stateCell("off", "not checked")
      : r.shift_broken ? stateCell("bad", "shifted")
      : r.spread_broken ? stateCell("bad", "noisier") : stateCell("", "holds");
    return `<tr><td>${esc(r.label)}</td>
      <td class="num">${r.skipped ? "–" : num(r.z, 1)}</td>
      <td class="num">${r.skipped ? "–" : num(r.spread, 2)}</td><td>${status}</td></tr>`;
  }).join("");
  $("strength").innerHTML = `<table class="data"><thead><tr><th>Relationship</th>
    <th class="num" title="Limit ±${num(zl, 1)}">Shift score <span class="muted">(limit ±${num(zl, 1)})</span></th>
    <th class="num">Scatter <span class="muted">(limit ${num(sl, 2)})</span></th><th>State</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="small muted" style="margin-top:.5rem">Confidence ${esc(ev.confidence)}.${
      ev.candidates.length > 1 && ev.candidates[0].explained !== null
        ? ` Best explanation ${esc(ev.candidates[0].label)} (${pct(ev.candidates[0].explained)}), runner-up ${esc(ev.candidates[1].label)} (${pct(ev.candidates[1].explained)}).` : ""}</p>`;
}

function renderDrift(ev) {
  const byUnit = Object.fromEntries(state.profile.features.map((f) => [f.name, f.unit]));
  const rows = (ev.drift || []).map((d) => {
    const u = byUnit[d.feature];
    const fmt = (v) => v === null ? "–" : u === "$" ? `$${num(v)}` : `${num(v)}${u ? (u === "%" ? "%" : " " + u) : ""}`;
    return `<tr><td>${esc(d.label)}</td><td class="num">${fmt(d.usual)}</td><td class="num">${fmt(d.now)}</td>
      <td class="num">${num(d.ks, 3)}</td><td class="num">${num(d.limit, 3)}</td>
      <td>${d.moved ? stateCell("moved", "moved") : stateCell("", "as usual")}</td></tr>`;
  }).join("");
  const masked = ev.masked.length
    ? `<p class="small muted">Left out (invalid data): ${ev.masked.map((m) => esc(state.profile.features.find((f) => f.name === m)?.label || m)).join(", ")}.</p>` : "";
  const b = ev.baseline;
  const label = (n) => state.profile.features.find((f) => f.name === n)?.label || n;
  const verdict = b.alarm
    ? `The standard check would raise an alarm on ${b.moved.length} feature(s) and blame <strong>${esc(label(b.blamed))}</strong>.`
    : "The standard check would stay quiet.";
  $("drift").innerHTML = `<table class="data"><thead><tr><th>Feature</th><th class="num">Usual median</th><th class="num">Now</th>
    <th class="num">KS</th><th class="num">Limit</th><th>State</th></tr></thead><tbody>${rows}</tbody></table>
    ${masked}<p class="small" style="margin-top:.5rem">${verdict}</p>`;
}

function renderImpact(ev) {
  const el = $("impact");
  const unit = state.profile.target_unit;
  const money = (v) => unit === "$" ? `$${num(v)}` : `${num(v)}${unit ? " " + unit : ""}`;
  const target = state.profile.target_label;
  const known = ev.mae !== null
    ? `<p class="small muted">Known later, when true ${esc(target.toLowerCase())} values arrive: the model's error on this ${esc(state.run.batch_word)} was ${money(ev.mae)} (usually ${money(ev.mae_ref)}).</p>` : "";
  const imp = ev.impact;
  if (!imp || !imp.available) {
    const why = imp ? `${esc(state.profile.features.find((f) => f.name === imp.feature)?.label)} has no reliable relationship to repair it with, so the impact cannot be estimated.`
      : "Estimated only when a relationship breaks: the suspect is repaired using its relationship and the model is run again.";
    el.innerHTML = `<p class="empty-panel">${why}</p>${known}`;
    return;
  }
  const fl = state.profile.features.find((f) => f.name === imp.feature)?.label;
  el.innerHTML = `<p>Repairing <strong>${esc(fl)}</strong> changes ${esc(target.toLowerCase())} predictions by
    <strong>${money(imp.mean_change)}</strong> on average: <strong>${num(imp.ratio, 2)}×</strong> the usual error
    (healthy ${esc(state.profile.batches_word)} stay below ${num(imp.null_p95, 2)}×). Impact is <strong>${imp.material ? "material" : "low"}</strong>.</p>
    <div class="chart-frame"><div class="chart" id="impact-chart"></div></div>
    ${imp.mae_raw !== undefined ? `<p class="small muted">Check once true values are known: error with the recorded ${esc(fl)} ${money(imp.mae_raw)},
      with the repaired value ${money(imp.mae_repaired)}.</p>` : known}`;
  const s = imp.series;
  const order = s.actual.map((v, i) => i).filter((i) => s.actual[i] !== null).sort((a, b) => s.actual[a] - s.actual[b]);
  const pick = (arr) => order.map((i) => arr[i]);
  const x = order.map((_, k) => k + 1);
  drawChart($("impact-chart"), [
    { x, y: pick(s.actual), mode: "lines", name: "True value", line: { color: COLORS.muted, width: 1.5 } },
    { x, y: pick(s.raw), mode: "markers", name: "Predicted from recorded data", marker: { color: COLORS[ev.verdict], size: 5 } },
    { x, y: pick(s.repaired), mode: "markers", name: "Predicted after repair", marker: { color: COLORS.ink, size: 5, symbol: "circle-open" } },
  ], {
    xaxis: { title: { text: `rows of this ${state.run.batch_word}, sorted by true value`, font: { size: 12 } } },
    yaxis: { title: { text: `${target}${unit ? ` (${unit})` : ""}`, font: { size: 12 } } },
  });
}

function renderChecks(ev) {
  const list = ev.checks.length
    ? `<ol>${ev.checks.map((c) => `<li>${esc(c)}</li>`).join("")}</ol>`
    : `<p class="empty-panel">Nothing to check.</p>`;
  const mind = ev.change_mind ? `<p class="change-mind"><strong>What would change this conclusion:</strong> ${esc(ev.change_mind)}</p>` : "";
  $("checks").innerHTML = list + mind;
}

// ---------- live injection ----------

function updateAmountField() {
  const fault = state.catalogue?.faults[$("inj-fault").value];
  if (!fault) return;
  $("inj-amount-field").hidden = fault.amount === null;
  $("inj-amount-label").textContent = fault.amount || "Amount";
  $("inj-amount").value = fault.default ?? "";
}

async function injectFault(e) {
  e.preventDefault();
  if (!state.run) return;
  const amount = $("inj-amount").value === "" ? null : Number($("inj-amount").value);
  await addEvent({ kind: "fault", feature: $("inj-feature").value, fault: $("inj-fault").value, amount });
}

async function injectShift() {
  if (state.run) await addEvent({ kind: "shift", strength: 1.0 });
}

async function addEvent(body) {
  if (state.run.finished) return notify("This run has finished. Start a new one to inject changes.", true);
  try {
    const res = await api(`/runs/${state.run.id}/events`, { method: "POST", body });
    state.run = { ...state.run, ...res };
    notify(`Added: ${res.added}. It takes effect from the next ${state.run.batch_word}. The system is not told about it.`);
    if (state.truth) { state.truth = await api(`/runs/${state.run.id}/truth`); renderTruth(); }
  } catch (err) {
    notify(err.message, true);
  }
}
