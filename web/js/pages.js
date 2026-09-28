// Relationships, Evaluation and Your data pages.
import { api, esc, num, pct } from "./api.js";
import { COLORS, drawChart, drawMap } from "./charts.js";

const $ = (id) => document.getElementById(id);
const money = (p, v) => p.target_unit === "$" ? `$${num(v)}` : `${num(v)}${p.target_unit ? " " + p.target_unit : ""}`;

// ================= Relationships =================

export function renderRelationships(p) {
  const usable = p.relationships.filter((r) => r.usable);
  $("rel-intro").innerHTML = `${esc(p.description)} The model predicts <strong>${esc(p.target_label)}</strong> with an average
    error of ${money(p, p.model.mae)} (guessing the average every time would be off by ${money(p, p.model.mae_naive)}).
    From ${p.sizes.train.toLocaleString()} healthy rows the system learned ${usable.length} usable relationship${usable.length === 1 ? "" : "s"}
    between ${p.features.length} monitored features, then calibrated its limits on ${p.sizes.calib.toLocaleString()} more.
    ${p.sizes.stream.toLocaleString()} rows are kept aside to replay as ${esc(p.batches_word)}.`;

  $("rule-list").innerHTML = p.relationships.map((r) => {
    if (!r.parents.length) {
      return `<li class="weak"><strong>${esc(r.label)}</strong>: no other feature predicts it. It can only be checked on its own.</li>`;
    }
    const fit = `fit ${pct(r.r2)}`;
    const head = `<strong>${esc(r.label)}</strong> from ${esc(r.parent_labels.join(", "))} <span class="muted small">· ${fit}</span>`;
    if (!r.usable) {
      return `<li class="weak">${head}<div class="small">Too weak to trust, not used.</div><div class="eq">${esc(r.linear)}</div></li>`;
    }
    return `<li>${head}${r.plain ? `<div class="plain">${esc(r.plain).replace(/\^([\d.\-]+)/g, "<sup>$1</sup>")}</div>` : ""}<div class="eq">${esc(r.linear)}</div></li>`;
  }).join("");

  const c = p.cleaning;
  const row = (name, x) => `<tr><td>${name}</td><td class="num">${x.rows_in.toLocaleString()}</td><td class="num">${x.missing}</td>
    <td class="num">${x.impossible}</td><td class="num">${x.extreme}</td><td class="num">${x.rows_out.toLocaleString()}</td></tr>`;
  $("cleaning").innerHTML = `<p class="hint">Relationships must be learned from clean data, so obviously bad rows are removed first:
    missing values, impossible values (zero or negative sizes on a log scale) and the most extreme 0.1% at each end.</p>
    <div class="table-scroll"><table class="data"><thead><tr><th>Part</th><th class="num">Rows</th><th class="num">Missing</th>
    <th class="num">Impossible</th><th class="num">Extreme</th><th class="num">Kept</th></tr></thead>
    <tbody>${row("Learning", c.train)}${row("Calibration", c.calib)}</tbody></table></div>`;

  drawMap($("rel-map"), p.features, p.relationships.filter((r) => r.parents.length)
    .map((r) => ({ target: r.target, parents: r.parents, weak: !r.usable })), { plain: true });

  const labels = p.features.map((f) => f.label);
  drawChart($("iso-chart"), [{
    type: "heatmap", z: p.isolability, x: labels, y: labels, zmin: 0, zmax: 1,
    colorscale: [[0, "#ffffff"], [0.7, "#d9dee4"], [0.95, COLORS.unclear], [1, COLORS.broken]],
    colorbar: { thickness: 10, len: 0.8, tickfont: { size: 11 } },
    text: p.isolability.map((r) => r.map((v) => v.toFixed(2))), texttemplate: "%{text}", textfont: { size: 11 },
    hovertemplate: "%{y} vs %{x}: %{z:.2f}<extra></extra>",
  }], { xaxis: { tickangle: -30, showgrid: false }, yaxis: { autorange: "reversed", showgrid: false },
    margin: { l: 80, r: 10, t: 8, b: 70 }, showlegend: false });

  const notes = [];
  for (const pair of p.same_pattern_pairs) {
    notes.push(`<strong>${esc(pair.a)}</strong> and <strong>${esc(pair.b)}</strong> leave the same fingerprint
      (${pair.cos.toFixed(2)}): a fault on either will be reported as unclear between the two.`);
  }
  if (p.blind_spots.length) {
    notes.push(`<strong>${esc(p.blind_spots.join(", "))}</strong> ${p.blind_spots.length === 1 ? "has" : "have"} no usable relationship
      of ${p.blind_spots.length === 1 ? "its" : "their"} own. If ${p.blind_spots.length === 1 ? "it moves" : "they move"}, the system
      cannot vouch that the change is harmless and says so.`);
  }
  $("iso-notes").innerHTML = notes.length ? `<p>Known limits for this dataset:</p><ul>${notes.map((n) => `<li>${n}</li>`).join("")}</ul>`
    : "<p>Every feature leaves a distinct fingerprint.</p>";

  const tr = Object.entries(p.thresholds).map(([k, t]) => {
    const v = k.endsWith("share") || k === "min_r2" || k === "explained_min" || k === "margin_min" || k === "ks_percentile"
      ? (k === "ks_percentile" ? `${(t.value * 100).toFixed(2)}th pct` : pct(t.value)) : num(t.value, 2);
    return `<tr><td>${esc(t.meaning)}</td><td class="num">${v}</td><td>${esc(t.source)}</td></tr>`;
  }).join("");
  const ks = Object.entries(p.ks_limits).map(([f, v]) => `${esc(f)} ${num(v, 3)}`).join(" · ");
  $("thresholds").innerHTML = `<table class="data"><thead><tr><th>What it decides</th><th class="num">Value</th><th>Where it comes from</th></tr></thead>
    <tbody>${tr}<tr><td>Drift limit per feature (KS)</td><td class="num" colspan="2" style="text-align:left">${ks}</td></tr></tbody></table>
    <p class="small muted" style="margin-top:.5rem">Profile built in ${num(p.build_seconds, 1)} s.</p>`;
}

// ================= Evaluation =================

export function initEvaluation(getDataset) {
  $("run-eval").addEventListener("click", () => runEvaluation(getDataset(), false));
}

export async function renderEvaluation(dataset) {
  $("eval-table").hidden = true;
  $("eval-summary").innerHTML = "";
  $("eval-status").textContent = "";
  // show cached results straight away if they exist; otherwise wait for the button
  const ready = (await api("/datasets")).find((d) => d.name === dataset);
  if (ready?.evaluated) await runEvaluation(dataset, false);
}

async function runEvaluation(dataset, fresh) {
  const button = $("run-eval");
  button.disabled = true;
  $("eval-status").innerHTML = `<span class="busy">Simulating a few hundred batches… this takes about a minute the first time.</span>`;
  try {
    const res = await api(`/datasets/${dataset}/evaluate${fresh ? "?fresh=true" : ""}`, { method: "POST" });
    drawEvaluation(res);
    $("eval-status").textContent = `${res.batches_per_row} batches per row, seed ${res.seed}. Results are cached; rebuild the dataset to recompute.`;
  } catch (e) {
    $("eval-status").textContent = e.message;
  } finally {
    button.disabled = false;
  }
}

function rateCell(r, better) {
  const ci = `${Math.round(r.ci[0] * 100)}–${Math.round(r.ci[1] * 100)}%`;
  return `<td class="num"><span class="rate${better ? " better" : ""}">${pct(r.rate)}</span> <span class="small muted">${r.k}/${r.n}</span>
    <div class="ci">${ci}</div></td>`;
}

function drawEvaluation(res) {
  let html = `<table class="data eval-table"><thead><tr><th>Situation</th><th>Question</th><th class="num">Sentinel</th>
    <th class="num">Standard check</th><th>Note</th></tr></thead><tbody>`;
  let group = null;
  for (const r of res.rows) {
    if (r.group !== group) {
      group = r.group;
      html += `<tr class="group"><td colspan="5">${esc(group)}</td></tr>`;
    }
    const lowerBetter = /lower is better/.test(r.question);
    const sBetter = lowerBetter ? r.sentinel.rate < r.baseline.rate : r.sentinel.rate > r.baseline.rate;
    const bBetter = lowerBetter ? r.baseline.rate < r.sentinel.rate : r.baseline.rate > r.sentinel.rate;
    html += `<tr><td>${esc(r.situation)}</td><td>${esc(r.question)}</td>${rateCell(r.sentinel, sBetter)}${rateCell(r.baseline, bBetter)}
      <td class="small muted">${esc(r.note || "")}</td></tr>`;
  }
  $("eval-table").innerHTML = html + "</tbody></table>";
  $("eval-table").hidden = false;

  // one honest summary sentence per group
  const rows = res.rows;
  const shift = rows.filter((r) => r.group === "Harmless shift");
  const faults = rows.filter((r) => r.group === "Small fault" && r.question === "Caught");
  const blamed = rows.filter((r) => r.group === "Small fault" && r.question === "Blamed the right feature");
  const healthy = rows.find((r) => r.group === "Healthy data");
  const avg = (xs, key) => xs.reduce((s, r) => s + r[key].rate, 0) / (xs.length || 1);
  const weakBlame = blamed.filter((r) => r.sentinel.rate < 0.5).map((r) => rows[rows.indexOf(r) - 1].situation.split(" reads")[0].split(" gets")[0]);
  $("eval-summary").innerHTML = `<p>On ordinary data both methods raised a false alarm in ${pct(healthy.sentinel.rate)} vs ${pct(healthy.baseline.rate)} of batches.
    Under a harmless shift the standard check alarmed on ${pct(avg(shift, "baseline"))} of batches on average; Sentinel on ${pct(avg(shift, "sentinel"))}.
    Small faults were caught in ${pct(avg(faults, "sentinel"))} of batches by Sentinel and ${pct(avg(faults, "baseline"))} by the standard check,
    and Sentinel named the right feature in ${pct(avg(blamed, "sentinel"))} (standard check: ${pct(avg(blamed, "baseline"))}).</p>
    ${weakBlame.length ? `<p>Where it is weak: ${esc(weakBlame.length > 1 ? weakBlame.slice(0, -1).join(", ") + " and " + weakBlame.at(-1) : weakBlame[0])} ${weakBlame.length > 1 ? "are" : "is"} usually reported as <em>unclear</em> rather than named,
    because another feature leaves almost the same fingerprint (see Relationships). Saying "unclear" there is the intended behaviour, not a crash.</p>` : ""}`;
}

// ================= Your data =================

let onUse = () => {};

export function initData({ use, notice }) {
  onUse = use;
  $("upload").addEventListener("change", (e) => upload(e.target.files[0], notice));
}

export async function renderData(currentDataset, notice) {
  const list = await api("/datasets");
  $("dataset-list").innerHTML = list.map((d) => `<div class="dataset-row">
      <div class="grow"><strong>${esc(d.title)}</strong> <span class="small muted">${d.source === "uploaded" ? "uploaded" : "example"}${d.name === currentDataset ? " · in use" : ""}</span>
        <div class="small">${esc(d.description || "")}</div>
        <div class="small muted">Predicts ${esc(d.target)} · monitors ${esc(d.monitored.join(", "))}</div></div>
      <div>${d.ready
        ? `<button class="btn quiet" data-rebuild="${esc(d.name)}">Rebuild</button> <button class="btn" data-use="${esc(d.name)}"${d.name === currentDataset ? " disabled" : ""}>Use</button>`
        : `<button class="btn" data-build="${esc(d.name)}">Set up</button>`}</div></div>`).join("");
  $("dataset-list").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    const name = b.dataset.use || b.dataset.build || b.dataset.rebuild;
    if (b.dataset.use) return onUse(name);
    b.disabled = true;
    b.innerHTML = `<span class="busy">Learning…</span>`;
    try {
      await api(`/datasets/${name}/build${b.dataset.rebuild ? "?rebuild=true" : ""}`, { method: "POST" });
      notice(`${name} is ready.`);
      await onUse(name, { rebuilt: true });
    } catch (e) {
      notice(e.message, true);
      renderData(currentDataset, notice);
    }
  }));
}

async function upload(file, notice) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  $("upload-form").innerHTML = `<p class="busy small">Reading ${esc(file.name)}…</p>`;
  let s;
  try {
    s = await api("/uploads", { method: "POST", form });
  } catch (e) {
    $("upload-form").innerHTML = "";
    return notice(e.message, true);
  }
  const numeric = s.columns.filter((c) => c.numeric);
  const role = (c) => c.name === s.target ? "target" : s.monitored.includes(c.name) ? "monitor"
    : s.context.includes(c.name) ? "input" : c.name === s.time_column ? "time" : "ignore";
  const rows = s.columns.map((c) => {
    const r = role(c);
    const opts = [["monitor", "Monitor", c.numeric], ["input", "Model input only", true], ["target", "Target (to predict)", c.numeric],
      ["time", "Time order", !c.numeric], ["ignore", "Ignore", true]]
      .filter(([, , ok]) => ok).map(([v, t]) => `<option value="${v}"${v === r ? " selected" : ""}>${t}</option>`).join("");
    return `<tr><td>${esc(c.name)}</td><td class="small muted">${c.numeric ? "number" : "text"} · ${c.unique} distinct${c.missing ? ` · ${pct(c.missing)} missing` : ""}</td>
      <td><select data-col="${esc(c.name)}">${opts}</select></td>
      <td>${c.numeric ? `<label class="small"><input type="checkbox" data-log="${esc(c.name)}"${s.log_features.includes(c.name) ? " checked" : ""}> log scale</label>` : ""}</td></tr>`;
  }).join("");
  $("upload-form").innerHTML = `<p class="small">${s.rows.toLocaleString()} rows. Check the roles below. Monitored columns should be
    related to each other; use a log scale for sizes, weights and amounts that only grow multiplicatively.</p>
    <div class="table-scroll"><table class="data col-table"><thead><tr><th>Column</th><th></th><th>Role</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="form-grid">
      <label>Title<input type="text" id="up-title" value="${esc(s.title)}"></label>
      <label>Rows per batch<input type="number" id="up-batch" min="30" max="2000" value="${s.batch_size}"></label>
      <label>A batch is called<input type="text" id="up-word" value="batch"></label>
      <label>Harmless shift favours rows with high<select id="up-driver">${numeric.map((c) =>
        `<option${c.name === s.shift_driver ? " selected" : ""}>${esc(c.name)}</option>`).join("")}</select></label>
    </div>
    <button class="btn" id="up-build">Learn from this data</button> <span class="small muted" id="up-status"></span>`;
  $("up-build").addEventListener("click", () => buildUpload(s, notice));
}

async function buildUpload(s, notice) {
  const roles = {};
  document.querySelectorAll("#upload-form select[data-col]").forEach((el) => { roles[el.dataset.col] = el.value; });
  const pick = (r) => Object.keys(roles).filter((c) => roles[c] === r);
  const targets = pick("target");
  if (targets.length !== 1) return notice("Choose exactly one target column.", true);
  const monitored = pick("monitor");
  const body = {
    name: s.name, title: $("up-title").value.trim() || s.name, target: targets[0], monitored,
    context: pick("input"), time_column: pick("time")[0] || null,
    log_features: [...document.querySelectorAll("#upload-form input[data-log]:checked")].map((el) => el.dataset.log).filter((c) => monitored.includes(c)),
    batch_size: Number($("up-batch").value) || s.batch_size,
    shift_driver: monitored.includes($("up-driver").value) ? $("up-driver").value : monitored[0],
    batch_word: $("up-word").value.trim() || "batch",
  };
  const button = $("up-build");
  button.disabled = true;
  $("up-status").innerHTML = `<span class="busy">Learning relationships and calibrating limits (under a minute for most files)…</span>`;
  try {
    await api(`/uploads/${s.name}/config`, { method: "POST", body });
    $("upload-form").innerHTML = "";
    $("upload").value = "";
    notice(`${body.title} is ready. See Relationships for what was learned.`);
    await onUse(s.name, { rebuilt: true, page: "relationships" });
  } catch (e) {
    $("up-status").textContent = "";
    button.disabled = false;
    notice(e.message, true);
  }
}
