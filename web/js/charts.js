// Charts (Plotly) and the relationship map (plain SVG).
import { esc } from "./api.js";

export const COLORS = {
  normal: "#a3acb6", shift: "#2f6db3", quality: "#7a4da6", broken: "#c03a2b",
  unclear: "#c58a0d", insufficient: "#cfd5db",
  ink: "#1e2a36", ink2: "#3a4856", muted: "#5e6b78", rule: "#d3d9df", soft: "#e6eaee",
};

const BASE_LAYOUT = {
  font: { family: "Source Sans 3, Segoe UI, system-ui, sans-serif", size: 12, color: COLORS.ink },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 56, r: 12, t: 34, b: 44 },
  xaxis: { gridcolor: COLORS.soft, zeroline: false, linecolor: COLORS.rule, automargin: true },
  yaxis: { gridcolor: COLORS.soft, zeroline: false, linecolor: COLORS.rule, automargin: true },
  // legend above the plot, so it never collides with axis titles or long tick labels
  legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 12 } },
  hoverlabel: { font: { family: "Source Sans 3, system-ui, sans-serif" } },
  showlegend: true,
};

const CONFIG = {
  displaylogo: false,
  responsive: true,
  doubleClick: "reset",           // double-click returns to the original view
  scrollZoom: false,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
    "hoverClosestCartesian", "hoverCompareCartesian"],
};

function merge(base, extra) {
  const out = { ...base };
  for (const [k, v] of Object.entries(extra || {})) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) && base[k] && typeof base[k] === "object"
      ? merge(base[k], v) : v;
  }
  return out;
}

/*
 * Zoom fix.
 * The old bug: a chart was updated in place, so Plotly's idea of the "original"
 * view went stale and zooming out could not get back to it.
 * Now every draw (1) purges the old plot, (2) draws fresh, (3) stores the exact
 * axis ranges it started with. Double-click, the toolbar's home button and our
 * own "Reset zoom" button (shown only while zoomed in) all return to that view.
 */
export async function drawChart(el, traces, layout = {}, { interactive = true } = {}) {
  if (!el) return;
  Plotly.purge(el);
  const fullLayout = merge(BASE_LAYOUT, layout);
  if (!interactive) {
    fullLayout.xaxis = { ...fullLayout.xaxis, fixedrange: true };
    fullLayout.yaxis = { ...fullLayout.yaxis, fixedrange: true };
  }
  await Plotly.newPlot(el, traces, fullLayout, interactive ? CONFIG : { ...CONFIG, displayModeBar: false });
  if (!interactive) return;

  const home = {};
  for (const ax of ["xaxis", "yaxis"]) {
    const range = el._fullLayout[ax]?.range;
    if (range) home[`${ax}.range`] = range.slice();
  }
  el._home = home;

  const frame = el.parentElement;
  let button = frame.querySelector(":scope > .reset-zoom");
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.className = "reset-zoom";
    button.textContent = "Reset zoom";
    frame.appendChild(button);
  }
  button.hidden = true;
  button.onclick = () => Plotly.relayout(el, { ...el._home });
  el.on("plotly_relayout", () => {
    const zoomed = Object.entries(el._home).some(([key, start]) => {
      const now = el._fullLayout[key.split(".")[0]].range;
      return Math.abs(now[0] - start[0]) > 1e-9 || Math.abs(now[1] - start[1]) > 1e-9;
    });
    button.hidden = !zoomed;
  });
}

export function resizeCharts(root) {
  root.querySelectorAll(".js-plotly-plot").forEach((el) => {
    if (el.offsetParent !== null) Plotly.Plots.resize(el);
  });
}

/*
 * Relationship map: features on a circle, one arrow per rule (from each input
 * feature to the feature it predicts). Colours follow the verdict colours.
 */
export function drawMap(el, features, rules, state = {}) {
  const W = 440, H = 330, cx = W / 2, cy = H / 2 + 4, R = Math.min(W, H) / 2 - 58, r = 17;
  const n = features.length;
  const pos = {};
  features.forEach((f, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
    pos[f.name] = { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a };
  });
  const label = Object.fromEntries(features.map((f) => [f.name, f.label]));
  const uid = el.id || "map";   // marker ids must be unique per page, or hidden maps swallow the arrowheads
  const edgeColour = { holds: "#8a96a3", broken: COLORS.broken, skipped: COLORS.rule, weak: COLORS.soft };

  let edges = "";
  for (const rule of rules) {
    const kind = rule.broken ? "broken" : rule.skipped ? "skipped" : rule.weak ? "weak" : "holds";
    for (const p of rule.parents) {
      const a = pos[p], b = pos[rule.target];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
      const ux = dx / d, uy = dy / d;
      // bend each arrow to its right, so A->B and B->A do not overlap
      const mx = (a.x + b.x) / 2 - uy * d * 0.12, my = (a.y + b.y) / 2 + ux * d * 0.12;
      const sx = a.x + ux * r, sy = a.y + uy * r;
      const ex = b.x - ux * (r + 4), ey = b.y - uy * (r + 4);
      const title = `${label[rule.target]} predicted from ${rule.parents.map((x) => label[x]).join(", ")}: ${
        { holds: "holds", broken: "broken", skipped: "not checked", weak: "too weak to use" }[kind]}`;
      edges += `<path d="M${sx.toFixed(1)},${sy.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}"
        fill="none" stroke="${edgeColour[kind]}" stroke-width="${kind === "broken" ? 3 : 1.4}"
        ${kind === "skipped" || kind === "weak" ? 'stroke-dasharray="4 4"' : ""} marker-end="url(#${uid}-arrow-${kind})"><title>${esc(title)}</title></path>`;
    }
  }

  let nodes = "";
  for (const f of features) {
    const { x, y, a } = pos[f.name];
    let fill = "#fff", stroke = COLORS.ink, width = 1.5, dash = "";
    if (state.moved?.includes(f.name)) { stroke = COLORS.shift; width = 3; }
    if (state.candidates?.includes(f.name)) { stroke = COLORS.unclear; width = 3.5; }
    if (state.masked?.includes(f.name)) { stroke = COLORS.quality; width = 3; dash = 'stroke-dasharray="4 3"'; fill = "#f4effa"; }
    if (state.suspect === f.name) { fill = COLORS.broken; stroke = COLORS.broken; }
    if (!f.has_rule && !state.plain) dash = dash || "";
    const lx = x + Math.cos(a) * (r + 10), ly = y + Math.sin(a) * (r + 10);
    const anchor = Math.abs(Math.cos(a)) < 0.3 ? "middle" : Math.cos(a) > 0 ? "start" : "end";
    const baseline = Math.sin(a) > 0.3 ? "hanging" : Math.sin(a) < -0.3 ? "auto" : "middle";
    nodes += `<g><title>${esc(f.label)}${f.has_rule ? "" : " (no reliable rule of its own)"}</title>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${width}" ${dash}/>
      <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="${baseline}"
        font-size="13" font-family="Source Sans 3, system-ui, sans-serif" fill="${COLORS.ink}">${esc(f.label)}</text></g>`;
  }

  const marker = (id, colour) => `<marker id="${uid}-arrow-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${colour}"/></marker>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Map of learned relationships">
    <defs>${Object.entries(edgeColour).map(([k, c]) => marker(k, c)).join("")}</defs>${edges}${nodes}</svg>`;
}
