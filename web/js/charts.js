// Charts (Plotly) and the relationship map (plain SVG).
import { esc } from "./api.js";

export const COLORS = {
  normal: "#a3acb6", shift: "#2f6db3", quality: "#7a4da6", broken: "#c03a2b",
  unclear: "#c58a0d", insufficient: "#cfd5db",
  ink: "#1e2a36", ink2: "#3a4856", muted: "#5e6b78", rule: "#d3d9df", soft: "#e6eaee",
};

// Family names must be quoted: unquoted "Source Sans 3" is invalid CSS (a name part starts with a digit),
// the browser then drops the whole font stack and Plotly's text falls back to wide Verdana, which
// overflows the legend box Plotly sized for the narrower font (the cut-off legend entries).
const FONT = '"Source Sans 3", "Segoe UI", system-ui, sans-serif';

const BASE_LAYOUT = {
  font: { family: FONT, size: 12, color: COLORS.ink },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 56, r: 12, t: 34, b: 44 },
  xaxis: { gridcolor: COLORS.soft, zeroline: false, linecolor: COLORS.rule, automargin: true },
  yaxis: { gridcolor: COLORS.soft, zeroline: false, linecolor: COLORS.rule, automargin: true },
  // legend above the plot, so it never collides with axis titles or long tick labels
  legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 12 } },
  hoverlabel: { font: { family: FONT } },
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
 *
 * Layout rules, so nothing overlaps whatever the number of features:
 * - the ring grows with the number of features, so neighbouring nodes never touch;
 * - labels sit radially outside the ring (arrows stay inside it), wrap onto two
 *   lines when long, and carry a white halo where an arrow does pass behind them;
 * - arrows start and end on the node boundary, following the curve's own direction,
 *   so arrowheads are never hidden under a node;
 * - a single arrow bends slightly towards the centre; when A->B and B->A both exist,
 *   each bends to its own right, so the pair always forms two separate arcs;
 * - the drawing area is computed from the nodes and labels, plus padding.
 */
const MAP_FONT = 13, MAP_LINE = 15, MAP_WRAP = 14;

function wrapLabel(text) {
  if (text.length <= MAP_WRAP) return [text];
  const words = text.split(" ");
  let best = [text], score = Infinity;
  for (let k = 1; k < words.length; k++) {  // the break that gives the most even two lines
    const a = words.slice(0, k).join(" "), b = words.slice(k).join(" ");
    const s = Math.max(a.length, b.length);
    if (s < score) { score = s; best = [a, b]; }
  }
  return best;
}

export function drawMap(el, features, rules, state = {}) {
  const n = features.length;
  const r = n > 8 ? 15 : 17;
  // ring big enough for a gap of at least 26 between neighbouring nodes
  const R = n < 2 ? 0 : Math.max(n === 2 ? 70 : 92, (2 * r + 26) / (2 * Math.sin(Math.PI / n)));
  const pos = {};
  features.forEach((f, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(n, 1);
    pos[f.name] = { x: R * Math.cos(a), y: R * Math.sin(a), a };
  });
  const label = Object.fromEntries(features.map((f) => [f.name, f.label]));
  const uid = el.id || "map";   // marker ids must be unique per page, or hidden maps swallow the arrowheads
  const edgeColour = { holds: "#8a96a3", broken: COLORS.broken, skipped: COLORS.rule, weak: "#c3cad2" };
  const strokeOf = {}, widthOf = {};
  for (const f of features) {
    let stroke = COLORS.ink, width = 1.5;
    if (state.moved?.includes(f.name)) { stroke = COLORS.shift; width = 3; }
    if (state.candidates?.includes(f.name)) { stroke = COLORS.unclear; width = 3.5; }
    if (state.masked?.includes(f.name)) { stroke = COLORS.quality; width = 3; }
    if (state.suspect === f.name) stroke = COLORS.broken;
    strokeOf[f.name] = stroke;
    widthOf[f.name] = width;
  }

  const pairs = new Set();
  for (const rule of rules) for (const p of rule.parents) pairs.add(`${p}>${rule.target}`);

  let edges = "";
  const order = { weak: 0, skipped: 1, holds: 2, broken: 3 };   // broken arrows drawn last, on top
  const list = [];
  for (const rule of rules) {
    const kind = rule.broken ? "broken" : rule.skipped ? "skipped" : rule.weak ? "weak" : "holds";
    for (const p of rule.parents) list.push({ rule, p, kind });
  }
  list.sort((x, y) => order[x.kind] - order[y.kind]);
  for (const { rule, p, kind } of list) {
    const a = pos[p], b = pos[rule.target];
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    const mx0 = (a.x + b.x) / 2, my0 = (a.y + b.y) / 2;
    let nx = -dy / d, ny = dx / d;              // the arrow's right-hand side
    let bend;
    if (pairs.has(`${rule.target}>${p}`)) {
      bend = Math.max(16, d * 0.16);            // two-way: each arrow to its own right
    } else {
      bend = d * 0.1;
      if (nx * -mx0 + ny * -my0 < 0) { nx = -nx; ny = -ny; }  // one-way: towards the centre
    }
    const qx = mx0 + nx * bend, qy = my0 + ny * bend;
    // leave each node along the curve's tangent, so the arrow meets the circle edge exactly
    const out = (from, to, gap) => {
      const tx = to.x - from.x, ty = to.y - from.y, t = Math.hypot(tx, ty);
      return [from.x + (tx / t) * gap, from.y + (ty / t) * gap];
    };
    const [sx, sy] = out(a, { x: qx, y: qy }, r + widthOf[p] / 2 + 1);
    const [ex, ey] = out(b, { x: qx, y: qy }, r + widthOf[rule.target] / 2 + 1);
    const title = `${label[rule.target]} predicted from ${rule.parents.map((x) => label[x]).join(", ")}: ${
      { holds: "holds", broken: "broken", skipped: "not checked", weak: "too weak to use" }[kind]}`;
    edges += `<path d="M${sx.toFixed(1)},${sy.toFixed(1)} Q${qx.toFixed(1)},${qy.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}"
      fill="none" stroke="${edgeColour[kind]}" stroke-width="${kind === "broken" ? 2.6 : 1.4}"
      ${kind === "skipped" || kind === "weak" ? 'stroke-dasharray="4 4"' : ""} marker-end="url(#${uid}-arrow-${kind})"><title>${esc(title)}</title></path>`;
  }

  let nodes = "", labels = "";
  let x0 = -R - r, x1 = R + r, y0 = -R - r, y1 = R + r;   // bounding box, grown by every label
  for (const f of features) {
    const { x, y, a } = pos[f.name];
    let fill = "#fff", dash = "";
    if (state.masked?.includes(f.name)) { dash = 'stroke-dasharray="4 3"'; fill = "#f4effa"; }
    if (state.suspect === f.name) fill = COLORS.broken;
    const cos = Math.cos(a), sin = Math.sin(a);
    const gap = r + widthOf[f.name] / 2 + 6;
    const lx = x + cos * gap, ly = y + sin * gap;
    const anchor = Math.abs(cos) < 0.35 ? "middle" : cos > 0 ? "start" : "end";
    const lines = wrapLabel(f.label);
    const h = lines.length * MAP_LINE;
    // first line's centre: below the node, above it, or vertically centred beside it
    const top = sin > 0.35 ? ly + MAP_LINE / 2 : sin < -0.35 ? ly - h + MAP_LINE / 2 : ly - h / 2 + MAP_LINE / 2;
    const w = Math.max(...lines.map((t) => t.length)) * MAP_FONT * 0.56;
    const left = anchor === "start" ? lx : anchor === "end" ? lx - w : lx - w / 2;
    x0 = Math.min(x0, left); x1 = Math.max(x1, left + w);
    y0 = Math.min(y0, top - MAP_LINE / 2); y1 = Math.max(y1, top + h - MAP_LINE / 2);
    const tspans = lines.map((t, k) => `<tspan x="${lx.toFixed(1)}" y="${(top + k * MAP_LINE).toFixed(1)}">${esc(t)}</tspan>`).join("");
    nodes += `<g><title>${esc(f.label)}${f.has_rule ? "" : " (no reliable rule of its own)"}</title>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${strokeOf[f.name]}" stroke-width="${widthOf[f.name]}" ${dash}/></g>`;
    labels += `<text text-anchor="${anchor}" dominant-baseline="central" font-size="${MAP_FONT}"
      font-family="${esc(FONT)}" fill="${COLORS.ink}" stroke="#fff" stroke-width="4"
      stroke-linejoin="round" paint-order="stroke"><title>${esc(f.label)}</title>${tspans}</text>`;
  }

  // padding, and a minimum size so a map with few features is not blown up to giant text
  const pad = 10, minW = 400, minH = 260;
  const w = Math.max(x1 - x0 + 2 * pad, minW), h = Math.max(y1 - y0 + 2 * pad, minH);
  const vb = [(x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2, w, h].map((v) => v.toFixed(1)).join(" ");
  // fixed-size arrowheads (not scaled by line width) whose tip sits exactly on the path's end point
  const marker = (id, colour) => `<marker id="${uid}-arrow-${id}" viewBox="0 0 10 10" refX="10" refY="5" markerUnits="userSpaceOnUse"
    markerWidth="${id === "broken" ? 11 : 9}" markerHeight="${id === "broken" ? 11 : 9}" orient="auto"><path d="M0,1 L10,5 L0,9 z" fill="${colour}"/></marker>`;
  el.innerHTML = `<svg viewBox="${vb}" role="img" aria-label="Map of learned relationships">
    <defs>${Object.entries(edgeColour).map(([k, c]) => marker(k, c)).join("")}</defs>${edges}${nodes}${labels}</svg>`;
}
