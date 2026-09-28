// Entry point: dataset choice, tabs, and page rendering on demand.
import { api, esc } from "./api.js";
import { resizeCharts } from "./charts.js";
import { initMonitor, loadMonitor } from "./monitor.js";
import { initData, initEvaluation, renderData, renderEvaluation, renderRelationships } from "./pages.js";

const $ = (id) => document.getElementById(id);
const app = { dataset: null, profile: null, page: "monitor", rendered: new Set() };

function notice(text, isError = false) {
  const el = $("notice");
  el.hidden = !text;
  el.className = "notice" + (isError ? " error" : "");
  el.innerHTML = text ? `${esc(text)} <button class="link small" type="button">Dismiss</button>` : "";
  el.querySelector("button")?.addEventListener("click", () => notice(""));
  clearTimeout(notice.timer);
  if (text && !isError) notice.timer = setTimeout(() => notice(""), 8000);
}

async function showPage(page) {
  app.page = page;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.page === page)));
  document.querySelectorAll("[data-page-body]").forEach((s) => { s.hidden = s.id !== `page-${page}`; });
  history.replaceState(null, "", `#${page}`);
  if (!app.profile && page !== "data") return;
  // each page is drawn once per dataset, then only resized
  if (!app.rendered.has(page)) {
    app.rendered.add(page);
    try {
      if (page === "relationships") renderRelationships(app.profile);
      if (page === "evaluation") await renderEvaluation(app.dataset);
      if (page === "data") await renderData(app.dataset, notice);
    } catch (e) {
      app.rendered.delete(page);
      notice(e.message, true);
    }
  }
  const body = $(`page-${page}`);
  requestAnimationFrame(() => resizeCharts(body));
}

async function useDataset(name, { page } = {}) {
  const select = $("dataset");
  select.disabled = true;
  try {
    let profile;
    try {
      profile = await api(`/datasets/${name}/profile`);
    } catch {
      // not set up yet: learn it now (a few seconds to a minute)
      notice("");
      $("status-headline").innerHTML = `<span class="busy">Learning relationships for this dataset for the first time…</span>`;
      profile = await api(`/datasets/${name}/build`, { method: "POST" });
    }
    app.dataset = name;
    app.profile = profile;
    app.rendered.clear();
    try { localStorage.setItem("sentinel.dataset", name); } catch { /* storage may be unavailable */ }
    $("dataset-title").textContent = `· ${profile.title}`;
    document.title = `${profile.title} · Invariant Sentinel`;
    await refreshDatasetList(name);
    await loadMonitor(name, profile);
    await showPage(page || app.page);
  } catch (e) {
    notice(e.message, true);
  } finally {
    select.disabled = false;
  }
}

async function refreshDatasetList(current) {
  const list = await api("/datasets");
  $("dataset").innerHTML = list.map((d) =>
    `<option value="${esc(d.name)}"${d.name === current ? " selected" : ""}>${esc(d.title)}${d.ready ? "" : " (not set up)"}</option>`).join("");
  return list;
}

async function boot() {
  initMonitor({ notice });
  initEvaluation(() => app.dataset);
  initData({ use: useDataset, notice });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));
  $("dataset").addEventListener("change", (e) => useDataset(e.target.value));

  const hash = location.hash.slice(1);
  if (["monitor", "relationships", "evaluation", "data"].includes(hash)) app.page = hash;
  let list;
  try {
    list = await refreshDatasetList(null);
  } catch (e) {
    return notice(`Cannot reach the server: ${e.message}`, true);
  }
  let saved = null;
  try { saved = localStorage.getItem("sentinel.dataset"); } catch { /* ignore */ }
  const first = list.find((d) => d.name === saved) || list.find((d) => d.name === "diamonds") || list[0];
  if (first) await useDataset(first.name);
  else showPage("data");
}

boot();
