"""HTTP API and static file server. All intelligence lives in the sentinel package;
this file only stores runs in memory and turns results into JSON."""
from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel.config import DatasetConfig, suggest_config
from sentinel.data import read_table
from sentinel.diagnose import analyse
from sentinel.evaluate import run_suite
from sentinel.profile import build_profile, load_profile, save_profile
from sentinel.scenarios import FAULTS, Event, events_from, make_batch, presets

ROOT = Path(__file__).resolve().parent.parent
DATASETS, UPLOADS, ARTIFACTS, WEB = ROOT / "datasets", ROOT / "uploads", ROOT / "artifacts", ROOT / "web"
UPLOADS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

app = FastAPI(title="Invariant Sentinel", docs_url="/api/docs")
PROFILES: dict = {}
RUNS: dict = {}
BUILD_LOCK = threading.Lock()


def clean(obj):
    """Make results JSON-safe: numpy -> Python, NaN/inf -> null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def reply(obj, status: int = 200):
    return JSONResponse(clean(obj), status_code=status)


# ---------- datasets and profiles ----------

def config_paths() -> dict:
    found = {}
    for folder, source in ((DATASETS, "example"), (UPLOADS, "uploaded")):
        for path in sorted(folder.glob("*.json")):
            found[path.stem] = (path, source)
    return found


def get_config(name: str) -> DatasetConfig:
    paths = config_paths()
    if name not in paths:
        raise HTTPException(404, f"No dataset called '{name}'.")
    return DatasetConfig.load(paths[name][0])


def get_profile(name: str, build: bool = False):
    if name in PROFILES:
        return PROFILES[name]
    p = load_profile(ARTIFACTS / name)
    if p is None and build:
        with BUILD_LOCK:
            p = load_profile(ARTIFACTS / name)
            if p is None:
                try:
                    p = build_profile(get_config(name), ROOT)
                except ValueError as e:
                    raise HTTPException(400, str(e))
                save_profile(p, ARTIFACTS / name)
                (ARTIFACTS / name / "evaluation.json").unlink(missing_ok=True)  # results belong to the old profile
    if p is None:
        raise HTTPException(409, "This dataset has not been set up yet.")
    PROFILES[name] = p
    return p


def name_of(cfg: DatasetConfig, c: str) -> str:
    return f"log {cfg.label(c)}" if c in cfg.log_features else cfg.label(c)


def equation(r, cfg: DatasetConfig) -> dict:
    terms = " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.2f} × {name_of(cfg, p)}" for p, b in zip(r.parents, r.coefs))
    linear = f"{name_of(cfg, r.target)} ≈ {r.intercept:.2f} {terms}" if r.parents else "no useful rule"
    plain = None
    if r.parents and r.target in cfg.log_features and all(p in cfg.log_features for p in r.parents):
        parts = [f"{cfg.label(p)}^{b:.2f}" for p, b in zip(r.parents, r.coefs)]
        plain = f"{cfg.label(r.target)} grows like {' × '.join(parts)}"
    return {"linear": linear, "plain": plain}


def profile_summary(p) -> dict:
    cfg = p.cfg
    rels = []
    for r in p.rels:
        rels.append({"target": r.target, "label": cfg.label(r.target), "parents": r.parents,
                     "parent_labels": [cfg.label(x) for x in r.parents], "r2": r.r2, "usable": r.usable,
                     **equation(r, cfg)})
    covered = {r.target for r in p.usable}
    n = len(p.features)
    pairs = [(p.features[i], p.features[j], p.iso[i, j]) for i in range(n) for j in range(i + 1, n)
             if p.iso[i, j] > p.thresholds["same_pattern"]["value"]]
    return {
        "name": cfg.name, "title": cfg.title, "description": cfg.description,
        "target": cfg.target, "target_label": cfg.label(cfg.target), "target_unit": cfg.unit(cfg.target),
        "batch_word": cfg.batch_word, "batches_word": cfg.batches_word, "batch_size": cfg.batch_size,
        "features": [{"name": f, "label": cfg.label(f), "unit": cfg.unit(f), "log": f in cfg.log_features,
                      "has_rule": f in covered} for f in p.features],
        "relationships": rels,
        "isolability": p.iso, "same_pattern_pairs": [{"a": cfg.label(a), "b": cfg.label(b), "cos": c} for a, b, c in pairs],
        "blind_spots": [cfg.label(f) for f in p.features if f not in covered],
        "thresholds": p.thresholds, "ks_limits": {cfg.label(k): v for k, v in p.ks_limit.items()},
        "model": {"mae": p.mae_ref, "mae_naive": p.mae_naive, "inputs": p.model.inputs},
        "cleaning": p.cleaning, "sizes": p.sizes, "n_batches": p.n_batches(),
        "build_seconds": p.build_seconds, "shift_story": cfg.shift_story, "driver": cfg.label(cfg.driver),
    }


@app.get("/api/datasets")
def list_datasets():
    out = []
    for name, (path, source) in config_paths().items():
        cfg = DatasetConfig.load(path)
        ready = name in PROFILES or (ARTIFACTS / name / "profile.joblib").exists()
        out.append({"name": name, "title": cfg.title, "description": cfg.description, "source": source,
                    "ready": ready, "evaluated": (ARTIFACTS / name / "evaluation.json").exists(), "target": cfg.label(cfg.target),
                    "monitored": [cfg.label(c) for c in cfg.monitored]})
    return reply(out)


@app.post("/api/datasets/{name}/build")
def build(name: str, rebuild: bool = False):
    if rebuild:
        PROFILES.pop(name, None)
        (ARTIFACTS / name / "profile.joblib").unlink(missing_ok=True)
        (ARTIFACTS / name / "evaluation.json").unlink(missing_ok=True)
    return reply(profile_summary(get_profile(name, build=True)))


@app.get("/api/datasets/{name}/profile")
def profile(name: str):
    return reply(profile_summary(get_profile(name)))


@app.get("/api/datasets/{name}/scenarios")
def scenarios(name: str):
    p = get_profile(name)
    items = []
    for s in presets(p):
        evs = events_from(s["events"])
        items.append({**s, "id": s.get("id"), "steps": [e.describe(p.cfg) for e in evs]})
    return reply({"presets": items, "faults": FAULTS,
                  "features": [{"name": f, "label": p.cfg.label(f), "unit": p.cfg.unit(f),
                                "log": f in p.cfg.log_features} for f in p.features],
                  "max_batches": max(p.n_batches(), 40)})


@app.post("/api/datasets/{name}/evaluate")
def evaluate(name: str, fresh: bool = False):
    p = get_profile(name)
    cache = ARTIFACTS / name / "evaluation.json"
    if cache.exists() and not fresh:
        return JSONResponse(json.loads(cache.read_text()))
    result = clean(run_suite(p, n=30))
    cache.write_text(json.dumps(result))
    return JSONResponse(result)


# ---------- uploads ----------

@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)):
    stem = re.sub(r"[^a-z0-9]+", "_", Path(file.filename or "data").stem.lower()).strip("_") or "data"
    if (DATASETS / f"{stem}.json").exists():
        stem = f"{stem}_upload"
    path = UPLOADS / f"{stem}.csv"
    path.write_bytes(await file.read())
    try:
        df = read_table(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read this file as CSV: {e}")
    if len(df) < 500:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"The file has {len(df)} rows; at least 500 are needed "
                                 "(half for training, a fifth for calibration, the rest to replay).")
    suggestion = suggest_config(df, stem, f"uploads/{stem}.csv")
    suggestion["rows"] = int(len(df))
    return reply(suggestion)


class UploadConfig(BaseModel):
    name: str
    title: str
    target: str
    monitored: list
    context: list = []
    time_column: str | None = None
    log_features: list = []
    batch_size: int = 200
    shift_driver: str | None = None
    batch_word: str = "batch"
    description: str = ""


@app.post("/api/uploads/{name}/config")
def save_upload_config(name: str, body: UploadConfig):
    csv = UPLOADS / f"{name}.csv"
    if not csv.exists():
        raise HTTPException(404, "Upload the CSV first.")
    if len(body.monitored) < 3:
        raise HTTPException(400, "Choose at least three monitored columns; relationships need company.")
    if body.target in body.monitored:
        raise HTTPException(400, "The target cannot also be a monitored column.")
    cfg = DatasetConfig(name=name, title=body.title or name, csv=f"uploads/{name}.csv", target=body.target,
                        monitored=body.monitored, context=[c for c in body.context if c not in body.monitored],
                        time_column=body.time_column or None,
                        log_features=[c for c in body.log_features if c in body.monitored],
                        batch_size=int(body.batch_size), shift_driver=body.shift_driver or body.monitored[0],
                        batch_word=body.batch_word or "batch", description=body.description)
    cfg.save(UPLOADS / f"{name}.json")
    PROFILES.pop(name, None)
    (ARTIFACTS / name / "profile.joblib").unlink(missing_ok=True)
    (ARTIFACTS / name / "evaluation.json").unlink(missing_ok=True)
    return build(name)


# ---------- runs ----------

class RunRequest(BaseModel):
    dataset: str
    preset: str | None = None
    batches: int | None = None
    events: list = []
    title: str | None = None
    seed: int = 0


class EventRequest(BaseModel):
    kind: str = "fault"
    feature: str | None = None
    fault: str | None = None
    amount: float | None = None
    strength: float = 1.0
    start: int | None = None
    end: int | None = None


def brief(r: dict, p) -> dict:
    cfg = p.cfg
    return {"index": r["index"], "verdict": r["verdict"], "title": r["title"], "headline": r["headline"],
            "confidence": r["confidence"], "baseline": r["baseline"]["alarm"],
            "baseline_moved": [cfg.label(f) for f in r["baseline"]["moved"]],
            "suspect": cfg.label(r["suspect"]) if r["suspect"] else None,
            "mae": r["mae"], "mae_ref": r["mae_ref"]}


def run_state(run_id: str) -> dict:
    run = RUNS[run_id]
    p = run["profile"]
    return {"id": run_id, "dataset": run["dataset"], "title": run["title"], "batches": run["batches"],
            "done": len(run["results"]), "finished": len(run["results"]) >= run["batches"],
            "timeline": [brief(r, p) for r in run["results"]], "batch_word": p.cfg.batch_word}


@app.post("/api/runs")
def create_run(body: RunRequest):
    p = get_profile(body.dataset)
    title, events, batches = body.title or "Custom scenario", body.events, body.batches
    if body.preset is not None:
        match = [s for s in presets(p) if str(s.get("id")) == body.preset]
        if not match:
            raise HTTPException(404, "Unknown scenario.")
        title, events, batches = match[0]["title"], match[0]["events"], batches or match[0]["batches"]
    run_id = f"r{len(RUNS) + 1}"
    RUNS[run_id] = {"dataset": body.dataset, "profile": p, "title": title,
                    "events": events_from(events), "batches": int(min(max(batches or 24, 4), 200)),
                    "results": [], "seed": body.seed}
    return reply(run_state(run_id))


def get_run(run_id: str) -> dict:
    if run_id not in RUNS:
        raise HTTPException(404, "This run no longer exists (the server may have restarted). Start a new one.")
    return RUNS[run_id]


@app.get("/api/runs/{run_id}")
def read_run(run_id: str):
    get_run(run_id)
    return reply(run_state(run_id))


@app.post("/api/runs/{run_id}/next")
def next_batch(run_id: str, count: int = 1):
    run = get_run(run_id)
    p = run["profile"]
    if len(run["results"]) >= run["batches"]:
        raise HTTPException(409, "This run has finished. Start a new one to keep going.")
    for _ in range(max(1, min(count, 50))):
        k = len(run["results"])
        if k >= run["batches"]:
            break
        run["results"].append(analyse(p, make_batch(p, k, run["events"], run["seed"]), k))
    return reply(run_state(run_id))


@app.get("/api/runs/{run_id}/batches/{index}")
def read_batch(run_id: str, index: int):
    run = get_run(run_id)
    if not 0 <= index < len(run["results"]):
        raise HTTPException(404, "That batch has not been analysed yet.")
    p = run["profile"]
    cfg = p.cfg
    columns = list(dict.fromkeys([*cfg.monitored, cfg.target]))
    batch = make_batch(p, index, run["events"], run["seed"])
    result = {
        **run["results"][index],
        "record_columns": [{"name": c, "label": cfg.label(c), "unit": cfg.unit(c), "target": c == cfg.target}
                           for c in columns],
        "records": [{"index": i + 1, "values": [row[c] for c in columns]}
                    for i, (_, row) in enumerate(batch[columns].iterrows())],
    }
    return reply(result)


@app.post("/api/runs/{run_id}/events")
def add_event(run_id: str, body: EventRequest):
    run = get_run(run_id)
    p = run["profile"]
    if body.kind == "fault":
        if body.feature not in p.features or body.fault not in FAULTS:
            raise HTTPException(422, "Choose a feature and a fault type.")
    start = len(run["results"]) if body.start is None else body.start
    ev = Event(kind=body.kind, start=start, end=body.end, feature=body.feature, fault=body.fault,
               amount=body.amount if body.amount is not None else (FAULTS.get(body.fault or "", {}) or {}).get("default"),
               strength=body.strength)
    run["events"].append(ev)
    return reply({"added": ev.describe(p.cfg), **run_state(run_id)})


@app.get("/api/runs/{run_id}/truth")
def truth(run_id: str):
    run = get_run(run_id)
    p = run["profile"]
    return reply({"events": [{"text": e.describe(p.cfg), "kind": e.kind, "start": e.start, "end": e.end,
                              "feature": p.cfg.label(e.feature) if e.feature else None} for e in run["events"]]})


# ---------- static files ----------

app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
