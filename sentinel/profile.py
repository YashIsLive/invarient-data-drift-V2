"""The reference profile: everything the system learns from healthy data, once, before monitoring.

    train split (50%)       -> the model, the relationships, reference distributions
    calibration split (20%) -> relationship quality (R², usual error) and every
                               data-driven threshold ("how big does a number get
                               on normal days?")
    stream split (30%)      -> never seen during setup; replayed as live batches

Thresholds are learned by cutting the calibration data into hundreds of
batch-sized windows, computing each statistic on them, and taking a high
percentile. That way roughly 1 in 100 healthy batches raises an alarm, and we
never rely on textbook p-values, which assume independent rows.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import checks
from .config import DatasetConfig
from .data import clean_reference, load_frame, split, to_space
from .impact import impact_ratio
from .model import TargetModel
from .relationships import isolability, learn, signature_matrix

VERSION = 5
LOG = logging.getLogger("sentinel")


@dataclass
class Profile:
    cfg: DatasetConfig
    features: list
    rels: list
    S: np.ndarray
    iso: np.ndarray
    thresholds: dict
    ks_limit: dict
    ref_sorted: dict
    plausible: dict
    familiar: dict
    typical: dict
    ref_std: dict
    model: TargetModel
    mae_ref: float
    mae_naive: float
    impact_null: dict
    cleaning: dict
    sizes: dict
    stream: pd.DataFrame
    driver_z: np.ndarray
    cloud: pd.DataFrame
    build_seconds: float = 0.0
    version: int = VERSION
    quality_rates: dict = field(default_factory=dict)  # feature -> share of healthy raw rows missing / invalid
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> list:
        return [r for r in self.rels if r.usable]

    @property
    def min_rows(self) -> int:
        return int(max(15, 0.3 * self.cfg.batch_size))

    def n_batches(self) -> int:
        return max(1, len(self.stream) // self.cfg.batch_size)


def _windows(df: pd.DataFrame, size: int, count: int, rng) -> list:
    if len(df) >= size + 5:
        starts = rng.integers(0, len(df) - size, count)
        return [np.arange(s, s + size) for s in starts]
    # tiny datasets: resample rows instead of cutting contiguous windows
    return [rng.integers(0, len(df), size) for _ in range(count)]


def build_profile(cfg: DatasetConfig, root: Path, seed: int = 42, log=print) -> Profile:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    parts = split(load_frame(cfg, root), cfg, seed)
    train, rep_train = clean_reference(parts["train"], cfg)
    calib, rep_calib = clean_reference(parts["calib"], cfg)
    B = cfg.batch_size
    if len(calib) < B:
        raise ValueError(f"Only {len(calib)} clean calibration rows; use a batch size below {len(calib)}.")
    features = list(cfg.monitored)
    Ttr, Tca = to_space(train, cfg), to_space(calib, cfg)

    log("Learning relationships")
    rels = learn(Ttr, Tca, features)
    S = signature_matrix(rels, features)
    iso = isolability(S)

    log("Training the model")
    model = TargetModel(cfg).fit(train)
    y_cal = calib[cfg.target].to_numpy(dtype=float)
    mae_ref = float(np.mean(np.abs(model.predict(calib) - y_cal)))
    mae_naive = float(np.mean(np.abs(train[cfg.target].mean() - y_cal)))

    ref_sorted, plausible, familiar, ref_std = {}, {}, {}, {}
    for c in features:
        v = Ttr[c].dropna().to_numpy()
        sample = rng.choice(v, min(len(v), 5000), replace=False)
        ref_sorted[c] = np.sort(sample)
        q005, q995 = np.quantile(v, [0.005, 0.995])
        span = q995 - q005
        plausible[c] = (float(q005 - 0.25 * span), float(q995 + 0.25 * span))
        familiar[c] = (float(q005), float(q995))
        ref_std[c] = float(np.std(v))
    typical = {c: float(train[c].median()) for c in features}

    # provisional profile so the batch checks below can run on calibration windows
    p = Profile(cfg, features, rels, S, iso, {}, {}, ref_sorted, plausible, familiar, typical,
                ref_std, model, mae_ref, mae_naive, {}, {}, {}, parts["stream"],
                np.zeros(0), pd.DataFrame())

    log("Calibrating thresholds on healthy windows")
    k = len(features)
    ks_null = {c: [] for c in features}
    z_null, spread_null, unfamiliar_null = [], [], []
    for idx in _windows(Tca, B, 800, rng):
        W = Tca.iloc[idx]
        for c in features:
            ks_null[c].append(checks.ks_stat(ref_sorted[c], W[c].to_numpy()))
        stats = [s for s in checks.rule_stats(rels, W, set(), p.min_rows) if not s["skipped"]]
        if stats:
            z_null.append(max(abs(s["z"]) for s in stats))
            spread_null.append(max(s["spread"] for s in stats))
        unfamiliar_null.append(_unfamiliar_share(p, W))
    ks_q = 1 - 0.01 / k  # split the 1% alarm budget across features
    ks_limit = {c: float(np.quantile(ks_null[c], ks_q)) for c in features}

    impact_null = {}
    base_windows = _windows(calib, B, 120, rng)
    for c in features:
        ratios = []
        for idx in base_windows:
            W = calib.iloc[idx].reset_index(drop=True)
            res = impact_ratio(p, W, c)
            if res is None:
                break
            ratios.append(res["ratio"])
        if ratios:
            impact_null[c] = float(np.quantile(ratios, 0.95))

    thresholds = {
        "missing_share": {"value": 0.2, "source": "Heuristic",
                          "meaning": "A feature missing in more than this share of a batch is a quality issue"},
        "outside_share": {"value": 0.05, "source": "Heuristic, bounds from reference data",
                          "meaning": "Share of values outside the plausible range that marks a unit/scale error"},
        "stuck_ratio": {"value": 0.01, "source": "Heuristic",
                        "meaning": "Spread below this fraction of the usual spread means a stuck feature"},
        "min_r2": {"value": 0.5, "source": "Heuristic",
                   "meaning": "Relationships that explain less than half the variation are not used"},
        "z_limit": {"value": float(np.quantile(z_null, 0.995)) if z_null else float("inf"),
                    "source": "Calibrated (99.5th percentile of healthy batches)",
                    "meaning": "Largest shift score across all relationships on a healthy batch"},
        "spread_limit": {"value": max(1.5, float(np.quantile(spread_null, 0.995))) if spread_null else float("inf"),
                         "source": "Calibrated (99.5th percentile of healthy batches), at least 1.5",
                         "meaning": "Largest residual scatter (× usual) across relationships on a healthy batch. "
                                    "The 1.5 floor tolerates natural changes in scatter when values shift"},
        "ks_percentile": {"value": ks_q, "source": "Calibrated (per feature, 1% split across features)",
                          "meaning": "Percentile of healthy KS values used as each feature's drift limit"},
        "unfamiliar_share": {"value": float(np.quantile(unfamiliar_null, 0.99)),
                             "source": "Calibrated (99th percentile)",
                             "meaning": "Share of rows outside the training range above which the model is extrapolating"},
        "explained_min": {"value": 0.7, "source": "Heuristic",
                          "meaning": "A single feature must explain at least this share of the broken pattern"},
        "margin_min": {"value": 0.2, "source": "Heuristic",
                       "meaning": "The suspect must beat the runner-up by at least this much"},
        "same_pattern": {"value": 0.95, "source": "Heuristic",
                         "meaning": "Fingerprints more similar than this cannot be told apart"},
        "impact_material": {"value": 0.25, "source": "Heuristic + calibrated null",
                            "meaning": "Impact counts as material above this and above the healthy 95th percentile"},
    }

    # Per-row limit for the affected-rows report: how far off (in usual errors) a single healthy row gets.
    # Computed from every clean calibration row, without the random generator, so all other
    # thresholds stay exactly as they were.
    row_limit, row_rate = {}, {}
    for r in p.usable:
        e = np.abs(r.residual(Tca))
        e = e[np.isfinite(e)]
        if len(e):
            row_limit[r.target] = float(np.quantile(e, 0.99))
            row_rate[r.target] = float(np.mean(e > row_limit[r.target]))
    thresholds["row_limit"] = {
        "value": row_limit, "healthy_rate": row_rate, "source": "Calibrated (99th percentile of healthy rows)",
        "meaning": "A single row is flagged when it is further than this from what its neighbours predict "
                   "(in usual errors, per feature), so about 1 healthy row in 100 is flagged"}

    # how often healthy raw rows are missing or invalid, so quality flags can be compared with a normal day
    raw_calib = parts["calib"]
    T_raw = to_space(raw_calib, cfg)
    quality_rates = {}
    for c in features:
        v = pd.to_numeric(raw_calib[c], errors="coerce")
        t = T_raw[c].to_numpy(dtype=float)
        lo, hi = plausible[c]
        invalid = (v.notna().to_numpy() & ~np.isfinite(t)) | (np.isfinite(t) & ((t < lo) | (t > hi)))
        quality_rates[c] = {"missing": float(v.isna().mean()), "out_of_range": float(invalid.mean())}

    # standardized driver values of the stream, used to simulate a harmless shift
    Tst = to_space(parts["stream"], cfg)
    d = Tst[cfg.driver].to_numpy(dtype=float)
    driver_z = (d - np.nanmean(Ttr[cfg.driver])) / (np.nanstd(Ttr[cfg.driver]) or 1.0)

    cloud = calib.sample(min(400, len(calib)), random_state=seed).reset_index(drop=True)

    p.thresholds, p.ks_limit, p.impact_null = thresholds, ks_limit, impact_null
    p.cleaning = {"train": rep_train, "calib": rep_calib}
    p.sizes = {k2: int(len(v)) for k2, v in parts.items()}
    p.driver_z, p.cloud = driver_z, cloud
    p.quality_rates = quality_rates
    p.extra["seed"] = seed
    p.build_seconds = round(time.time() - t0, 1)
    log(f"Profile ready in {p.build_seconds}s")
    return p


def _unfamiliar_share(p: Profile, T: pd.DataFrame) -> float:
    outside = np.zeros(len(T), dtype=bool)
    for c in p.features:
        lo, hi = p.familiar[c]
        v = T[c].to_numpy(dtype=float)
        outside |= np.isfinite(v) & ((v < lo) | (v > hi))
    return float(outside.mean()) if len(T) else 0.0


def save_profile(p: Profile, folder: Path) -> None:
    """Two files: profile.joblib holds only numbers, tables and plain Python objects, so it loads under
    any scikit-learn version; model.joblib holds the scikit-learn model, the one version-sensitive part."""
    import sklearn
    folder.mkdir(parents=True, exist_ok=True)
    p.extra["sklearn"] = sklearn.__version__
    p.extra.pop("needs_model", None)
    fitted = p.model.model
    p.model.model = None
    try:
        joblib.dump(p, folder / "profile.joblib")
    finally:
        p.model.model = fitted
    joblib.dump(fitted, folder / "model.joblib")


def load_profile(folder: Path, root: Path | None = None) -> Profile | None:
    """None if there is no usable profile (missing, unreadable, or from an older code version).

    A profile saved under another scikit-learn version keeps its rules, thresholds and evaluation;
    only the model is retrained (seconds, not a full rebuild). That needs `root` to find the CSV;
    without it such a profile counts as missing."""
    path = folder / "profile.joblib"
    if not path.exists():
        return None
    import warnings
    import sklearn
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an old single-file profile may carry a model from another version
            p = joblib.load(path)
    except Exception:
        return None
    if getattr(p, "version", 0) != VERSION:
        return None
    same_sklearn = p.extra.get("sklearn") == sklearn.__version__
    if p.model.model is None:  # current layout: the model sits in its own file
        model_path = folder / "model.joblib"
        if same_sklearn and model_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p.model.model = joblib.load(model_path)
            except Exception:
                pass
    elif not same_sklearn:
        p.model.model = None   # old single-file layout, model from another version: do not trust it
    LOG.info("Loaded profile %s in %.2fs", p.cfg.name, time.perf_counter() - t0)
    if p.model.model is None:
        if root is None:
            return None
        retrain_model(p, root)
        save_profile(p, folder)
    return p


def retrain_model(p: Profile, root: Path) -> Profile:
    """Refit only the model, on exactly the same training rows as the original build.
    Rules, thresholds and the cached evaluation do not depend on the model and are kept."""
    t0 = time.perf_counter()
    cfg = p.cfg
    parts = split(load_frame(cfg, root), cfg, p.extra.get("seed", 42))
    train, _ = clean_reference(parts["train"], cfg)
    calib, _ = clean_reference(parts["calib"], cfg)
    p.model = TargetModel(cfg).fit(train)
    p.mae_ref = float(np.mean(np.abs(p.model.predict(calib) - calib[cfg.target].to_numpy(dtype=float))))
    p.extra["retrain_seconds"] = round(time.perf_counter() - t0, 1)
    LOG.info("Retrained the model for %s (scikit-learn version changed) in %.1fs",
             cfg.name, time.perf_counter() - t0)
    return p
