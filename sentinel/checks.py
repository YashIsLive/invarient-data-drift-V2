"""The three checks run on every batch, in this order:

1. Quality  - is the data even valid? (missing values, impossible values, stuck readings)
2. Drift    - did any feature's values move? (Kolmogorov-Smirnov statistic)
3. Rules    - do the features still agree with each other? (relationship residuals)

Quality runs first so that a unit bug or a missing burst is called what it is,
instead of showing up as huge 'drift' and contaminating the other checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ks_stat(ref_sorted: np.ndarray, values: np.ndarray) -> float:
    """Largest vertical gap between the two cumulative distribution curves (0 = identical, 1 = no overlap)."""
    x = np.sort(values[np.isfinite(values)])
    if len(x) == 0:
        return float("nan")
    grid = np.concatenate([ref_sorted, x])
    cdf_ref = np.searchsorted(ref_sorted, grid, side="right") / len(ref_sorted)
    cdf_new = np.searchsorted(x, grid, side="right") / len(x)
    return float(np.max(np.abs(cdf_ref - cdf_new)))


def quality_check(profile, raw: pd.DataFrame, T: pd.DataFrame):
    """Returns (issues, masked features, notes, T with individual bad values removed).

    A feature is *masked* (left out of every later check) when a large part of the
    batch is invalid. A handful of bad rows are just ignored and mentioned in a note."""
    thr = profile.thresholds
    issues, notes, masked = [], [], set()
    T = T.copy()
    n = len(raw)
    for c in profile.features:
        raw_v = pd.to_numeric(raw[c], errors="coerce")
        t = T[c].to_numpy(dtype=float)
        missing = float(raw_v.isna().mean()) if n else 1.0
        impossible = (raw_v.notna().to_numpy() & ~np.isfinite(t))  # e.g. zero size on a log column
        lo, hi = profile.plausible[c]
        outside = np.isfinite(t) & ((t < lo) | (t > hi))
        bad = impossible | outside
        bad_share = float(bad.mean()) if n else 0.0
        label = profile.cfg.label(c)
        if missing > thr["missing_share"]["value"]:
            issues.append({"feature": c, "issue": "missing",
                           "text": f"{label} is missing in {missing:.0%} of rows."})
            masked.add(c)
        elif bad_share > thr["outside_share"]["value"]:
            usual = profile.typical[c]
            now = float(np.nanmedian(raw_v)) if raw_v.notna().any() else float("nan")
            issues.append({"feature": c, "issue": "out_of_range",
                           "text": f"{bad_share:.0%} of {label} values are outside the plausible range "
                                   f"(typical value now {fmt(now)}, usually {fmt(usual)}). "
                                   f"This looks like a unit or scaling error."})
            masked.add(c)
        else:
            valid = t[np.isfinite(t) & ~bad]
            if len(valid) >= 10 and np.std(valid) < thr["stuck_ratio"]["value"] * profile.ref_std[c]:
                issues.append({"feature": c, "issue": "stuck",
                               "text": f"{label} barely changes within this {profile.cfg.batch_word} "
                                       f"(it looks frozen at {fmt(float(np.nanmedian(raw_v)))})."})
                masked.add(c)
            elif bad.any():
                notes.append(f"{int(bad.sum())} row(s) had an impossible {label} value and were ignored.")
                T.loc[bad, c] = np.nan
            if 0 < missing <= thr["missing_share"]["value"]:
                notes.append(f"{label} is missing in {int(raw_v.isna().sum())} row(s).")
    return issues, masked, notes, T


def drift_check(profile, T: pd.DataFrame, masked: set) -> list:
    out = []
    for c in profile.features:
        if c in masked:
            continue
        v = T[c].to_numpy(dtype=float)
        ks = ks_stat(profile.ref_sorted[c], v)
        out.append({"feature": c, "ks": ks, "limit": profile.ks_limit[c],
                    "moved": bool(ks > profile.ks_limit[c])})
    return out


def rule_stats(rels: list, T: pd.DataFrame, masked: set, min_rows: int) -> list:
    """For every usable rule: mean residual (shift), z-score of that mean, and spread.

    Spread is the residual's scatter divided by its usual scatter (1.0 = as usual,
    2.0 = twice as noisy). It uses the median absolute deviation, so a couple of
    odd rows cannot inflate it."""
    out = []
    for i, r in enumerate([r for r in rels if r.usable]):
        involved = [r.target, *r.parents]
        if any(c in masked for c in involved):
            out.append({"rule": i, "skipped": True})
            continue
        e = r.residual(T)
        e = e[np.isfinite(e)]
        if len(e) < min_rows:
            out.append({"rule": i, "skipped": True})
            continue
        m = float(e.mean())
        out.append({"rule": i, "skipped": False, "n": int(len(e)), "mean": m,
                    "z": m * np.sqrt(len(e)),
                    "spread": float(1.4826 * np.median(np.abs(e - np.median(e)))) / r.spread_ref})
    return out


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"
