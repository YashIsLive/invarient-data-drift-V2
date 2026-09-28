"""Finding the faulty feature by matching fingerprints.

We observe how far each rule's residual moved (vector v). For every candidate
feature j we know the pattern a fault on j would cause (column j of the
signature matrix). We ask: "if j alone were wrong by some amount d, how much of
v would that explain?"

    d_j = (S_j . v) / (S_j . S_j)                 best-fitting fault size (least squares)
    E_j = 1 - |v - d_j S_j|^2 / |v|^2             share of the pattern explained (0..1)

The feature with the highest E is the suspect. If no one explains most of the
pattern, or two explain it about equally well, we say so instead of guessing.

Candidates that would need an absurd error (over 5x the feature's normal
variation) to explain the pattern are dropped.

The same idea works for noisy sensors, using variances: extra noise on j adds
S_j^2 x (noise variance) to each rule's residual variance.
"""
from __future__ import annotations

import numpy as np


def locate(profile, stats: list, masked: set, mode: str) -> dict:
    rows = [s for s in stats if not s["skipped"]]
    idx = [s["rule"] for s in rows]
    S = profile.S[idx]
    if mode == "shift":
        v, M = np.array([s["mean"] for s in rows]), S
    else:
        # spread is measured relative to each rule's usual scatter, so rescale the signatures the same way
        scale = np.array([profile.usable[i].sigma / profile.usable[i].spread_ref for i in idx])
        v, M = np.array([s["spread"] ** 2 - 1.0 for s in rows]), (S * scale[:, None]) ** 2
    total = float(v @ v) or 1e-12
    candidates = []
    for j, c in enumerate(profile.features):
        col = M[:, j]
        if c in masked or not np.any(col):
            continue
        d = float(col @ v / (col @ col))
        size = abs(d) if mode == "shift" else np.sqrt(max(d, 0))
        if size > 5 * profile.ref_std[c]:
            continue  # would need an absurdly large error on this feature: not a plausible cause
        explained = 0.0 if (mode == "spread" and d <= 0) else 1 - float(np.sum((v - d * col) ** 2)) / total
        candidates.append({"feature": c, "explained": max(0.0, explained), "delta": d,
                           "expected": (d * col).tolist()})
    candidates.sort(key=lambda x: -x["explained"])
    return {"mode": mode, "rules": idx, "observed": v.tolist(), "candidates": candidates}
