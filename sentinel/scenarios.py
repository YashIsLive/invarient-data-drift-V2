"""Simulated days: a stream of batches with optional harmless shifts and injected faults.

Harmless shift: rows are re-sampled from real data, favouring rows with a
larger value of the driver feature (e.g. heavier diamonds). Whole real rows are
kept, so every relationship inside a row stays true, while the values move.

Faults: applied to real rows with simple formulas, and always logged, so the
'reveal' can show the ground truth after the system has made its call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

FAULTS = {
    "percent": {"name": "Reads a percentage too high or low", "amount": "Percent (e.g. 3 or -5)", "default": 3.0},
    "offset": {"name": "Reads a fixed amount too high or low", "amount": "Amount in the feature's unit", "default": 1.0},
    "noise": {"name": "Noisy readings", "amount": "Extra random error, % of a typical value", "default": 3.0},
    "stuck": {"name": "Stuck at one value", "amount": None, "default": None},
    "missing": {"name": "Values missing", "amount": "Percent of rows", "default": 60.0},
    "unit": {"name": "Unit changed", "amount": "Multiply by (e.g. 0.1 for mm → cm)", "default": 0.1},
}


@dataclass
class Event:
    kind: str                   # "shift" or "fault"
    start: int                  # first batch it affects (0-based)
    end: int | None = None      # last batch (inclusive); None = until the end
    feature: str | None = None
    fault: str | None = None
    amount: float | None = None
    strength: float = 1.0       # shift strength

    def active(self, k: int) -> bool:
        return k >= self.start and (self.end is None or k <= self.end)

    def describe(self, cfg) -> str:
        span = f"from {cfg.batch_word} {self.start + 1}" + (f" to {self.end + 1}" if self.end is not None else "")
        if self.kind == "shift":
            return f"Harmless shift ({cfg.shift_story or 'more high ' + cfg.label(cfg.driver)}, strength {self.strength:g}) {span}"
        label = cfg.label(self.feature)
        a = self.amount
        text = {
            "percent": f"{label} reads {abs(a):g}% too {'high' if a > 0 else 'low'}",
            "offset": f"{label} reads {abs(a):g} {('percentage point' if abs(a) == 1 else 'percentage points') if cfg.unit(self.feature) == '%' else cfg.unit(self.feature)} too {'high' if a > 0 else 'low'}".replace("  ", " "),
            "noise": f"{label} gets noisy (±{a:g}% of a typical value)",
            "stuck": f"{label} gets stuck at one value",
            "missing": f"{label} goes missing in {a:g}% of rows",
            "unit": f"{label} changes unit (values × {a:g})",
        }[self.fault]
        return f"{text} {span}"

    def to_dict(self) -> dict:
        return asdict(self)


def apply_fault(batch: pd.DataFrame, e: Event, rng, profile) -> None:
    c = e.feature
    v = pd.to_numeric(batch[c], errors="coerce").astype(float)
    a = e.amount if e.amount is not None else FAULTS[e.fault]["default"]
    if e.fault == "percent":
        v = v * (1 + a / 100)
    elif e.fault == "offset":
        v = v + a
    elif e.fault == "noise":
        v = v + rng.normal(0, abs(a) / 100 * abs(profile.typical[c]), len(v))
    elif e.fault == "stuck":
        v[:] = v.dropna().iloc[0] if v.notna().any() else np.nan
    elif e.fault == "missing":
        v[rng.random(len(v)) < a / 100] = np.nan
    elif e.fault == "unit":
        v = v * a
    batch[c] = v


def make_batch(profile, k: int, events: list, seed: int = 0) -> pd.DataFrame:
    B = profile.cfg.batch_size
    stream = profile.stream
    N = len(stream)
    rng = np.random.default_rng(seed * 100_003 + k)
    shifts = [e for e in events if e.kind == "shift" and e.active(k)]
    if shifts:
        lam = shifts[-1].strength
        z = np.nan_to_num(profile.driver_z, nan=-50.0)
        w = np.exp(np.clip(lam * z, -50, 50))
        idx = rng.choice(N, B, replace=True, p=w / w.sum())
    else:
        start = (k * B) % N
        idx = np.arange(start, start + B) % N
    batch = stream.iloc[idx].copy().reset_index(drop=True)
    for e in events:
        if e.kind == "fault" and e.active(k):
            apply_fault(batch, e, rng, profile)
    return batch


def default_fault_size(profile, feature: str) -> tuple[str, float]:
    """A fault big enough to be caught but small enough to look like normal values:
    about 1.5 of the relationship's usual errors."""
    rule = next((r for r in profile.usable if r.target == feature), None)
    sigma = rule.sigma if rule else profile.ref_std[feature] * 0.3
    if feature in profile.cfg.log_features:
        return "percent", float(max(1.0, round((np.exp(1.5 * sigma) - 1) * 100, 0)))
    step = 1.5 * sigma
    mag = 10 ** np.floor(np.log10(step)) if step > 0 else 1
    return "offset", float(f"{round(step / mag) * mag:.3g}")


def presets(profile) -> list:
    """Hand-written scenarios from the dataset config, or sensible generated ones."""
    cfg = profile.cfg
    if cfg.scenarios:
        return [{"id": f"s{i}", **s} for i, s in enumerate(cfg.scenarios)]
    n = min(24, profile.n_batches())
    # the feature whose fingerprint is most distinct is the best demo target
    feats = [f for f in profile.features if any(r.target == f for r in profile.usable)]
    iso = profile.iso
    ranked = sorted(feats, key=lambda f: max([iso[profile.features.index(f), profile.features.index(g)]
                                                for g in feats if g != f] or [0]))
    f = ranked[0] if ranked else profile.features[0]
    kind, size = default_fault_size(profile, f)
    L = cfg.label
    third, half = max(2, n // 3), max(3, n // 2)
    return [
        {"id": "normal", "title": f"Ordinary {cfg.batches_word}", "batches": n, "events": []},
        {"id": "shift", "title": f"Harmless shift: {cfg.shift_story or 'more high ' + L(cfg.driver)}",
         "batches": n, "events": [{"kind": "shift", "start": third, "strength": 1.0}]},
        {"id": "fault", "title": f"{L(f)} reads slightly wrong", "batches": n,
         "events": [{"kind": "fault", "start": half, "feature": f, "fault": kind, "amount": size}]},
        {"id": "both", "title": "Harmless shift, then a small fault", "batches": n,
         "events": [{"kind": "shift", "start": third, "strength": 1.0},
                    {"kind": "fault", "start": half + 2, "feature": f, "fault": kind, "amount": size}]},
        {"id": "unit", "title": f"{L(f)} recorded in a different unit", "batches": n,
         "events": [{"kind": "fault", "start": half, "feature": f, "fault": "unit", "amount": 0.1}]},
        {"id": "missing", "title": f"{L(profile.features[-1])} goes missing", "batches": n,
         "events": [{"kind": "fault", "start": half, "end": half + 3, "feature": profile.features[-1],
                     "fault": "missing", "amount": 60}]},
    ]


def events_from(dicts: list) -> list:
    known = Event.__dataclass_fields__.keys()
    return [Event(**{k: v for k, v in d.items() if k in known}) for d in dicts]
