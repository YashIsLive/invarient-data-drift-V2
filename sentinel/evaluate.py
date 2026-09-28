"""Repeatable evaluation: Sentinel versus a standard drift check on the same batches.

Each situation is simulated on many different batches (different rows, different
random seeds). The standard check is the drift step on its own: KS per feature
with the same calibrated limits, alarm if any feature moved, blame the feature
that moved most. Using the same limits means any difference comes from the
relationships, not from better thresholds.

Every rate comes with a 95% Wilson interval, because 30 batches is not many.
"""
from __future__ import annotations

import numpy as np

from .diagnose import analyse
from .scenarios import Event, default_fault_size, make_batch

ALARMS = ("broken", "unclear", "quality")


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [0.0, 0.0]
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _rate(k, n):
    return {"k": int(k), "n": int(n), "rate": k / n if n else 0.0, "ci": wilson(k, n)}


def run_suite(profile, n: int = 30, seed: int = 7, progress=None) -> dict:
    rng = np.random.default_rng(seed)
    cfg = profile.cfg
    L = cfg.label
    total_batches = profile.n_batches()

    def batches(events):
        for i in range(n):
            k = int(rng.integers(0, max(total_batches, 1) * 4))
            yield analyse(profile, make_batch(profile, k, events, seed=seed + i), k, detail=False)

    rows = []

    # 1. healthy data: how often does each method raise a false alarm?
    res = list(batches([]))
    rows.append({"group": "Healthy data", "situation": f"Ordinary {cfg.batches_word}",
                 "question": "False alarms (lower is better)",
                 "sentinel": _rate(sum(r["verdict"] in ALARMS for r in res), n),
                 "baseline": _rate(sum(r["baseline"]["alarm"] for r in res), n)})

    # 2. harmless shift: values move, nothing is broken
    for lam in (0.5, 1.0, 1.5):
        res = list(batches([Event("shift", 0, strength=lam)]))
        rows.append({"group": "Harmless shift", "situation": f"{(cfg.shift_story or 'Shift').capitalize()} (strength {lam:g})",
                     "question": "False alarms (lower is better)",
                     "sentinel": _rate(sum(r["verdict"] in ALARMS for r in res), n),
                     "baseline": _rate(sum(r["baseline"]["alarm"] for r in res), n),
                     "note": f"Sentinel called it harmless in {sum(r['verdict'] == 'shift' for r in res)}/{n}"})

    # 3. small faults on each feature: caught? blamed on the right feature?
    for f in profile.features:
        kind, size = default_fault_size(profile, f)
        ev = Event("fault", 0, feature=f, fault=kind, amount=size)
        res = list(batches([ev]))
        caught = sum(r["verdict"] in ALARMS for r in res)
        right = sum((r["verdict"] == "broken" and r["suspect"] == f) or
                    (r["verdict"] == "unclear" and [c["feature"] for c in r["candidates"]] == [f]) for r in res)
        wrong = sum(r["verdict"] == "broken" and r["suspect"] != f for r in res)
        unclear = sum(r["verdict"] == "unclear" and [c["feature"] for c in r["candidates"]] != [f] for r in res)
        rows.append({"group": "Small fault", "situation": ev.describe(cfg).split(" from ")[0],
                     "feature": f, "question": "Caught",
                     "sentinel": _rate(caught, n),
                     "baseline": _rate(sum(r["baseline"]["alarm"] for r in res), n)})
        rows.append({"group": "Small fault", "situation": "", "feature": f,
                     "question": "Blamed the right feature",
                     "sentinel": _rate(right, n),
                     "baseline": _rate(sum(r["baseline"]["blamed"] == f for r in res), n),
                     "note": f"Sentinel said 'unclear' {unclear}×, blamed the wrong feature {wrong}×"})
        if progress:
            progress(f"Tested {L(f)}")

    # 4. a unit bug: should be called a data quality issue, not drift
    f = profile.features[0]
    res = list(batches([Event("fault", 0, feature=f, fault="unit", amount=0.1)]))
    rows.append({"group": "Data quality", "situation": f"{L(f)} recorded in a different unit (× 0.1)",
                 "question": "Named as a data quality issue on the right feature",
                 "sentinel": _rate(sum(r["verdict"] == "quality" and f in r["masked"] for r in res), n),
                 "baseline": _rate(0, n),
                 "note": f"The standard check alarmed {sum(r['baseline']['alarm'] for r in res)}/{n}, but can only call it drift"})

    return {"batches_per_row": n, "seed": seed, "rows": rows,
            "fault_sizes": {f: default_fault_size(profile, f) for f in profile.features}}
