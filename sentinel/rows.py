"""Which rows are affected? A per-row report for the feature a batch's verdict points at.

Broken relationship: each row's recorded value is compared with what the suspect's own rule
predicts from the row's other features:

    standardized residual  e = (recorded - expected) / usual error      (on the comparison scale)

A row is flagged when |e| is above the calibrated row limit (the 99th percentile of |e| on healthy
calibration rows), so about 1 healthy row in 100 is flagged anyway. That is why every report says
"N of M rows flagged; about X expected on healthy days": a small fault shows up in the average of
many rows long before it shows up in single rows, and the report says so instead of overclaiming.

Data quality issue: the exact rows that broke the quality rule (missing, impossible or out of the
plausible range, or stuck at the frozen value) are flagged.

Normal days and harmless shifts flag nothing: nothing is suspected, so no row is singled out.

Incidents: consecutive batches with the same verdict and the same suspect are merged into one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import from_space, to_space
from .diagnose import _delta_text
from .impact import rule_for

COLUMNS = ["row", "recorded", "expected", "percent_off", "standardized_residual", "flagged"]


def _focus(evidence: dict):
    """(feature, kind, note) the report is about, or (None, None, reason) when no row is suspect."""
    verdict = evidence["verdict"]
    if verdict == "quality":
        return evidence["issues"][0]["feature"], "quality", None
    if verdict == "broken":
        return evidence["suspect"], "residual", None
    if verdict == "unclear":
        cands = evidence.get("candidates") or []
        if cands and cands[0].get("explained") is not None and not evidence.get("unverified"):
            return cands[0]["feature"], "residual", (
                f"The cause is unclear; rows are checked against the most likely candidate, {cands[0]['label']}.")
        return None, None, "The feature that moved has no reliable relationship, so single rows cannot be checked."
    return None, None, {
        "normal": "Nothing is wrong, so no row is singled out.",
        "shift": "Values moved but every relationship held: a harmless shift, so no row is singled out.",
        "insufficient": "Too little valid data to check single rows.",
    }.get(verdict, "No row is suspect.")


def _about(x: float) -> str:
    if x < 0.05:
        return "almost none"
    return f"about {x:.0f}" if x >= 1 else f"about {x:.1f}"


def flag_rows(profile, raw: pd.DataFrame, evidence: dict, feature: str | None = None) -> dict:
    """Per-row table for the batch's suspect feature (or for `feature`, if given).

    Returns the feature, its rows (COLUMNS), how many are flagged, how many would be flagged on a
    healthy batch of the same size, and a one-sentence summary."""
    cfg = profile.cfg
    raw = raw.reset_index(drop=True)
    m = len(raw)
    note = None
    if feature is None:
        feature, kind, note = _focus(evidence)
    else:
        kind = "quality" if any(i["feature"] == feature for i in evidence.get("issues", [])) else "residual"
    out = {"feature": feature, "label": cfg.label(feature) if feature else None,
           "unit": cfg.unit(feature) if feature else "", "kind": kind, "verdict": evidence["verdict"],
           "n_rows": m, "n_flagged": 0, "expected": None, "limit": None, "rows": [], "note": note}
    if feature is None:
        out["summary"] = f"0 of {m} rows flagged. {note}"
        return out

    T = to_space(raw, cfg)
    recorded = pd.to_numeric(raw[feature], errors="coerce").to_numpy(dtype=float)
    rule = rule_for(profile, feature)
    masked = set(evidence.get("masked", []))
    if rule is not None and not any(x in masked for x in rule.parents):
        pred = rule.predict(T)
        expected = from_space(pred, feature, cfg)
        resid = (T[feature].to_numpy(dtype=float) - pred) / rule.sigma
    else:
        expected = np.full(m, np.nan)
        resid = np.full(m, np.nan)

    if kind == "quality":
        issue = next(i["issue"] for i in evidence["issues"] if i["feature"] == feature)
        t = T[feature].to_numpy(dtype=float)
        if issue == "missing":
            flagged = np.isnan(recorded)
        elif issue == "out_of_range":
            lo, hi = profile.plausible[feature]
            flagged = (~np.isnan(recorded) & ~np.isfinite(t)) | (np.isfinite(t) & ((t < lo) | (t > hi)))
        else:  # stuck: the rows that carry the frozen value
            frozen = np.nanmedian(recorded)
            flagged = np.isclose(recorded, frozen, rtol=1e-9, atol=0)
        rates = profile.quality_rates.get(feature, {})
        rate = rates.get(issue, 0.0) if issue != "stuck" else 0.0
        why = {"missing": "missing", "out_of_range": "outside the plausible range",
               "stuck": "stuck at the frozen value"}[issue]
        out["issue"] = issue
    else:
        if rule is None:
            out["kind"] = None
            out["summary"] = (f"0 of {m} rows flagged. {cfg.label(feature)} has no reliable relationship, "
                              f"so single rows cannot be checked.")
            return out
        limit = profile.thresholds["row_limit"]["value"].get(feature)
        rate = profile.thresholds["row_limit"]["healthy_rate"].get(feature, 0.01)
        flagged = np.isfinite(resid) & (np.abs(resid) > limit)
        out["limit"] = limit
        why = f"more than {limit:.1f} usual errors from what the other features predict"

    with np.errstate(divide="ignore", invalid="ignore"):
        percent = (recorded - expected) / np.abs(expected) * 100
    n_flagged = int(flagged.sum())
    exp_count = rate * m
    out.update(n_flagged=n_flagged, expected=exp_count)
    out["rows"] = [{"row": i + 1, "recorded": _num(recorded[i]), "expected": _num(expected[i]),
                    "percent_off": _num(percent[i]), "standardized_residual": _num(resid[i]),
                    "flagged": bool(flagged[i])} for i in range(m)]

    days = cfg.batches_word
    text = (f"{n_flagged} of {m} rows flagged ({cfg.label(feature)} {why}); "
            f"{_about(exp_count)} expected on healthy {days}.")
    if kind == "residual" and evidence["verdict"] in ("broken", "unclear") and n_flagged <= 2 * exp_count + 2:
        text += (" The fault is small next to the rule's usual error: it shows in the average over many rows, "
                 "not in single rows, so these flags do not pick out the bad rows.")
    if note:
        text += " " + note
    out["summary"] = text
    return out


def _num(x) -> float | None:
    return float(x) if np.isfinite(x) else None


def to_csv(report: dict) -> str:
    df = pd.DataFrame(report["rows"], columns=COLUMNS)
    df[["percent_off", "standardized_residual"]] = df[["percent_off", "standardized_residual"]].round(3)
    df.insert(1, "feature", report["feature"] or "")
    return df.to_csv(index=False)


# ---------------- incidents ----------------

def day_summary(profile, evidence: dict, report: dict) -> dict:
    """What an incident needs to know about one batch."""
    key = report["feature"] if evidence["verdict"] in ("broken", "unclear", "quality") else None
    delta, mode = None, None
    for c in evidence.get("candidates") or []:
        if c["feature"] == key and c.get("delta") is not None:
            delta = c["delta"]
    if evidence.get("locate"):
        mode = evidence["locate"]["mode"]
    imp = evidence.get("impact") or {}
    return {"index": evidence["index"], "verdict": evidence["verdict"], "suspect": key,
            "delta": delta, "mode": mode, "issue": report.get("issue"),
            "impact": imp.get("ratio") if imp.get("available") else None,
            "material": bool(imp.get("material")) if imp.get("available") else None,
            "n_flagged": report["n_flagged"], "n_rows": report["n_rows"],
            "expected": report["expected"] or 0.0}


def merge_incidents(days: list) -> list:
    """Consecutive batches with the same verdict and suspect become one incident. Normal batches
    end an incident and are not incidents themselves. `days` must be in batch order."""
    incidents = []
    for d in days:
        if d["verdict"] == "normal":
            continue
        last = incidents[-1] if incidents else None
        if last and last["end"] == d["index"] - 1 and last["verdict"] == d["verdict"] and last["suspect"] == d["suspect"]:
            last["end"] = d["index"]
            last["days"].append(d)
        else:
            incidents.append({"start": d["index"], "end": d["index"], "verdict": d["verdict"],
                              "suspect": d["suspect"], "days": [d]})
    for inc in incidents:
        ds = inc.pop("days")
        deltas = [d["delta"] for d in ds if d["delta"] is not None]
        impacts = [d["impact"] for d in ds if d["impact"] is not None]
        inc.update(
            batches=len(ds),
            delta=float(np.median(deltas)) if deltas else None,
            mode=next((d["mode"] for d in ds if d["mode"]), None),
            issue=next((d["issue"] for d in ds if d["issue"]), None),
            impact=float(np.mean(impacts)) if impacts else None,
            material=any(d["material"] for d in ds) if impacts else None,
            rows_flagged=int(sum(d["n_flagged"] for d in ds)),
            rows=int(sum(d["n_rows"] for d in ds)),
            rows_expected=float(sum(d["expected"] for d in ds)),
        )
    return incidents


def describe_incidents(profile, incidents: list) -> list:
    """Add labels and a plain-language estimated size to merged incidents."""
    cfg = profile.cfg
    for inc in incidents:
        f = inc["suspect"]
        inc["label"] = cfg.label(f) if f else None
        if inc["verdict"] == "quality":
            inc["size_text"] = {"missing": "missing values", "out_of_range": "values outside the plausible range",
                                "stuck": "stuck at one value"}.get(inc["issue"], "invalid values")
        elif f and inc["delta"] is not None:
            inc["size_text"] = _delta_text(profile, f, inc["delta"], inc["mode"] or "shift")
        else:
            inc["size_text"] = None
    return incidents
