"""Analyse one batch: run the checks, decide a verdict, and explain it in plain words.

Decision order (first match wins):

    quality issue           -> QUALITY        some feature is not valid data
    too little valid data   -> INSUFFICIENT
    a relationship broke    -> BROKEN         one feature clearly explains the pattern
                               UNCLEAR        several features could explain it
    values moved            -> SHIFT          harmless: everything moved together
    otherwise               -> NORMAL

The standard drift check we compare against is the drift step alone: it alarms
whenever any feature's distribution moved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import checks
from .checks import fmt
from .data import from_space, to_space
from .impact import impact_ratio, rule_for
from .locate import locate
from .profile import Profile, _unfamiliar_share

VERDICTS = {
    "normal": "All normal",
    "shift": "Harmless shift",
    "quality": "Data quality issue",
    "broken": "Broken relationship",
    "unclear": "Unclear break",
    "insufficient": "Not enough data",
}


def analyse(p: Profile, raw: pd.DataFrame, index: int = 0, detail: bool = True) -> dict:
    cfg, thr = p.cfg, p.thresholds
    raw = raw.reset_index(drop=True)
    T0 = to_space(raw, cfg)

    issues, masked, notes, T = checks.quality_check(p, raw, T0)
    drift = checks.drift_check(p, T, masked)
    baseline = _baseline(p, T0)
    stats = checks.rule_stats(p.rels, T, masked, p.min_rows)
    checked = [s for s in stats if not s["skipped"]]
    z_lim, s_lim = thr["z_limit"]["value"], thr["spread_limit"]["value"]
    for s in checked:
        s["shift_broken"] = abs(s["z"]) > z_lim
        s["spread_broken"] = s["spread"] > s_lim
    broken_shift = [s for s in checked if s["shift_broken"]]
    broken_spread = [s for s in checked if s["spread_broken"]]
    moved = [d for d in drift if d["moved"]]
    unfamiliar = _unfamiliar_share(p, T)

    out = {"index": index, "rows": int(len(raw)), "issues": issues, "masked": sorted(masked),
           "notes": notes, "baseline": baseline, "unfamiliar": unfamiliar,
           "unfamiliar_limit": thr["unfamiliar_share"]["value"], "suspect": None,
           "candidates": [], "impact": None, "locate": None}

    valid_rows = int(T[[c for c in p.features if c not in masked]].notna().all(axis=1).sum()) \
        if len(masked) < len(p.features) else 0
    if valid_rows < p.min_rows or (not checked and p.usable and not issues):
        verdict = "insufficient"
    elif issues:
        verdict = "quality"
    elif broken_shift or broken_spread:
        mode = "shift" if broken_shift else "spread"
        loc = locate(p, stats, masked, mode)
        out["locate"] = loc
        cands = loc["candidates"]
        if cands:
            best = cands[0]
            second = cands[1]["explained"] if len(cands) > 1 else 0.0
            clear = best["explained"] >= thr["explained_min"]["value"] and \
                best["explained"] - second >= thr["margin_min"]["value"]
            verdict = "broken" if clear else "unclear"
            out["suspect"] = best["feature"] if clear else None
            out["candidates"] = [{"feature": c["feature"], "label": cfg.label(c["feature"]),
                                  "explained": c["explained"], "delta": c["delta"],
                                  "delta_text": _delta_text(p, c["feature"], c["delta"], mode)}
                                 for c in cands[:5]]
        else:
            verdict = "unclear"
    elif moved:
        # a feature with no reliable relationship of its own cannot be vouched for
        covered = {r.target for r in p.usable}
        unchecked = [d["feature"] for d in moved if d["feature"] not in covered]
        if unchecked:
            verdict = "unclear"
            out["unverified"] = unchecked
            out["candidates"] = [{"feature": f, "label": cfg.label(f), "explained": None, "delta": None,
                                  "delta_text": ""} for f in unchecked]
        else:
            verdict = "shift"
    else:
        verdict = "normal"

    # the model's real error on this batch (in real life known only later, when true values arrive)
    y_true = pd.to_numeric(raw[cfg.target], errors="coerce").to_numpy(dtype=float)
    pred = p.model.predict(raw)
    ok = np.isfinite(y_true)
    out["mae"] = float(np.mean(np.abs(pred[ok] - y_true[ok]))) if ok.any() else None
    out["mae_ref"] = p.mae_ref

    focus = out["suspect"] or (out["candidates"][0]["feature"] if out["candidates"] else None)
    if focus and verdict in ("broken", "unclear"):
        out["impact"] = _impact(p, raw, T, focus, pred, y_true, ok, detail)

    out["verdict"] = verdict
    out["title"] = VERDICTS[verdict]
    out.update(_explain(p, out, verdict, drift, checked, moved, masked))
    out["confidence"] = _confidence(p, out, verdict, checked)
    if detail:
        out["drift"] = [{**d, "label": cfg.label(d["feature"]),
                         "usual": p.typical[d["feature"]],
                         "now": float(np.nanmedian(from_space(T[d["feature"]], d["feature"], cfg)))
                         if T[d["feature"]].notna().any() else None} for d in drift]
        out["rules"] = _rule_table(p, stats)
        out["scatter"] = _scatter(p, raw, T, masked)
        if out["locate"]:
            out["locate"]["rule_names"] = [p.usable[i].name for i in out["locate"]["rules"]]
            for c in out["locate"]["candidates"]:
                c["label"] = cfg.label(c["feature"])
    else:
        out.pop("locate", None)
    return out


def _baseline(p: Profile, T0: pd.DataFrame) -> dict:
    """Standard drift check: KS per feature, no quality gate, no relationships."""
    scores = []
    for c in p.features:
        ks = checks.ks_stat(p.ref_sorted[c], T0[c].to_numpy(dtype=float))
        if np.isfinite(ks):
            scores.append((ks / p.ks_limit[c], c))
    moved = [c for s, c in scores if s > 1]
    blamed = max(scores)[1] if moved else None
    return {"alarm": bool(moved), "moved": moved, "blamed": blamed}


def _impact(p, raw, T, feature, pred, y_true, ok, detail):
    res = impact_ratio(p, raw, feature, T, base_pred=pred)
    if res is None:
        return {"feature": feature, "available": False}
    null = p.impact_null.get(feature, 0.0)
    material = res["ratio"] >= max(null, p.thresholds["impact_material"]["value"])
    info = {"feature": feature, "available": True, "ratio": res["ratio"], "null_p95": null,
            "material": bool(material),
            "mean_change": float(np.mean(np.abs(res["pred_raw"] - res["pred_repaired"])))}
    if ok.any():
        info["mae_raw"] = float(np.mean(np.abs(res["pred_raw"][ok] - y_true[ok])))
        info["mae_repaired"] = float(np.mean(np.abs(res["pred_repaired"][ok] - y_true[ok])))
    if detail:
        n = min(80, len(raw))
        info["series"] = {"actual": _clean(y_true[:n]), "raw": _clean(res["pred_raw"][:n]),
                          "repaired": _clean(res["pred_repaired"][:n])}
    return info


def _amount(x: float, unit: str) -> str:
    """A number with its unit: $83, 0.17 mm, 1.2 percentage points."""
    if unit == "$":
        return f"${fmt(x)}"
    if unit == "%":
        return f"{fmt(x)} percentage points"
    return f"{fmt(x)} {unit}".strip()


def _delta_text(p: Profile, feature: str, d: float, mode: str) -> str:
    """How the suspect is wrong, phrased to follow the feature name: 'Length reads ...'."""
    cfg = p.cfg
    unit = cfg.unit(feature)
    if mode == "spread":
        extra = float(np.sqrt(max(d, 0)))
        if feature in cfg.log_features:
            return f"is noisier than usual (extra scatter of about ±{(np.exp(extra) - 1) * 100:.1f}%)"
        return f"is noisier than usual (extra scatter of about ±{_amount(extra, unit)})"
    word = "high" if d > 0 else "low"
    if feature in cfg.log_features:
        pct = (np.exp(d) - 1) * 100
        absolute = p.typical[feature] * (np.exp(d) - 1)
        return (f"reads about {abs(pct):.1f}% too {word} "
                f"({'+' if absolute > 0 else '−'}{_amount(abs(absolute), unit)} on a typical value)")
    return f"reads about {_amount(abs(d), unit)} too {word}"


def _names(labels: list) -> str:
    labels = list(dict.fromkeys(labels))
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _explain(p, out, verdict, drift, checked, moved, masked) -> dict:
    cfg = p.cfg
    L = cfg.label
    bw = cfg.batch_word
    n_rules = len(p.usable)
    paragraphs, checks_list = [], []
    change_mind = None
    moved_names = _names([L(d["feature"]) for d in moved])

    if verdict == "insufficient":
        headline = "Too little valid data to judge."
        paragraphs.append("Most rows are missing or invalid, so the relationship checks cannot run reliably.")
        checks_list = ["Check whether the data source was offline or delivered a partial file."]

    elif verdict == "quality":
        first = out["issues"][0]
        headline = f"{L(first['feature'])} values are invalid." if len(out["issues"]) == 1 \
            else f"{_names([L(i['feature']) for i in out['issues']])} values are invalid."
        paragraphs += [i["text"] for i in out["issues"]]
        paragraphs.append(f"{'This feature was' if len(masked) == 1 else 'These features were'} left out "
                          f"of the other checks, so the problem does not show up as fake drift.")
        rest = [s for s in checked if s.get("shift_broken") or s.get("spread_broken")]
        if rest:
            paragraphs.append(f"Among the remaining features, {len(rest)} relationship(s) also look broken; "
                              f"fix the data problem first, then re-check.")
        for i in out["issues"]:
            name = L(i["feature"])
            if i["issue"] == "missing":
                checks_list.append(f"Check the source that delivers {name}: offline device, failed join, or an empty column.")
            elif i["issue"] == "out_of_range":
                checks_list.append(f"Check the units of {name} (for example mm vs cm) and any recent format or software change.")
            else:
                checks_list.append(f"Check whether the {name} reading is frozen: a dead sensor or a default value being filled in.")

    elif verdict == "normal":
        headline = f"All {n_rules} relationships hold and nothing moved."
        paragraphs.append(f"Every feature looks like the reference data and still agrees with the others.")

    elif verdict == "shift":
        headline = f"{moved_names} moved, but the features still agree with each other."
        paragraphs.append(f"{len(moved)} of {len(drift)} features changed their distribution, yet all "
                          f"{len(checked)} checked relationships hold. The world changed; the data is still consistent. "
                          f"A standard drift check would raise {len(out['baseline']['moved'])} alarm(s) here.")
        if out["unfamiliar"] > out["unfamiliar_limit"]:
            paragraphs.append(f"Note: {out['unfamiliar']:.1%} of rows lie outside the range the model was "
                              f"trained on (usually at most {out['unfamiliar_limit']:.1%}). The data is fine, "
                              f"but the model is working in less familiar territory, so watch its error.")
            checks_list.append("If the shift lasts, retrain the model so it sees more of these cases.")
        checks_list.insert(0, "No data fix needed. Confirm the change matches something real (season, new customers, a promotion).")
        change_mind = "If nothing in the real world explains the move, check whether every source changed at once (a firmware or pipeline update can shift all features together, which relationships cannot see)."

    elif verdict in ("broken", "unclear"):
        broken_rules = [p.usable[s["rule"]] for s in checked if s["shift_broken"] or s["spread_broken"]]
        nb = len(broken_rules)
        cands = out["candidates"]
        if verdict == "broken":
            c = cands[0]
            f = c["feature"]
            involved = [r for r in broken_rules if f in (r.target, *r.parents)]
            neighbours = [x for r in involved for x in (r.target, *r.parents) if x != f]
            headline = f"{L(f)} no longer agrees with {_names([L(x) for x in neighbours]) or 'the other features'}."
            paragraphs.append(f"{nb} of {len(checked)} relationships stopped holding"
                              f"{', and every one of them involves ' + L(f) if len(involved) == nb else ''}. "
                              f"{L(f)} {c['delta_text']} compared with what the other features say it should be.")
            runner = f"; the next best explanation ({cands[1]['label']}) explains {cands[1]['explained']:.0%}" if len(cands) > 1 else ""
            paragraphs.append(f"A single fault on {L(f)} explains {c['explained']:.0%} of the pattern{runner}.")
            drifted = any(d["feature"] == f and d["moved"] for d in drift)
            paragraphs.append(f"{L(f)}'s values on their own {'did' if drifted else 'did not'} move enough for a "
                              f"standard drift check to flag them"
                              f"{'' if drifted else ' — they still look like normal values'}.")
            if out["locate"]["mode"] == "spread":
                checks_list += [f"Check the {L(f)} measurement for a loose connection, failing device or noisy source.",
                                f"Look for rows of {L(f)} merged from a different, less precise source."]
            else:
                checks_list += [f"Check the calibration of whatever measures {L(f)}.",
                                f"Look for recent changes to how {L(f)} is recorded: units, rounding, device or software updates.",
                                f"Compare a few recent records of {L(f)} against the original source."]
            change_mind = (f"If {L(f)} is verified against the source and is correct, the likelier story is a real "
                           f"change in how the data is produced, and the reference profile should be rebuilt.")
        elif out.get("unverified"):
            names = _names([L(f) for f in out["unverified"]])
            others = [d for d in moved if d["feature"] not in out["unverified"]]
            headline = f"{names} moved, and no relationship can confirm the change is harmless."
            if others:
                paragraphs.append(f"{_names([L(d['feature']) for d in others])} moved too, and they still agree "
                                  f"with each other, so that part is a harmless shift.")
            paragraphs.append(f"{names} {'has' if len(out['unverified']) == 1 else 'have'} no reliable relationship with "
                              f"the other features (see Relationships), so the system cannot tell a real change "
                              f"from a fault. Here it can only do what a standard drift check does.")
            checks_list += [f"Check {L(f)} against its source." for f in out["unverified"]]
            change_mind = f"If {names} changed for a known real reason, this is harmless."
        else:
            group = [x for x in cands if x["explained"] >= cands[0]["explained"] - p.thresholds["margin_min"]["value"]] if cands else []
            headline = (f"{nb} relationship(s) broke, but the cause could be {_names([x['label'] for x in group[:3]])}."
                        if group else f"{nb} relationship(s) broke, and no single feature explains it.")
            if cands and cands[0]["explained"] < p.thresholds["explained_min"]["value"]:
                paragraphs.append(f"No single feature explains the pattern well (best: {cands[0]['label']}, "
                                  f"{cands[0]['explained']:.0%}). Several things may be wrong at once, or features may have been swapped.")
            else:
                paragraphs.append("Two or more features leave almost the same fingerprint, so the system does not pick one.")
            for x in group[:3]:
                paragraphs.append(f"If it is {x['label']}: {x['label']} {x['delta_text']} ({x['explained']:.0%} of the pattern explained).")
            checks_list += [f"Check {x['label']} first." for x in group[:3]] or ["Inspect the broken relationships one by one."]
            checks_list.append("Look for pipeline changes that could affect several columns at once (renamed or swapped columns).")
            change_mind = "Checking one candidate and finding it correct points to the other."

        imp = out.get("impact")
        if imp and imp.get("available"):
            txt = (f"Does the model care? Replacing {L(imp['feature'])} with the value its neighbours suggest changes "
                   f"{L(cfg.target)} predictions by {_amount(imp['mean_change'], cfg.unit(cfg.target))} on average, which is "
                   f"{imp['ratio']:.2f}× the model's usual error ({_amount(p.mae_ref, cfg.unit(cfg.target))}). On healthy {cfg.batches_word} this is typically "
                   f"below {imp['null_p95']:.2f}×, so the impact is {'material' if imp['material'] else 'low'}.")
            paragraphs.append(txt)
        elif imp:
            paragraphs.append(f"{L(imp['feature'])} has no reliable relationship of its own, so the impact cannot be estimated.")

    return {"headline": headline, "paragraphs": paragraphs, "checks": checks_list, "change_mind": change_mind}


def _confidence(p, out, verdict, checked) -> str:
    if verdict in ("quality", "normal", "insufficient"):
        return "high"
    zl = p.thresholds["z_limit"]["value"]
    if verdict == "shift":
        worst = max((abs(s["z"]) for s in checked), default=0)
        return "high" if worst < 0.7 * zl else "medium"
    if verdict == "unclear":
        return "low"
    c = out["candidates"]
    margin = c[0]["explained"] - (c[1]["explained"] if len(c) > 1 else 0)
    strength = max((abs(s["z"]) / zl for s in checked), default=0)
    if out["locate"]["mode"] == "spread":
        strength = max(s["spread"] for s in checked) / p.thresholds["spread_limit"]["value"]
    if c[0]["explained"] >= 0.85 and margin >= 0.4 and strength >= 1.5:
        return "high"
    if c[0]["explained"] < 0.8 or margin < 0.3 or strength < 1.2:
        return "low"
    return "medium"


def _rule_table(p, stats) -> list:
    cfg = p.cfg
    rows = []
    by_rule = {s["rule"]: s for s in stats}
    for i, r in enumerate(p.usable):
        s = by_rule.get(i, {"skipped": True})
        rows.append({"name": r.name, "target": r.target, "parents": r.parents,
                     "label": f"{cfg.label(r.target)} from {_names([cfg.label(x) for x in r.parents])}",
                     "skipped": s["skipped"], "z": s.get("z"), "spread": s.get("spread"),
                     "mean": s.get("mean"),
                     "shift_broken": bool(s.get("shift_broken", False)),
                     "spread_broken": bool(s.get("spread_broken", False))})
    return rows


def _scatter(p, raw, T, masked) -> dict:
    """Expected-vs-actual points for every feature that has a usable rule."""
    cfg = p.cfg
    out = {}
    Tc = to_space(p.cloud, cfg)
    for r in p.usable:
        f = r.target
        if f in masked or any(x in masked for x in r.parents):
            continue
        exp_now = from_space(r.predict(T), f, cfg)
        act_now = from_space(T[f].to_numpy(dtype=float), f, cfg)
        exp_ref = from_space(r.predict(Tc), f, cfg)
        act_ref = from_space(Tc[f].to_numpy(dtype=float), f, cfg)
        out[f] = {"label": cfg.label(f), "unit": cfg.unit(f), "log": f in cfg.log_features,
                  "ref": {"x": _clean(exp_ref), "y": _clean(act_ref)},
                  "now": {"x": _clean(exp_now), "y": _clean(act_now)}}
    return out


def _clean(a) -> list:
    return [None if not np.isfinite(v) else round(float(v), 5) for v in np.asarray(a, dtype=float)]
