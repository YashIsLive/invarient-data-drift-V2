"""Does the model care? Estimated without knowing the true answers.

The same rule that caught the fault can also say what the faulty feature
*should* have been. We replace the suspect's values with that estimate
("repair"), run the model again, and measure how much predictions change,
relative to the model's usual error:

    impact = mean |prediction(raw) - prediction(repaired)| / usual model error

Repairing healthy data also changes predictions a little (the rule is not
perfect), so we compare against the impact seen on healthy batches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import from_space, to_space


def rule_for(profile, feature: str):
    for r in profile.rels:
        if r.target == feature and r.usable:
            return r
    return None


def repair(profile, raw: pd.DataFrame, feature: str, T: pd.DataFrame | None = None):
    rule = rule_for(profile, feature)
    if rule is None:
        return None
    T = to_space(raw, profile.cfg) if T is None else T
    estimate = from_space(rule.predict(T), feature, profile.cfg)
    repaired = raw.copy()
    ok = np.isfinite(estimate)
    repaired[feature] = repaired[feature].astype(float)
    repaired.loc[ok, feature] = estimate[ok]
    return repaired


def impact_ratio(profile, raw: pd.DataFrame, feature: str, T: pd.DataFrame | None = None,
                 base_pred: np.ndarray | None = None):
    repaired = repair(profile, raw, feature, T)
    if repaired is None:
        return None
    y_raw = profile.model.predict(raw) if base_pred is None else base_pred
    y_rep = profile.model.predict(repaired)
    ratio = float(np.mean(np.abs(y_raw - y_rep)) / profile.mae_ref)
    return {"ratio": ratio, "pred_raw": y_raw, "pred_repaired": y_rep}
