"""Relationships between features: the heart of the project.

For every monitored feature we learn a small straight-line rule that predicts it
from two or three other features, e.g. on a log scale

    weight  ~  3.0 x length  + constant      ("weight grows with length cubed")

A rule's *residual* is actual minus predicted, divided by the rule's usual
error (sigma). On healthy data residuals hover around 0 even when every value
moves (a harmless shift). When one feature is wrong, the rules that involve it
stop holding, and which rules break (and in which direction) points at it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class Relationship:
    target: str
    parents: list
    intercept: float
    coefs: list
    sigma: float      # usual size of the error on calibration data
    r2: float         # how well the rule fits calibration data (1.0 = perfect)
    usable: bool      # rules that fit poorly are shown but never used for decisions
    spread_ref: float = 1.0  # usual robust scatter of the residual on calibration data

    def predict(self, T: pd.DataFrame) -> np.ndarray:
        X = T[self.parents].to_numpy(dtype=float)
        return self.intercept + X @ np.asarray(self.coefs)

    def residual(self, T: pd.DataFrame) -> np.ndarray:
        """Standardized residual: 0 means 'exactly as expected', 1 means 'one usual error off'."""
        return (T[self.target].to_numpy(dtype=float) - self.predict(T)) / self.sigma

    @property
    def name(self) -> str:
        return f"{self.target} ~ {' + '.join(self.parents)}"


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0


def learn(train: pd.DataFrame, calib: pd.DataFrame, features: list,
          max_parents: int = 3, min_gain: float = 0.005, min_r2: float = 0.5) -> list:
    """Greedy forward selection: add the parent that improves calibration R² the most,
    stop when the gain is tiny or we reach max_parents. Few parents keep each
    feature's 'fingerprint' distinct (if every rule used every feature, every
    fault would disturb every rule and all fingerprints would look alike)."""
    rels = []
    for target in features:
        parents, best = [], 0.0
        while len(parents) < max_parents:
            scores = []
            for c in features:
                if c == target or c in parents:
                    continue
                cols = parents + [c]
                m = LinearRegression().fit(train[cols], train[target])
                scores.append((_r2(calib[target].to_numpy(), m.predict(calib[cols])), c))
            if not scores:
                break
            r2, c = max(scores)
            if r2 - best < min_gain:
                break
            parents.append(c)
            best = r2
        if not parents:  # nothing helps: keep a placeholder rule that is never used
            rels.append(Relationship(target, [], float(train[target].mean()), [],
                                     float(train[target].std() or 1.0), 0.0, False))
            continue
        m = LinearRegression().fit(train[parents], train[target])
        pred = m.predict(calib[parents])
        sigma = float(np.std(calib[target].to_numpy() - pred)) or 1e-9
        resid = (calib[target].to_numpy() - pred) / sigma
        spread_ref = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) or 1.0
        rels.append(Relationship(target, parents, float(m.intercept_),
                                 [float(b) for b in m.coef_], sigma, best, best >= min_r2, spread_ref))
    return rels


def signature_matrix(rels: list, features: list) -> np.ndarray:
    """Expected residual shift (in sigma units) of each usable rule when a feature
    reads +1 unit too high:

        +1/sigma        if the feature is the rule's target (actual goes up)
        -coef/sigma     if the feature is one of the rule's parents (prediction goes up)
        0               if the rule does not involve the feature

    Rows = usable rules, columns = features. Column j is feature j's fingerprint."""
    usable = [r for r in rels if r.usable]
    S = np.zeros((len(usable), len(features)))
    for i, r in enumerate(usable):
        S[i, features.index(r.target)] = 1.0 / r.sigma
        for p, b in zip(r.parents, r.coefs):
            S[i, features.index(p)] = -b / r.sigma
    return S


def isolability(S: np.ndarray) -> np.ndarray:
    """|cosine similarity| between fingerprints. Close to 1 means two features
    would leave the same pattern, so the system cannot tell them apart.
    Known before any monitoring starts."""
    k = S.shape[1]
    norms = np.linalg.norm(S, axis=0)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if norms[i] > 0 and norms[j] > 0:
                M[i, j] = abs(S[:, i] @ S[:, j]) / (norms[i] * norms[j])
    return M
