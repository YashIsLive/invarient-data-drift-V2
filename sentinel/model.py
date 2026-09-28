"""The model we are protecting (e.g. the diamond price model).

A gradient-boosted tree model: strong on tables, fast, tolerates missing values.
Text columns (cut, colour, payment type ...) are turned into category codes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import DatasetConfig


class TargetModel:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.inputs = list(cfg.monitored) + [c for c in cfg.context if c not in cfg.monitored]
        self.categories: dict = {}
        self.model = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.inputs:
            if c in self.categories:
                lookup = self.categories[c]
                cols.append(_as_text(df[c]).map(lookup).astype(float).to_numpy())
            else:
                cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
        return np.column_stack(cols)

    def fit(self, train: pd.DataFrame) -> "TargetModel":
        for c in self.inputs:
            if not pd.api.types.is_numeric_dtype(train[c]):
                values = sorted(_as_text(train[c]).unique())[:250]
                self.categories[c] = {v: i for i, v in enumerate(values)}
        is_cat = np.array([c in self.categories for c in self.inputs])
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, categorical_features=is_cat if is_cat.any() else None,
            random_state=0)
        self.model.fit(self._matrix(train), train[self.cfg.target].to_numpy(dtype=float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._matrix(df))


def _as_text(col: pd.Series) -> pd.Series:
    return col.astype(object).where(col.notna(), "(missing)").astype(str)
