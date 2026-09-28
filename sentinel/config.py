"""Dataset configuration.

Everything dataset-specific lives in one small JSON file (see datasets/*.json).
The rest of the code never mentions diamonds or taxis, which is what lets the
same system run on any CSV with a few related numeric columns.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    name: str                       # short id, used for folder names
    title: str                      # shown in the interface
    csv: str                        # path to the CSV, relative to the project root
    target: str                     # what the model predicts
    monitored: list                 # numeric columns whose relationships we learn and watch
    context: list = field(default_factory=list)       # extra model inputs we do not monitor
    time_column: str | None = None                   # if set, data is streamed in time order
    log_features: list = field(default_factory=list)  # compared on a log (percentage) scale
    batch_size: int = 200
    shift_driver: str | None = None  # feature used to simulate a harmless shift
    shift_story: str = ""            # plain-language name for that shift
    batch_word: str = "batch"
    description: str = ""
    labels: dict = field(default_factory=dict)  # column -> human name
    units: dict = field(default_factory=dict)   # column -> unit symbol
    scenarios: list = field(default_factory=list)  # optional hand-written demo scenarios

    def label(self, col: str) -> str:
        if col in self.labels:
            return self.labels[col]
        # column names like "machine_load" read better in sentences as "Machine load"
        text = str(col).replace("_", " ").strip()
        return text[:1].upper() + text[1:] if text else str(col)

    def unit(self, col: str) -> str:
        return self.units.get(col, "")

    @property
    def batches_word(self) -> str:
        w = self.batch_word
        return w + ("es" if w.endswith(("s", "ch", "sh", "x")) else "s")

    @property
    def driver(self) -> str:
        return self.shift_driver or self.monitored[0]

    @classmethod
    def load(cls, path: str | Path) -> "DatasetConfig":
        data = json.loads(Path(path).read_text())
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def to_dict(self) -> dict:
        return asdict(self)


def should_use_log(values: pd.Series) -> bool:
    """Suggest a log scale for strictly positive columns that span a wide range
    (sizes, prices, durations). On a log scale 'reads 5% high' is a constant
    shift, and multiplicative laws such as weight ~ length^3 become straight lines."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 20 or (v <= 0).any():
        return False
    lo, hi = np.percentile(v, [1, 99])
    return bool(lo > 0 and hi / lo >= 2.0)


def suggest_config(df: pd.DataFrame, name: str, csv: str) -> dict:
    """First guess of a configuration for an uploaded CSV. The user edits it."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    other = [c for c in df.columns if c not in numeric]
    time_col = None
    for c in other:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            time_col = c
            break
    target = numeric[-1] if numeric else None
    monitored = [c for c in numeric if c != target and df[c].nunique() > 10]
    context = [c for c in df.columns if c not in monitored + [target, time_col]
               and df[c].nunique() <= 50]
    driver = None
    if target and monitored:
        corr = df[monitored + [target]].corr()[target].drop(target).abs()
        driver = str(corr.idxmax()) if corr.notna().any() else monitored[0]
    batch = int(np.clip(len(df) * 0.3 / 25, 50, 300))
    return {
        "name": name, "title": name.replace("_", " ").capitalize(), "csv": csv,
        "target": target, "monitored": monitored, "context": context,
        "time_column": time_col,
        "log_features": [c for c in monitored if should_use_log(df[c])],
        "batch_size": batch, "shift_driver": driver,
        "columns": [{"name": c, "numeric": c in numeric, "unique": int(df[c].nunique()),
                     "missing": float(df[c].isna().mean())} for c in df.columns],
    }
