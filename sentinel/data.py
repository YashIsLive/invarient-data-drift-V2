"""Loading, splitting and cleaning data, and moving values to the comparison scale."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig

def read_table(path) -> pd.DataFrame:
    """Read a CSV whatever its separator: comma, semicolon (common in European files such as the
    UCI wine data), tab or pipe. The separator that gives the most columns wins."""
    best = None
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
        except Exception:
            continue
        if best is None or df.shape[1] > best.shape[1]:
            best = df
    if best is None:
        raise ValueError("Could not read this file as a table.")
    best.columns = [str(c).strip() for c in best.columns]
    return best
def load_frame(cfg: DatasetConfig, root: Path) -> pd.DataFrame:
    path = Path(cfg.csv)
    if not path.is_absolute():
        path = root / path
    df = read_table(path)
    missing = [c for c in [cfg.target, *cfg.monitored, *cfg.context] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {path.name}: {', '.join(missing)}")
    for c in [cfg.target, *cfg.monitored]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if cfg.time_column:
        df[cfg.time_column] = pd.to_datetime(df[cfg.time_column], errors="coerce")
        df = df.sort_values(cfg.time_column, kind="stable")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, cfg: DatasetConfig, seed: int = 42) -> dict:
    """50% train, 20% calibration, 30% stream.

    With a time column the split is chronological (the stream is the future).
    Without one, rows are shuffled once with a fixed seed.
    """
    if not cfg.time_column:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.5), int(n * 0.7)
    return {"train": df.iloc[:a].reset_index(drop=True),
            "calib": df.iloc[a:b].reset_index(drop=True),
            "stream": df.iloc[b:].reset_index(drop=True)}


def clean_reference(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, dict]:
    """Keep the healthy reference boring, so a few broken rows cannot bend the rules we learn.

    Drops rows with missing target/monitored values, impossible values for
    log-scale columns (zero or negative sizes), and the most extreme 0.1% at
    each end of every monitored column.
    """
    report = {"rows_in": int(len(df))}
    keep = df[[cfg.target, *cfg.monitored]].notna().all(axis=1)
    report["missing"] = int((~keep).sum())
    impossible = np.zeros(len(df), dtype=bool)
    for c in cfg.log_features:
        impossible |= (df[c] <= 0).to_numpy()
    report["impossible"] = int((impossible & keep.to_numpy()).sum())
    keep &= ~impossible
    extreme = np.zeros(len(df), dtype=bool)
    for c in cfg.monitored:
        lo, hi = df.loc[keep, c].quantile([0.001, 0.999])
        extreme |= ((df[c] < lo) | (df[c] > hi)).to_numpy()
    report["extreme"] = int((extreme & keep.to_numpy()).sum())
    keep &= ~extreme
    out = df[keep].reset_index(drop=True)
    report["rows_out"] = int(len(out))
    return out, report


def to_space(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Monitored columns on the comparison scale (log where configured).
    Impossible values (<= 0 on a log column) become NaN; the quality check reports them."""
    out = pd.DataFrame(index=df.index)
    for c in cfg.monitored:
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if c in cfg.log_features:
            v = np.log(v.where(v > 0))
        out[c] = v
    return out


def from_space(values, feature: str, cfg: DatasetConfig):
    values = np.asarray(values, dtype=float)
    return np.exp(values) if feature in cfg.log_features else values
