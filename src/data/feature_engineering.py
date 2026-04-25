"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd


def add_time_delta(df: pd.DataFrame, time_col: str = "charttime") -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        ts = pd.to_datetime(out[time_col])
        out["delta_t_hours"] = ts.diff().dt.total_seconds().fillna(0.0) / 3600.0
    else:
        out["delta_t_hours"] = 0.0
    return out
