"""Preprocessing placeholders for MIMIC-IV tabular exports."""

from __future__ import annotations

import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and forward-fill missing values by stay_id if present."""
    out = df.drop_duplicates().copy()
    if "stay_id" in out.columns:
        out = out.sort_values(["stay_id", "charttime"] if "charttime" in out.columns else ["stay_id"])
        out = out.groupby("stay_id", group_keys=False).ffill()
    else:
        out = out.ffill()
    return out
