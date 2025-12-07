# src/data/calendar.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .series_config import SeriesMeta, load_series_meta


# -----------------------------------------------------------------------------
# 基础日期工具
# -----------------------------------------------------------------------------


def period_end(ref_period: pd.Timestamp, freq: str) -> pd.Timestamp:
    """
    Map a ref_period + freq into the end of that period.

    Parameters
    ----------
    ref_period : pd.Timestamp
        The reference period as used in raw data (e.g. 1980-01-01 for month,
        1980-01-01 for 1980Q1).
    freq : {"Q", "M", "W", "D"}
        Frequency code from SeriesMeta.

    Returns
    -------
    pd.Timestamp
        The period end date.

    Notes
    -----
    - For "Q":
        We interpret ref_period as the first day of the quarter and use
        pandas Period with quarterly frequency to get the quarter end.
    - For "M":
        We interpret ref_period as a day within the month and use month end.
    - For "W":
        We assume ref_period is already the week-end date used by FRED and
        return it unchanged.
    - For "D":
        We return ref_period itself.
    """
    freq = freq.upper()

    if freq == "Q":
        # Interpret as quarterly period, return its end date
        per = ref_period.to_period("Q")
        return per.asfreq("D", "end").to_timestamp()

    if freq == "M":
        # Monthly period end
        per = ref_period.to_period("M")
        return per.asfreq("D", "end").to_timestamp()

    if freq == "W":
        # FRED weekly data typically uses a week-end date already
        return ref_period.normalize()

    if freq == "D":
        return ref_period.normalize()

    raise ValueError(f"Unsupported frequency for period_end: {freq}")


def _compute_release_base_date(ref_period: pd.Timestamp, meta: SeriesMeta) -> pd.Timestamp:
    """
    Compute the base date from which release_lag_days will be applied.

    In most cases:
      - For quarterly/monthly data: base = period_end(ref_period, freq)
      - For weekly/daily data:      base = ref_period (already a specific date)

    Special cases (if needed later) can be keyed off meta.release_rule.
    """
    freq = meta.freq.upper()

    if freq in ("Q", "M"):
        return period_end(ref_period, freq)

    if freq in ("W", "D"):
        return ref_period.normalize()

    # Fallback: use ref_period as-is
    return ref_period.normalize()


def compute_release_date_for_row(
    ref_period: pd.Timestamp,
    meta: SeriesMeta,
) -> pd.Timestamp:
    """
    Compute the theoretical economic release date for a single observation,
    based on:
      - ref_period
      - meta.freq
      - meta.release_rule
      - meta.release_lag_days

    For now we use a simple rule:
      - base_date = quarter/month end (for Q/M), or ref_period (for W/D)
      - release_date = base_date + release_lag_days

    The release_rule field is kept for future refinement (e.g. exact BEA/BLS
    calendars), but in this first version we do not distinguish rules beyond
    freq + lag.
    """
    base = _compute_release_base_date(ref_period, meta)
    return base + timedelta(days=int(meta.release_lag_days))


def compute_available_date(
    release_date: pd.Timestamp,
    vintage_date: pd.Timestamp | None,
) -> pd.Timestamp:
    """
    Baseline version:
      - For nowcast and backtest, we approximate the information arrival date
        by the theoretical economic release date only.

    We still keep vintage_date in the data for future revision analysis, but
    do NOT delay availability when ALFRED only started storing old history
    decades later.
    """
    return release_date


# -----------------------------------------------------------------------------
# 主函数：对单个 series 的 DataFrame 增加日历列
# -----------------------------------------------------------------------------


def add_calendar_columns_for_series(
    df: pd.DataFrame,
    meta: SeriesMeta,
) -> pd.DataFrame:
    """
    Given a raw DataFrame for a single series (as returned by loaders.py),
    add calendar-related columns:

      - period_end     : end-of-period date for ref_period
      - release_date   : theoretical economic release date
      - available_date : earliest date when this value can be used in backtests

    Input conventions
    -----------------
    For vintage series (meta.vintage_source == "alfred"):
        df columns: ['ref_period', 'vintage_date', 'value']

    For final series (vintage_source != "alfred"):
        df columns: ['ref_period', 'value']

    Output
    ------
    pd.DataFrame
        Original df with three extra columns:
            - 'period_end'     (datetime64[ns])
            - 'release_date'   (datetime64[ns])
            - 'available_date' (datetime64[ns])

    Notes
    -----
    - For vintage series, available_date is computed as
          max(release_date, vintage_date)
      row-by-row.
    - For final series (no vintage_date), available_date == release_date.
    - For a given ref_period, release_date is the same across all vintages.
    """
    if "ref_period" not in df.columns:
        raise ValueError("DataFrame must contain 'ref_period' column.")

    df = df.copy()

    # Ensure ref_period is datetime
    df["ref_period"] = pd.to_datetime(df["ref_period"])

    # period_end
    df["period_end"] = df["ref_period"].apply(lambda d: period_end(d, meta.freq))

    # release_date
    df["release_date"] = df["ref_period"].apply(
        lambda d: compute_release_date_for_row(d, meta)
    )

    # available_date
    df["available_date"] = df["release_date"]

    return df


# -----------------------------------------------------------------------------
# 针对所有 series 的封装（可选）
# -----------------------------------------------------------------------------


def build_calendar_for_all_series(
    raw_data: Dict[str, pd.DataFrame],
    meta_dict: Dict[str, SeriesMeta] | None = None,
    config_path: str | Path = "config/series.yaml",
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function: given raw_data dict (from loaders.load_all_raw_series),
    add calendar columns for all series.

    Parameters
    ----------
    raw_data : dict[str, DataFrame]
        Mapping: series_name -> DataFrame as returned by load_raw_series().
    meta_dict : dict[str, SeriesMeta], optional
        Mapping: series_name -> SeriesMeta.
        If None, this will be loaded from config_path.
    config_path : str or Path
        Path to series.yaml (used only if meta_dict is None).

    Returns
    -------
    dict[str, DataFrame]
        Mapping: series_name -> DataFrame with calendar columns added.
    """
    if meta_dict is None:
        meta_dict = load_series_meta(config_path)

    out: Dict[str, pd.DataFrame] = {}

    for name, df in raw_data.items():
        if name not in meta_dict:
            raise KeyError(f"Series '{name}' not found in metadata.")

        meta = meta_dict[name]
        df_cal = add_calendar_columns_for_series(df, meta)
        out[name] = df_cal

    return out