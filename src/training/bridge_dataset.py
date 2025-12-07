# src/training/bridge_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd


@dataclass
class BridgeTarget:
    """
    GDP nowcast target for a given quarter.

    Attributes
    ----------
    ref_period : pd.Timestamp
        Quarter start date (e.g. 2014-04-01 for 2014Q2).
    release_date : pd.Timestamp
        First official release date of this quarter's GDP growth.
    value : float
        GDP growth (% SAAR) for this quarter, as target y.
    """
    ref_period: pd.Timestamp
    release_date: pd.Timestamp
    value: float


def build_gdp_target_table(gdp_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table of GDP nowcast targets from the calendar DataFrame.

    Input gdp_cal is expected to be the calendar-augmented DataFrame for
    the 'gdp_growth' series, with at least the following columns:

        - ref_period   : quarter start date (Timestamp)
        - release_date : theoretical economic release date (Timestamp)
        - value        : GDP growth (% SAAR) for that ref_period

    Because the underlying data may contain multiple rows per quarter
    (due to multiple vintages / revisions), we collapse to ONE target
    per ref_period:

        - sort by (ref_period, release_date)
        - take the earliest release_date (first official release)

    Returns
    -------
    pd.DataFrame
        index: ref_period (quarter start)
        columns:
            - target_release_date : first release date for this quarter
            - y                   : target GDP growth value
    """
    required_cols = {"ref_period", "release_date", "value"}
    missing = required_cols - set(gdp_cal.columns)
    if missing:
        raise ValueError(
            f"gdp_cal is missing required columns: {missing}. "
            f"Available columns: {gdp_cal.columns.tolist()}"
        )

    df = gdp_cal.copy()
    df["ref_period"] = pd.to_datetime(df["ref_period"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows without essential info
    df = df.dropna(subset=["ref_period", "release_date", "value"])

    # Earliest release per ref_period = "first release"
    df = df.sort_values(["ref_period", "release_date"])
    df_first = df.groupby("ref_period", as_index=False).head(1)

    out = pd.DataFrame(
        {
            "target_release_date": df_first["release_date"].values,
            "y": df_first["value"].values,
        },
        index=pd.to_datetime(df_first["ref_period"].values),
    )
    out.index.name = "ref_period"

    return out


def _as_quarter_start_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Map a DatetimeIndex of as_of_date to quarter start dates.

    Example
    -------
    2010-02-15 -> 2010-01-01  (2010Q1)
    2010-04-01 -> 2010-04-01  (2010Q2)
    """
    return dates.to_period("Q").start_time.normalize()


def build_bridge_dataset(
    panels: Dict[str, pd.DataFrame],
    gdp_cal: pd.DataFrame,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    predictor_series: Optional[Sequence[str]] = None,
    exclude_targets: Sequence[str] = ("gdp_growth",),
    min_non_missing_features: int = 1,
) -> pd.DataFrame:
    """
    Construct a training dataset for GDP bridge models from:

        - daily macro panels (value / ref_period / age_days / is_new)
        - GDP calendar (gdp_cal)

    Each row in the output corresponds to one origin_date (as_of_date)
    where we form a nowcast for the CURRENT quarter's GDP growth.

    High-level logic
    ----------------
    1) From gdp_cal, build a target table:
           ref_period (quarter start) -> (target_release_date, y)

    2) From panels["value"], iterate over as_of_date in [start_date, end_date]:
           - Determine target_ref_period = quarter_start(as_of_date)
           - Join the corresponding (target_release_date, y)
           - Keep the row IFF:
                * y is known (non-NaN)
                * origin_date < target_release_date
                  (we only keep true "nowcast" periods, before GDP is released)
                * feature vector has at least `min_non_missing_features`
                  non-NaN values

    3) For features X, we use:
           - the macro series from panels["value"], at that as_of_date
           - by default, all series EXCEPT 'gdp_growth' (target itself)

    Parameters
    ----------
    panels : dict[str, DataFrame]
        Output of build_macro_panels():
            panels["value"]      : as_of_date × series_name
            panels["ref_period"] : as_of_date × series_name
            panels["age_days"]   : as_of_date × series_name
            panels["is_new"]     : as_of_date × series_name

        Only panels["value"] is required for the baseline implementation.
        The others can be used later to enrich features (lags, ages, etc.).

    gdp_cal : DataFrame
        Calendar DataFrame for 'gdp_growth', as returned by
        calendar.add_calendar_columns_for_series().

    start_date, end_date : str or Timestamp, optional
        Restrict origin_date range. If None, use the full index range
        of panels["value"].

    predictor_series : sequence of str, optional
        Series names to be used as predictors X.
        If None, we use ALL columns of panels["value"] except those in
        `exclude_targets`.

    exclude_targets : sequence of str
        Series names to be excluded from X by default (e.g. ['gdp_growth']).

    min_non_missing_features : int
        Minimum number of non-NaN feature values required for a row to be
        retained. Rows with fewer non-missing X are dropped.

    Returns
    -------
    pd.DataFrame
        Columns:
            - origin_date         : as_of_date (prediction time)
            - target_ref_period   : quarter start date of the target GDP
            - target_release_date : first official release date of the target
            - y                   : GDP growth (% SAAR) for that quarter
            - x_<series_name>     : feature columns for each predictor series

        Each row corresponds to a valid "nowcast" snapshot.
    """
    if "value" not in panels:
        raise ValueError("panels must contain a 'value' entry (daily macro values).")

    value_panel = panels["value"].copy()
    value_panel.index = pd.to_datetime(value_panel.index)

    # 1) Determine as_of_date range
    as_of_index = value_panel.index
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        as_of_index = as_of_index[as_of_index >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        as_of_index = as_of_index[as_of_index <= end_ts]

    if len(as_of_index) == 0:
        raise ValueError("No as_of_date in the specified range.")

    # 2) Build GDP target table
    target_table = build_gdp_target_table(gdp_cal)  # index = ref_period

    # 3) Build base DataFrame indexed by origin_date (as_of_date)
    df = pd.DataFrame(index=as_of_index)
    df.index.name = "origin_date"

    # 3a) For each as_of_date, determine which quarter it aims to nowcast
    df["target_ref_period"] = _as_quarter_start_index(df.index)

    # 3b) Join target_release_date and y via target_ref_period
    df = df.join(
        target_table,
        on="target_ref_period",
        how="left",
    )

    # Drop rows where target is unknown
    df = df.dropna(subset=["y", "target_release_date"])

    # 3c) Enforce "true nowcast": origin_date < target_release_date
    df["origin_date"] = df.index
    df["target_release_date"] = pd.to_datetime(df["target_release_date"])
    mask_nowcast = df["origin_date"] < df["target_release_date"]
    df = df.loc[mask_nowcast]

    if df.empty:
        raise ValueError("No valid nowcast rows after applying release-date filter.")

    # 4) Determine predictor series
    all_series = list(value_panel.columns)

    if predictor_series is None:
        exclude_set = set(exclude_targets)
        predictors = [s for s in all_series if s not in exclude_set]
    else:
        predictors = list(predictor_series)

    if not predictors:
        raise ValueError("No predictor series selected for bridge dataset.")

    # 5) Attach feature columns x_<series_name>
    #    For each origin_date, take value_panel.loc[origin_date, series]
    value_panel_sub = value_panel[predictors].reindex(df.index)

    for name in predictors:
        df[f"x_{name}"] = pd.to_numeric(value_panel_sub[name], errors="coerce")

    # 6) Drop rows with too few non-missing features
    feature_cols = [f"x_{name}" for name in predictors]
    non_missing_count = df[feature_cols].notna().sum(axis=1)
    df = df[non_missing_count >= min_non_missing_features]

    if df.empty:
        raise ValueError(
            "No rows left after enforcing min_non_missing_features="
            f"{min_non_missing_features}."
        )

    # Reset index to have origin_date as a normal column
    df = df.reset_index(drop=True)

    return df