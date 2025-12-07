# src/training/bridge_dataset.py

from __future__ import annotations

# --- allow running this file directly ---
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # .../GDPNowcast
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# --- imports that work in both modes ---
try:
    # when run as module: python -m src.training.bridge_dataset
    from ..data.series_config import SeriesMeta, load_series_meta
    from ..data.panel import build_macro_panels_from_raw
except ImportError:
    # when run as script: python src/training/bridge_dataset.py
    from src.data.series_config import SeriesMeta, load_series_meta
    from src.data.panel import build_macro_panels_from_raw


# -----------------------------------------------------------------------------
# Core idea of Step 2
# -----------------------------------------------------------------------------
# We already have daily panels (value/ref_period/age_days/is_new).
# Panels are raw "latest-available headline values" per day.
#
# Step 2 makes X consistent with the transform rules in series.yaml:
#   - level
#   - mom
#   - diff
#   - log_diff
#
# Important:
#   - We DO NOT standardize here.
#   - Rolling standardization belongs to the training layer (Step 3).
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Quarter utilities
# -----------------------------------------------------------------------------

def quarter_start(dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt)
    q = ((dt.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=dt.year, month=q, day=1)


def quarter_end(q_start: pd.Timestamp) -> pd.Timestamp:
    q_start = pd.Timestamp(q_start)
    # next quarter start - 1 day
    if q_start.month == 10:
        next_q = pd.Timestamp(year=q_start.year + 1, month=1, day=1)
    else:
        next_q = pd.Timestamp(year=q_start.year, month=q_start.month + 3, day=1)
    return next_q - pd.Timedelta(days=1)


def _shift_quarter_start(q_start: pd.Timestamp, n_quarters: int) -> pd.Timestamp:
    q_start = pd.Timestamp(q_start)
    return q_start + pd.DateOffset(months=3 * n_quarters)


def compute_model_target_ref_period(origin_date: pd.Timestamp, lag_days: int) -> pd.Timestamp:
    """
    Option A: Sticky until previous quarter's advance release.

    Let:
      q0     = quarter_start(origin_date)
      q_prev = q0 - 1 quarter
      prev_release = quarter_end(q_prev) + lag_days

    Rule:
      if origin_date <= prev_release:
         model target is q_prev
      else:
         model target is q0
    """
    origin_date = pd.Timestamp(origin_date)

    q0 = quarter_start(origin_date)
    q_prev = _shift_quarter_start(q0, -1)

    prev_release = quarter_end(q_prev) + pd.Timedelta(days=int(lag_days))

    return q_prev if origin_date <= prev_release else q0


# -----------------------------------------------------------------------------
# Transform utilities
# -----------------------------------------------------------------------------

ALLOWED_TRANSFORMS = {"", "level", "mom", "diff", "log_diff"}


def _compute_ref_level_series_from_daily(
    value_daily: pd.Series,
    ref_daily: pd.Series,
) -> pd.Series:
    """
    Build a ref_period-level 'headline' series from daily panels.

    value_daily: daily latest value
    ref_daily  : daily latest ref_period corresponding to the value

    We compress daily repeated values into a ref_period-indexed series by
    taking the last observed value for each ref_period in the daily timeline.
    """
    df = pd.DataFrame({"value": value_daily, "ref_period": ref_daily})
    df = df.dropna(subset=["value", "ref_period"])
    df["ref_period"] = pd.to_datetime(df["ref_period"])
    ref_level = df.groupby("ref_period")["value"].last().sort_index()
    return ref_level


def _transform_ref_level(
    ref_level: pd.Series,
    transform: str,
) -> pd.Series:
    """
    Apply transform on ref_period-level series.
    Return a ref_period-indexed transformed series.
    """
    if transform not in ALLOWED_TRANSFORMS:
        raise ValueError(f"Unknown transform='{transform}'. Allowed={ALLOWED_TRANSFORMS}")

    t = transform or "level"

    if t == "level":
        out = ref_level.copy()
    elif t == "diff":
        out = ref_level.diff()
    elif t == "mom":
        # percent change, not *100 to keep scale moderate
        out = ref_level.pct_change()
    elif t == "log_diff":
        safe = ref_level.where(ref_level > 0)
        out = np.log(safe).diff()
    else:
        out = ref_level.copy()

    return out


def _map_ref_transform_back_to_daily(
    ref_daily: pd.Series,
    ref_trans: pd.Series,
) -> pd.Series:
    """
    Given:
      - daily ref_period (latest headline ref_period)
      - ref_period-indexed transformed series

    Map transform value back to each day using the day's latest ref_period.
    """
    ref_daily = pd.to_datetime(ref_daily)
    return ref_daily.map(ref_trans)


def build_transformed_feature_daily(
    panels: Dict[str, pd.DataFrame],
    series_name: str,
    meta: SeriesMeta,
) -> pd.Series:
    """
    Build a daily feature series for one macro series based on transform rule.

    panels must contain:
      - panels["value"]      : DataFrame index=as_of_date, columns=series_name
      - panels["ref_period"] : DataFrame index=as_of_date, columns=series_name
    """
    if "value" not in panels or "ref_period" not in panels:
        raise ValueError("panels must include 'value' and 'ref_period'.")

    value_panel = panels["value"]
    ref_panel = panels["ref_period"]

    if series_name not in value_panel.columns or series_name not in ref_panel.columns:
        raise KeyError(f"Series '{series_name}' not found in panels.")

    value_daily = value_panel[series_name].astype("float64")
    ref_daily = ref_panel[series_name]

    ref_level = _compute_ref_level_series_from_daily(value_daily, ref_daily)
    ref_trans = _transform_ref_level(ref_level, meta.transform)
    feat_daily = _map_ref_transform_back_to_daily(ref_daily, ref_trans)

    return feat_daily


def _infer_panel_date_range(panels: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    idx = panels["value"].index
    return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())


# -----------------------------------------------------------------------------
# Step-2 dataset builder
# -----------------------------------------------------------------------------

def build_bridge_dataset(
    *,
    panels: Optional[Dict[str, pd.DataFrame]] = None,
    config_path: str | Path = "config/series.yaml",
    raw_data_dir: str | Path = "data/raw",
    start_date: str | pd.Timestamp = "1980-01-01",
    end_date: str | pd.Timestamp | None = None,
    target_name: str = "gdp_growth",
    feature_names: Optional[Iterable[str]] = None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Step-2 bridge dataset builder.

    Output columns:
      - target_ref_period           (visible fact as-of origin_date)
      - target_release_date         (proxy derived from target_ref_period)
      - model_target_ref_period     (H=0 task quarter based on origin_date, sticky mapping)
      - model_target_release_date   (proxy derived from model_target_ref_period)
      - y
      - origin_date
      - x_{feature}

    Notes:
      - X are transformed per series.yaml.
      - No rolling standardization here.
      - Uses daily panels; if panels not provided, builds them from raw CSV.
      - "y" is taken from the target's daily value panel (latest-available headline value).
        This is consistent with your current calendar + vintage rules.
    """
    config_path = Path(config_path)
    raw_data_dir = Path(raw_data_dir)

    meta_dict = load_series_meta(config_path)

    if target_name not in meta_dict:
        raise KeyError(f"target_name='{target_name}' not found in {config_path}")

    if feature_names is None:
        feature_names = [k for k in meta_dict.keys() if k != target_name]
    else:
        feature_names = list(feature_names)

    # Build panels if not provided
    if panels is None:
        panels = build_macro_panels_from_raw(
            config_path=config_path,
            raw_data_dir=raw_data_dir,
            start_date=start_date,
            end_date=end_date,
        )

    # Resolve end_date if still None
    if end_date is None:
        _, inferred_end = _infer_panel_date_range(panels)
        end_date = inferred_end

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Daily index of origin dates
    origin_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # -----------------------------
    # Target y daily series (visible latest)
    # -----------------------------
    y_daily = panels["value"][target_name].reindex(origin_index)

    # Visible fact ref_period as-of each origin_date
    target_ref_daily = panels["ref_period"][target_name].reindex(origin_index)

    # YAML meta for GDP growth
    target_meta = meta_dict[target_name]
    gdp_lag = int(target_meta.release_lag_days or 0)

    # -----------------------------
    # Visible-fact release date proxy
    # -----------------------------
    target_release_daily = target_ref_daily.map(
        lambda rp: (quarter_end(rp) + pd.Timedelta(days=gdp_lag)) if pd.notna(rp) else pd.NaT
    )

    # -----------------------------
    # Model-task (H=0) target quarter with sticky mapping
    # -----------------------------
    model_target_ref_daily = pd.Series(
        [compute_model_target_ref_period(d, gdp_lag) for d in origin_index],
        index=origin_index,
        dtype="datetime64[ns]",
    )

    model_target_release_daily = model_target_ref_daily.map(
        lambda rp: (quarter_end(rp) + pd.Timedelta(days=gdp_lag)) if pd.notna(rp) else pd.NaT
    )

    # -----------------------------
    # Build transformed X daily
    # -----------------------------
    X_cols: Dict[str, pd.Series] = {}
    for fname in feature_names:
        meta = meta_dict[fname]
        feat_daily = build_transformed_feature_daily(
            panels=panels,
            series_name=fname,
            meta=meta,
        ).reindex(origin_index)

        X_cols[f"x_{fname}"] = feat_daily

    # -----------------------------
    # Assemble dataset
    # -----------------------------
    ds = pd.DataFrame(
        {
            "target_ref_period": pd.to_datetime(target_ref_daily),
            "target_release_date": pd.to_datetime(target_release_daily),
            "model_target_ref_period": pd.to_datetime(model_target_ref_daily),
            "model_target_release_date": pd.to_datetime(model_target_release_daily),
            "y": pd.to_numeric(y_daily, errors="coerce"),
            "origin_date": origin_index,
            **X_cols,
        }
    )

    col_order = (
        [
            "target_ref_period",
            "target_release_date",
            "model_target_ref_period",
            "model_target_release_date",
            "y",
            "origin_date",
        ]
        + list(X_cols.keys())
    )
    ds = ds[col_order]

    # Optional save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(save_path, index=False)

    return ds


# -----------------------------------------------------------------------------
# CLI-style quick runner (not pytest)
# -----------------------------------------------------------------------------

def main():
    """
    Example usage:
      python -m src.training.bridge_dataset
      python src/training/bridge_dataset.py

    This builds a Step-2 dataset from:
      - local raw CSV under data/raw
      - series metadata & transform rules in config/series.yaml

    Output:
      - data/processed/bridge_dataset/bridge_dataset_step2.parquet
    """
    # Robust project ROOT
    ROOT = Path(__file__).resolve().parents[2]

    config_path = ROOT / "config" / "series.yaml"
    raw_dir = ROOT / "data" / "raw"
    out_path = ROOT / "data" / "processed" / "bridge_dataset" / "bridge_dataset_step2.parquet"

    ds = build_bridge_dataset(
        panels=None,
        config_path=config_path,
        raw_data_dir=raw_dir,
        start_date="1980-01-01",
        end_date=None,
        target_name="gdp_growth",
        feature_names=None,
        save_path=out_path,
    )

    x_cols = [c for c in ds.columns if c.startswith("x_")]

    print("========== Bridge Dataset (Step 2) ==========")
    print(f"Saved to: {out_path}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {ds.columns.tolist()}")
    print(f"origin_date range: {ds['origin_date'].min()} -> {ds['origin_date'].max()}")
    print(f"n_features: {len(x_cols)}")
    print("feature columns:", x_cols)

    print("\nTail (last 8 rows):")
    print(ds.tail(8).to_string(index=False))

    print("\nNon-missing ratio (top-level):")
    for c in x_cols:
        ratio = ds[c].notna().mean()
        print(f"  {c}: {ratio:.3f}")

    # Quick sanity check for the new dual target semantics
    print("\n[CHECK] Target semantics sample (last 8 rows):")
    print(
        ds[
            [
                "origin_date",
                "target_ref_period",
                "target_release_date",
                "model_target_ref_period",
                "model_target_release_date",
                "y",
            ]
        ]
        .tail(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()