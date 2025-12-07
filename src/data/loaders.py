# src/data/loaders.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .series_config import SeriesMeta, load_series_meta


def load_raw_series(
    meta: SeriesMeta,
    data_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """
    Load a single macro series from local CSV under `data/raw/`.

    Behavior depends on `meta.vintage_source`:

      - If meta.vintage_source == "alfred":
            expects: data/raw/{code}_vintage.csv
            columns: ref_period, vintage_date, value

      - Else:
            expects: data/raw/{code}.csv
            columns: ref_period, value

    This function DOES NOT:
      - compute release_date
      - apply transformations
      - perform as-of filtering

    It only normalizes the raw CSV into a standard DataFrame.
    """
    data_dir = Path(data_dir)
    code = meta.code

    if meta.vintage_source == "alfred":
        path = data_dir / f"{code}_vintage.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Vintage CSV not found for {code}: {path}\n"
                "Please run sync_all_series() to download/update raw data."
            )

        df = pd.read_csv(path)

        # Basic column sanity check
        if "ref_period" not in df.columns:
            raise ValueError(f"{path} missing 'ref_period' column.")
        if "vintage_date" not in df.columns:
            raise ValueError(f"{path} missing 'vintage_date' column.")
        if "value" not in df.columns:
            raise ValueError(f"{path} missing 'value' column.")

        # Normalize types
        df["ref_period"] = pd.to_datetime(df["ref_period"])
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df[["ref_period", "vintage_date", "value"]].sort_values(
            ["ref_period", "vintage_date"]
        )
        df.reset_index(drop=True, inplace=True)
        return df

    else:
        path = data_dir / f"{code}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Final CSV not found for {code}: {path}\n"
                "Please run sync_all_series() to download/update raw data."
            )

        df = pd.read_csv(path)

        if "ref_period" not in df.columns:
            raise ValueError(f"{path} missing 'ref_period' column.")
        if "value" not in df.columns:
            raise ValueError(f"{path} missing 'value' column.")

        df["ref_period"] = pd.to_datetime(df["ref_period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df[["ref_period", "value"]].sort_values("ref_period")
        df.reset_index(drop=True, inplace=True)
        return df


def load_all_raw_series(
    data_dir: str | Path = "data/raw",
    config_path: str | Path = "config/series.yaml",
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function: load raw data for ALL series defined in series.yaml.

    Returns
    -------
    dict[str, DataFrame]
        key   : series name (e.g. "gdp_growth", "industrial_production")
        value : DataFrame as returned by `load_raw_series` for that series.
    """
    data_dir = Path(data_dir)
    meta_dict = load_series_meta(config_path)

    out: Dict[str, pd.DataFrame] = {}
    for name, meta in meta_dict.items():
        df = load_raw_series(meta, data_dir=data_dir)
        out[name] = df

    return out