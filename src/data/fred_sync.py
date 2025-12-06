# src/data/fred_sync.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .series_config import SeriesMeta, load_series_meta
from .fred_client import FredClient, FredAPIError


def sync_single_series(
    meta: SeriesMeta,
    client: FredClient,
    data_dir: str | Path = "data/raw",
    observation_start: str = "1970-01-01",
    observation_end: Optional[str] = None,
) -> Path:
    """
    Download a single macro series using the FRED/ALFRED API and
    persist it as a CSV file under `data/raw/`.

    Behavior depends on `meta.vintage_source`:

      - If meta.vintage_source == "alfred":
          * Use FredClient.fetch_vintage_series_full()
          * Save to: data/raw/{meta.code}_vintage.csv
          * Columns: ref_period, vintage_date, value

      - Else (treated as "final"):
          * Use FredClient.fetch_final_series()
          * Save to: data/raw/{meta.code}.csv
          * Columns: ref_period, value

    Parameters
    ----------
    meta : SeriesMeta
        Metadata for the series (from series.yaml).
    client : FredClient
        Initialized FRED/ALFRED client.
    data_dir : str or Path
        Directory where CSV files will be saved.
    observation_start : str
        Start date for the reference period range (e.g. "1970-01-01").
    observation_end : str, optional
        End date for the reference period range. If None, FRED default is used.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    code = meta.code

    if meta.vintage_source == "alfred":
        # Full vintage history
        df = client.fetch_vintage_series_full(
            series_id=code,
            observation_start=observation_start,
            observation_end=observation_end,
        )

        # Normalize & sanity check
        expected_cols = {"ref_period", "vintage_date", "value"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Vintage data for {code} missing columns: {missing}. "
                f"Got columns: {df.columns.tolist()}"
            )

        path = data_dir / f"{code}_vintage.csv"
        df = df[["ref_period", "vintage_date", "value"]].copy()
        df.to_csv(path, index=False)

    else:
        # Final (non-vintage) series
        df = client.fetch_final_series(
            series_id=code,
            observation_start=observation_start,
            observation_end=observation_end,
        )

        # Normalize & sanity check
        expected_cols = {"ref_period", "value"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Final data for {code} missing columns: {missing}. "
                f"Got columns: {df.columns.tolist()}"
            )

        path = data_dir / f"{code}.csv"
        df = df[["ref_period", "value"]].copy()
        df.to_csv(path, index=False)

    return path


def sync_all_series(
    config_path: str | Path = "config/series.yaml",
    data_dir: str | Path = "data/raw",
    observation_start: str = "1970-01-01",
    observation_end: Optional[str] = None,
    series_filter: Optional[list[str]] = None,
    client: Optional[FredClient] = None,
) -> Dict[str, Path]:
    """
    Synchronize ALL series defined in series.yaml by downloading from
    FRED/ALFRED and saving standardized CSV files into `data/raw/`.

    Parameters
    ----------
    config_path : str or Path
        Path to series.yaml.
    data_dir : str or Path
        Directory where CSV files will be saved.
    observation_start : str
        Start date for reference period.
    observation_end : str, optional
        End date for reference period.
    series_filter : list[str], optional
        If provided, only series whose names are in this list will be synced.
        (Use series names, e.g. "gdp_growth", "industrial_production", ...)
    client : FredClient, optional
        Existing client instance. If None, a new one is created via from_env().

    Returns
    -------
    dict[str, Path]
        Mapping from series_name → Path to the written CSV file.
    """
    meta_dict = load_series_meta(config_path)
    data_dir = Path(data_dir)

    if client is None:
        client = FredClient.from_env()

    results: Dict[str, Path] = {}

    for name, meta in meta_dict.items():
        if series_filter is not None and name not in series_filter:
            continue

        try:
            print(f"[SYNC] {name} ({meta.code}), vintage_source={meta.vintage_source}")
            path = sync_single_series(
                meta=meta,
                client=client,
                data_dir=data_dir,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            print(f"[OK]   {name} -> {path}")
            results[name] = path
        except (FredAPIError, ValueError) as e:
            # For now, we raise immediately so you notice issues early.
            # If you prefer "best effort", you could log & continue.
            print(f"[ERROR] {name} ({meta.code}) failed: {e}")
            raise

    return results
