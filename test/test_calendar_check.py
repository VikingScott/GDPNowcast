# test/test_calendar_check.py
"""
Real-data calendar sanity check:

1. Load all raw series from data/raw using loaders.py.
2. For each series, add calendar columns:
     - period_end
     - release_date
     - available_date
3. For each series, print:
     - basic info
     - head of [ref_period, period_end, release_date, vintage_date?, available_date]
4. Check:
     - available_date >= release_date
     - if vintage_date exists: available_date >= vintage_date
   If there are violations, print summary and sample offending rows.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# 1. Set up sys.path so we can import from src/
ROOT = Path(__file__).resolve().parents[1]  # project root: GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 2. Import project modules
from data.series_config import load_series_meta  # noqa: E402
from data.loaders import load_all_raw_series     # noqa: E402
from data.calendar import add_calendar_columns_for_series  # noqa: E402


def main() -> None:
    config_path = ROOT / "config" / "series.yaml"
    data_dir = ROOT / "data" / "raw"

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("        Please run sync_all_series() first to download raw data.")
        return

    # Load metadata & raw data
    meta_dict = load_series_meta(config_path)
    if not meta_dict:
        print("[ERROR] No series defined in config/series.yaml")
        return

    raw_all = load_all_raw_series(data_dir=data_dir, config_path=config_path)

    print(f"[INFO] Loaded {len(meta_dict)} series from {config_path}")
    print(f"[INFO] Using raw data directory: {data_dir}")
    print("------------------------------------------------------------")

    any_violation = False

    for name, meta in meta_dict.items():
        if name not in raw_all:
            print(f"[ERROR] Raw data for series '{name}' not found in raw_all.")
            continue

        df_raw = raw_all[name]

        print(f"\n========== {name} ({meta.code}) ==========")
        print(f"Raw rows: {len(df_raw)}, columns: {list(df_raw.columns)}")

        # Add calendar columns
        try:
            df_cal = add_calendar_columns_for_series(df_raw, meta)
        except Exception as e:
            print(f"[ERROR] Failed to add calendar columns for {name}: {e}")
            continue

        # Print head of key columns
        cols_to_show = ["ref_period", "period_end", "release_date", "available_date"]
        if "vintage_date" in df_cal.columns:
            cols_to_show.insert(3, "vintage_date")  # ref, period_end, release, vintage, available

        print("\nHead (first 8 rows) of calendar columns:")
        print(df_cal[cols_to_show].head(8))

        # Basic range info
        ref_min = df_cal["ref_period"].min()
        ref_max = df_cal["ref_period"].max()
        avail_min = df_cal["available_date"].min()
        avail_max = df_cal["available_date"].max()
        print(f"\nref_period range    : {ref_min}  →  {ref_max}")
        print(f"available_date range: {avail_min}  →  {avail_max}")

        # Sanity checks
        # 1) available_date >= release_date
        mask_rel = df_cal["available_date"] < df_cal["release_date"]
        n_rel = int(mask_rel.sum())

        # 2) if vintage_date exists: available_date >= vintage_date
        n_vint = 0
        if "vintage_date" in df_cal.columns:
            mask_vint = df_cal["available_date"] < df_cal["vintage_date"]
            n_vint = int(mask_vint.sum())
        else:
            mask_vint = pd.Series(False, index=df_cal.index)

        if n_rel == 0 and n_vint == 0:
            print("[CHECK] OK: no violations in available_date ordering.")
        else:
            any_violation = True
            print("[CHECK] WARNING: found violations in available_date constraints.")
            if n_rel > 0:
                print(f"  - available_date < release_date: {n_rel} rows")
                df_bad_rel = df_cal.loc[mask_rel, cols_to_show].copy()
                # Print a few samples & date span
                print("    sample rows (up to 5):")
                print(df_bad_rel.head(5))
                print("    offending ref_period range:",
                      df_bad_rel["ref_period"].min(), "→", df_bad_rel["ref_period"].max())

            if n_vint > 0:
                print(f"  - available_date < vintage_date: {n_vint} rows")
                df_bad_vint = df_cal.loc[mask_vint, cols_to_show].copy()
                print("    sample rows (up to 5):")
                print(df_bad_vint.head(5))
                print("    offending ref_period range:",
                      df_bad_vint["ref_period"].min(), "→", df_bad_vint["ref_period"].max())

    print("\n------------------------------------------------------------")
    if any_violation:
        print("[SUMMARY] Some series have calendar inconsistencies (see warnings above).")
    else:
        print("[SUMMARY] All series passed calendar checks (available_date ordering OK).")


if __name__ == "__main__":
    main()