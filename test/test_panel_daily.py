# test/test_build_panels.py
# Generate daily macro panel files into data/processed/panels

import sys
from pathlib import Path
import pandas as pd

# Add project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import load_all_raw_series, build_macro_panels  # noqa: E402
from data.series_config import load_series_meta  # noqa: E402


def main():
    print("=== Loading raw series ===")
    raw = load_all_raw_series(
        data_dir=ROOT / "data" / "raw",
        config_path=ROOT / "config" / "series.yaml"
    )

    print("=== Building calendars ===")
    meta_dict = load_series_meta(ROOT / "config" / "series.yaml")

    from data.calendar import build_calendar_for_series  # noqa: E402

    calendars = {}
    for name, meta in meta_dict.items():
        cal = build_calendar_for_series(meta, raw[name])
        calendars[name] = cal

    print("=== Building daily macro panels ===")
    panels = build_macro_panels(raw, calendars)

    outdir = ROOT / "data" / "processed" / "panels"
    outdir.mkdir(parents=True, exist_ok=True)

    for key, df in panels.items():
        path = outdir / f"{key}.parquet"
        df.to_parquet(path)
        print(f"Saved panel: {path}  shape={df.shape}")

    print("\n=== All panels generated successfully ===")


if __name__ == "__main__":
    main()