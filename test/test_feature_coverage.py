# test/test_feature_coverage.py
"""
Quick diagnostic: coverage ratio per decade for each x_<series> in bridge dataset.

Pipeline:
  series.yaml + data/raw/*.csv
    → load_all_raw_series
    → build_calendar_for_all_series
    → build_macro_panels
    → build_bridge_dataset
    → coverage stats by decade
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# 1. 把 src/ 加入 sys.path，按你现在的项目结构
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 2. 导入项目内部模块（注意来源）
from data.series_config import load_series_meta  # noqa: E402
from data.loaders import load_all_raw_series     # noqa: E402
from data.calendar import build_calendar_for_all_series  # noqa: E402
from data.panel import build_macro_panels        # noqa: E402
from training.bridge_dataset import build_bridge_dataset  # noqa: E402


def decade_label(dt: pd.Timestamp) -> str:
    """Convert datetime → decade label, e.g. 1980s, 1990s."""
    y = dt.year
    decade_start = (y // 10) * 10
    return f"{decade_start}s"


def main() -> None:
    config_path = ROOT / "config" / "series.yaml"
    raw_dir = ROOT / "data" / "raw"

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return
    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("        请先运行 fred_sync.sync_all_series() 下载 FRED/ALFRED 数据。")
        return

    print("[STEP] 1. Load metadata & raw series")
    meta_dict = load_series_meta(config_path)
    raw_all = load_all_raw_series(data_dir=raw_dir, config_path=config_path)

    print("[STEP] 2. Build calendars for all series")
    cal_all = build_calendar_for_all_series(
        raw_data=raw_all,
        meta_dict=meta_dict,
        config_path=config_path,
    )

    if "gdp_growth" not in cal_all:
        print("[ERROR] 'gdp_growth' not found in calendar data.")
        return
    gdp_cal = cal_all["gdp_growth"]

    # 确定 as_of_date 范围（和 test_bridge_dataset 一样的逻辑）
    min_avail = None
    max_avail = None
    for name, df_cal in cal_all.items():
        if "available_date" not in df_cal.columns:
            continue
        ad = pd.to_datetime(df_cal["available_date"]).dropna()
        if ad.empty:
            continue
        m1, m2 = ad.min(), ad.max()
        min_avail = m1 if min_avail is None else min(min_avail, m1)
        max_avail = m2 if max_avail is None else max(max_avail, m2)

    if min_avail is None or max_avail is None:
        print("[ERROR] 没有任何 available_date，检查 calendar 层。")
        return

    start_date = (min_avail - pd.Timedelta(days=30)).normalize()
    end_date = (max_avail + pd.Timedelta(days=30)).normalize()
    print(f"[STEP] 3. Build macro panels from {start_date.date()} to {end_date.date()}")

    panels = build_macro_panels(
        cal_data=cal_all,
        start_date=start_date,
        end_date=end_date,
    )

    print("[STEP] 4. Build bridge dataset (H=0 nowcast)")
    ds = build_bridge_dataset(
        panels=panels,
        gdp_cal=gdp_cal,
        start_date=start_date,
        end_date=end_date,
        predictor_series=None,           # 默认：全部除 gdp_growth
        exclude_targets=("gdp_growth",),
        min_non_missing_features=1,
    )

    print(f"[INFO] bridge dataset rows: {len(ds)}")
    print(f"[INFO] origin_date range: {ds['origin_date'].min().date()} → {ds['origin_date'].max().date()}")

    # -----------------------------------------------------------------
    # Coverage ratio per feature per decade
    # -----------------------------------------------------------------
    feature_cols = [c for c in ds.columns if c.startswith("x_")]
    print(f"\n[INFO] detected feature columns ({len(feature_cols)}):")
    print(feature_cols)

    ds["decade"] = ds["origin_date"].apply(decade_label)
    decades = ["1980s", "1990s", "2000s", "2010s", "2020s"]

    rows = []
    for feat in feature_cols:
        for dec in decades:
            sub = ds[ds["decade"] == dec]
            if sub.empty:
                ratio = np.nan
                non_missing = 0
                total = 0
            else:
                non_missing = sub[feat].notna().sum()
                total = len(sub)
                ratio = non_missing / total if total > 0 else np.nan
            rows.append(
                {
                    "feature": feat,
                    "decade": dec,
                    "coverage_ratio": ratio,
                    "non_missing_count": non_missing,
                    "total_count": total,
                }
            )

    cov_df = pd.DataFrame(rows)

    print("\n========== Coverage Ratio Per Feature Per Decade ==========")
    for feat in feature_cols:
        print(f"\n--- {feat} ---")
        sub = cov_df[cov_df["feature"] == feat].copy()
        # 格式化一下输出
        sub["coverage_ratio"] = sub["coverage_ratio"].round(3)
        print(sub[["decade", "coverage_ratio", "non_missing_count", "total_count"]].to_string(index=False))

    out_path = ROOT / "test" / "feature_coverage_summary.csv"
    cov_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Detailed coverage table saved to: {out_path}")


if __name__ == "__main__":
    main()