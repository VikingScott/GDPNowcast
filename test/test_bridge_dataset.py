# test/test_bridge_dataset.py
"""
Bridge dataset sanity check on real data.

功能：
  1. 从 data/raw + config/series.yaml 构造：
       - calendar 层 cal_all
       - panel 层 panels
       - bridge_dataset (build_bridge_dataset)
  2. 输出：
       - bridge_dataset 基本信息
       - 每个季度的 origin_date 样本数量
       - 选两个季度，分 early/mid/late 看 feature 非缺失数量分布
       - 指定月份（1990-01, 2000-03, 2010-05, 2020-08, 2024-10）
         各取前 10 天样本做切片展示
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 1. 把 src/ 加入 sys.path
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录：GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 2. 导入项目模块
from data.series_config import load_series_meta  # noqa: E402
from data.loaders import load_all_raw_series     # noqa: E402
from data.calendar import build_calendar_for_all_series  # noqa: E402
from data.panel import build_macro_panels        # noqa: E402
from training.bridge_dataset import build_bridge_dataset  # noqa: E402


def main() -> None:
    config_path = ROOT / "config" / "series.yaml"
    raw_dir = ROOT / "data" / "raw"

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return
    if not raw_dir.exists():
        print(f"[ERROR] Data directory not found: {raw_dir}")
        print("        请先运行 fred_sync.sync_all_series() 下载数据。")
        return

    # 1) 读 meta + raw
    meta_dict = load_series_meta(config_path)
    raw_all = load_all_raw_series(data_dir=raw_dir, config_path=config_path)

    # 2) calendar 层
    cal_all = build_calendar_for_all_series(
        raw_all,
        meta_dict=meta_dict,
        config_path=config_path,
    )

    if "gdp_growth" not in cal_all:
        print("[ERROR] 'gdp_growth' not found in calendar data.")
        return

    gdp_cal = cal_all["gdp_growth"]

    # 3) panel 层：先确定 as_of_date 范围
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

    # 给 panel 一点 buffer
    start_date = (min_avail - pd.Timedelta(days=30)).normalize()
    end_date = (max_avail + pd.Timedelta(days=30)).normalize()

    print(f"[INFO] panel 使用 as_of_date 范围: {start_date.date()} → {end_date.date()}")

    panels = build_macro_panels(
        cal_data=cal_all,
        start_date=start_date,
        end_date=end_date,
    )

    # 4) 构造 bridge dataset（H=0 nowcast）
    print("[INFO] 构造 bridge dataset（H=0 nowcast）...")
    ds = build_bridge_dataset(
        panels=panels,
        gdp_cal=gdp_cal,
        start_date=start_date,
        end_date=end_date,
        predictor_series=None,           # 默认：使用除 gdp_growth 以外的所有 series
        exclude_targets=("gdp_growth",),
        min_non_missing_features=1,
    )

    # 5) 基本信息
    print("\n========== Bridge Dataset: 基本信息 ==========")
    print(f"Rows (样本数): {len(ds)}")
    print(f"Columns: {ds.columns.tolist()}")
    print(
        "origin_date 范围:",
        ds["origin_date"].min().date(),
        "→",
        ds["origin_date"].max().date(),
    )

    feature_cols = [c for c in ds.columns if c.startswith("x_")]
    print(f"特征数量: {len(feature_cols)}")
    print("特征列示例:", feature_cols[:5])

    # 6) 每个季度的样本数量
    print("\n========== 每个季度的 origin_date 样本数量 ==========")
    counts = ds.groupby("target_ref_period").size().sort_index()
    print("季度数量:", len(counts))
    print("\n前 5 个季度样本数:")
    print(counts.head())
    print("\n中间 5 个季度样本数:")
    mid_start = max(0, len(counts) // 2 - 2)
    print(counts.iloc[mid_start:mid_start + 5])
    print("\n最后 5 个季度样本数:")
    print(counts.tail())

    # 7) 选两个季度，看 early / mid / late 的 feature 非缺失数量分布
    print("\n========== 部分季度的 early/mid/late feature 密度 ==========")
    if len(counts) >= 2:
        selected_quarters = [
            counts.index[0],
            counts.index[-1],
        ]
    else:
        selected_quarters = counts.index.tolist()

    for q in selected_quarters:
        q_df = ds[ds["target_ref_period"] == q].copy()
        if q_df.empty:
            continue

        q_df = q_df.sort_values("origin_date").reset_index(drop=True)
        n = len(q_df)
        q_df["non_missing_features"] = q_df[feature_cols].notna().sum(axis=1)

        print(f"\n--- Quarter {q.date()} ---")
        print(f"样本数: {n}")

        # 切成 early / mid / late 三段
        thirds = np.linspace(0, n, 4, dtype=int)  # 0, t1, t2, n
        segments = {
            "early": (thirds[0], thirds[1]),
            "mid": (thirds[1], thirds[2]),
            "late": (thirds[2], thirds[3]),
        }

        for label, (i_start, i_end) in segments.items():
            if i_end <= i_start:
                print(f"  [{label}] 样本过少，跳过")
                continue
            seg = q_df.iloc[i_start:i_end]
            print(
                f"  [{label}] 行数={len(seg)}, "
                f"non_missing_features 平均={seg['non_missing_features'].mean():.2f}, "
                f"中位数={seg['non_missing_features'].median():.2f}"
            )

    # 8) 指定月份切片，每个展示 10 天
    print("\n========== 指定月份切片展示（每个 10 天） ==========")
    month_specs = [
        ("1990-01", "1990-01-01"),
        ("2000-03", "2000-03-01"),
        ("2010-05", "2010-05-01"),
        ("2020-08", "2020-08-01"),
        ("2024-10", "2024-10-01"),
    ]

    for label, start_str in month_specs:
        start = pd.to_datetime(start_str)
        end = start + pd.Timedelta(days=9)  # 共 10 天

        mask = (ds["origin_date"] >= start) & (ds["origin_date"] <= end)
        sub = ds.loc[mask].copy()

        print(f"\n--- Slice: {label} ({start.date()} → {end.date()}) ---")
        if sub.empty:
            print("  [WARN] 该时间段内没有样本（可能超出数据范围或 GDP 目标尚未定义）。")
            continue

        # 只展示部分列，避免太宽
        display_cols = (
            ["origin_date", "target_ref_period", "target_release_date", "y"]
            + feature_cols[:5]
        )
        print(sub[display_cols].head(10))

    print("\n[SUMMARY] Bridge dataset 检查完成。")


if __name__ == "__main__":
    main()