# test/test_data_loader.py
"""
使用真实本地数据的检查脚本：

1. 从 config/series.yaml 读取所有 series 元数据
2. 对每个 series：
   - 从 data/raw 加载对应 CSV
   - 打印：行数、列名、时间范围、value 的统计、前几行切片
3. 如果某个 CSV 不存在，明确提示需要先同步下载
"""

from pathlib import Path
import sys

import pandas as pd

# 1. 确定项目根目录和 src 路径
ROOT = Path(__file__).resolve().parents[1]  # GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 2. 导入项目内部模块（基于 src/）
from data.series_config import load_series_meta  # noqa: E402
from data.loaders import load_raw_series         # noqa: E402


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

    meta_dict = load_series_meta(config_path)
    if not meta_dict:
        print("[ERROR] No series defined in config/series.yaml")
        return

    print(f"[INFO] Loaded {len(meta_dict)} series from {config_path}")
    print(f"[INFO] Using raw data directory: {data_dir}")
    print("------------------------------------------------------------")

    for name, meta in meta_dict.items():
        code = meta.code
        print(f"\n========== {name} ({code}) ==========")

        try:
            df = load_raw_series(meta, data_dir=data_dir)
        except FileNotFoundError as e:
            print(f"[MISSING] {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Failed to load {name} ({code}): {e}")
            continue

        # 基本信息
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # 时间范围
        if "ref_period" in df.columns:
            ref_min = df["ref_period"].min()
            ref_max = df["ref_period"].max()
            print(f"ref_period range: {ref_min}  →  {ref_max}")

        # value 的统计
        if "value" in df.columns:
            desc = df["value"].describe()
            print("\n[value] summary stats:")
            print(desc)

        # 前几行切片
        print("\nHead (first 5 rows):")
        print(df.head(5))


if __name__ == "__main__":
    main()