# run_daily.py

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# 自动修复路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from nowcast.export.macro_features import build_macro_features
except ImportError as e:
    print("❌ Error: Could not import nowcast package.")
    raise e

# 配置
OUTPUT_FILE = current_dir / "data" / "output" / "macro_signals_latest.csv"

def main():
    print("==================================================")
    print(f"   Macro Nowcast Daily Update - {datetime.now()}")
    print("==================================================")

    # 确保目录存在
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 运行核心计算 (自动历史)
    try:
        # 使用 auto，它会自动找到数据源的最早日期开始跑
        df = build_macro_features(start_date="1990-01-01", end_date=None)
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        return

    # 保存
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE)

    print("✅ All Done!")
    print(df.tail())

if __name__ == "__main__":
    main()