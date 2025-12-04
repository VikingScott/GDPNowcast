# run_daily.py

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- 1. 自动修复路径问题 ---
# 无论你在哪里运行这个脚本，它都会自动找到 nowcast 包的位置
# 获取当前脚本所在的目录
current_dir = Path(__file__).resolve().parent
# 将根目录加入 Python 搜索路径
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 只有设置好路径后，才能导入 nowcast
try:
    from nowcast.export.macro_features import build_macro_features
except ImportError as e:
    print("❌ 错误: 找不到 nowcast 包。")
    print(f"请确保 'run_daily.py' 放在项目根目录下 (GDPNowcast/)。")
    print(f"当前路径: {current_dir}")
    raise e

# --- 2. 配置部分 ---
# 输出文件夹
OUTPUT_DIR = current_dir / "data" / "output"
# 输出文件名 (固定文件名方便下游读取，也可以加上日期时间戳)
OUTPUT_FILE = OUTPUT_DIR / "gdp_nowcast_latest.csv"

def main():
    print("==================================================")
    print(f"   GDP Nowcast Daily Update - {datetime.now()}")
    print("==================================================")

    # 1. 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 运行核心计算
    # start_date 设早一点以保证 Z-Score 计算有足够的历史窗口 (Rolling Window)
    # end_date=None 表示一直算到今天
    print("⚙️  Running pipeline...")
    try:
        daily_df = build_macro_features(start_date="1990-01-01", end_date=None)
    except Exception as e:
        print(f"\n❌ Pipeline 运行失败: {e}")
        return

    # 3. 保存文件
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    daily_df.to_csv(OUTPUT_FILE)

    print("✅ Done! File updated successfully.")
    print("==================================================")
    
    # 打印最后几行确认
    print("\n[Preview - Last 5 Days]")
    print(daily_df.tail())

if __name__ == "__main__":
    main()