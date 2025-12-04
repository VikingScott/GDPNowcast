# debug_cpi_engine.py

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 修复路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from nowcast.data.fred import FredDataProvider
from nowcast.features.cpi_engine import CPIFeatureEngine

def test_engine():
    print("🚀 Testing CPI Feature Engine...")
    
    provider = FredDataProvider(api_key="offline_mode")
    engine = CPIFeatureEngine(provider)
    
    # 设定测试视角：2025年2月10日
    test_date = "2025-02-10"
    target_month = "2025-02-28" # 我们要在 DataFrame 里找这行
    
    print(f"\n📅 Scenario: Standing at {test_date}")
    print(f"   Looking for data of month: {target_month}")
    
    # 获取数据 (应用 Masking)
    df = engine.get_clean_dataset('cpi_headline', as_of_date=test_date)
    
    # 1. 检查是否存在 2025-02-28 这一行
    try:
        row = df.loc[pd.Timestamp(target_month)]
        print(f"\n✅ Found row for {target_month}")
        print(row)
    except KeyError:
        print(f"\n❌ Row {target_month} not found in dataset!")
        print("   Dataset tail index:", df.index[-3:])
        return

    print("\n--- Verification ---")
    
    # 2. 检查 Oil (Lag=0) -> 应该有值
    # 逻辑：2月10日，应该能看到 2月1日-10日 的平均油价
    oil_val = row.get('oil_wti', np.nan)
    if not pd.isna(oil_val):
        print(f"✅ [PASS] Feb Oil is visible: {oil_val:.4f}")
    else:
        print(f"❌ [FAIL] Feb Oil is NaN! (Check if source data exists for Feb 2025)")

    # 3. 检查 CPI Lag (Lag=15) -> 应该为 NaN
    # 逻辑：这行是 'cpi_headline_lag1'，装的是 1月 CPI。
    # 1月 CPI 发布日 = 1月31 + 15天 = 2月15日。
    # 今天是 2月10日 < 2月15日，所以还没发布 -> 应该 Mask 为 NaN
    cpi_lag_val = row.get('cpi_headline_lag1', np.nan)
    if pd.isna(cpi_lag_val):
        print(f"✅ [PASS] Jan CPI (Lag1) is Masked. Correct! (Release date > Feb 10)")
    else:
        print(f"❌ [FAIL] Jan CPI is visible: {cpi_lag_val}. Future Leakage!")

    # 4. 检查相关性 (用全量历史，不带 Masking)
    print("\n--- Correlation Check (Full History) ---")
    df_full = engine.get_clean_dataset('cpi_headline', as_of_date=None)
    corr = df_full.corr()['target'].sort_values(ascending=False)
    print(corr)

if __name__ == "__main__":
    test_engine()