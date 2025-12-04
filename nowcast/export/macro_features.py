# nowcast/export/macro_features.py

import pandas as pd
import numpy as np
from nowcast.pipeline.run_gdp_nowcast import run_backtest as run_gdp
from nowcast.pipeline.run_cpi_nowcast import run_cpi_backtest as run_cpi
from nowcast.features.to_daily import to_daily_features

def build_macro_features(start_date: str = "1990-01-01", 
                         end_date: str = None) -> pd.DataFrame:
    """
    [主入口] 并行生成 GDP 和 CPI 的 Nowcast，并合并为单一宽表。
    """
    print(f"\n========== 1. Running GDP Nowcast (from {start_date}) ==========")
    gdp_res = run_gdp(start_date=start_date, end_date=end_date, freq="M")
    
    print(f"\n========== 2. Running CPI Nowcast (from {start_date}) ==========")
    cpi_res = run_cpi(start_date=start_date, end_date=end_date, freq="W-FRI")
    
    print("\n========== 3. Merging & Processing ==========")
    
    # GDP 处理
    gdp_daily = to_daily_features(gdp_res, prefix="gdp")
    
    # CPI 处理
    # 1. 扩展到日频
    cpi_daily = cpi_res.resample('D').ffill()
    
    # 2. 识别需要计算 Z-Score 的列 (排除元数据和字符串)
    # 找出所有 "cpi_xxx" 且不是 std, completeness, z 的列
    cpi_targets = [c for c in cpi_daily.columns 
                   if 'cpi_' in c 
                   and '_std' not in c 
                   and '_z' not in c 
                   and 'completeness' not in c
                   and c != 'target_period']
    
    # 3. 平滑与 Z-Score
    for col in cpi_targets:
        # [Fix] 强制转为 float，防止 Object 类型报错
        series = cpi_daily[col].astype(float)
        
        # 平滑
        cpi_daily[col] = series.rolling(window=5, min_periods=1).mean()
        
        # Z-Score
        rmean = series.rolling(window=252, min_periods=60).mean()
        rstd = series.rolling(window=252, min_periods=60).std()
        cpi_daily[f"{col}_z"] = (series - rmean) / (rstd + 1e-6)
    
    # 4. 重命名 CPI 元数据，防止与 GDP 冲突
    cpi_daily = cpi_daily.rename(columns={
        'data_completeness': 'cpi_data_completeness',
        'hard_data_z': 'cpi_hard_data_z',
        'soft_data_z': 'cpi_soft_data_z'
    })
    
    # 5. 合并
    combined = pd.merge(gdp_daily, cpi_daily, left_index=True, right_index=True, how='outer')
    
    # 填充与清洗
    combined = combined.ffill().dropna(subset=['gdp_nowcast', 'cpi_headline'])
    
    print(f"✅ Final Dataset Shape: {combined.shape}")
    print("Columns:", combined.columns.tolist())
    
    return combined

if __name__ == "__main__":
    df = build_macro_features()
    print(df.tail())