# nowcast/export/macro_features.py

import pandas as pd
import numpy as np
from nowcast.pipeline.run_gdp_nowcast import run_backtest as run_gdp
from nowcast.pipeline.run_cpi_nowcast import run_cpi_backtest as run_cpi
from nowcast.features.to_daily import to_daily_features

def build_macro_features(start_date: str = "1990-01-01", 
                         end_date: str = None) -> pd.DataFrame:
    print(f"\n========== 1. Running GDP Nowcast (from {start_date}) ==========")
    gdp_res = run_gdp(start_date=start_date, end_date=end_date, freq="M")
    
    print(f"\n========== 2. Running CPI Nowcast (from {start_date}) ==========")
    cpi_res = run_cpi(start_date=start_date, end_date=end_date, freq="W-FRI")
    
    print("\n========== 3. Merging & Processing ==========")
    
    # GDP 处理
    gdp_daily = to_daily_features(gdp_res, prefix="gdp")
    
    # CPI 处理
    # [关键修复] 显式选择要 Resample 的数值列
    # 排除 target_period 等字符串
    cols_to_resample = [c for c in cpi_res.columns 
                        if c != 'target_period' 
                        and c != 'date']
    
    # 使用 .last() 聚合，如果有非数值列混入会自动被忽略或报错，所以先切片
    cpi_numeric = cpi_res[cols_to_resample].apply(pd.to_numeric, errors='coerce')
    
    # 聚合到日频
    cpi_daily = cpi_numeric.resample('D').last().ffill()
    
    # 计算 Z-Score
    cpi_targets = [c for c in cpi_daily.columns 
                   if 'cpi_' in c 
                   and 'completeness' not in c]
    
    for col in cpi_targets:
        series = cpi_daily[col]
        # 平滑
        cpi_daily[col] = series.rolling(window=5, min_periods=1).mean()
        # Z-Score
        rmean = series.rolling(window=252, min_periods=60).mean()
        rstd = series.rolling(window=252, min_periods=60).std()
        cpi_daily[f"{col}_z"] = (series - rmean) / (rstd + 1e-6)
    
    # 重命名
    cpi_daily = cpi_daily.rename(columns={
        'data_completeness': 'cpi_data_completeness'
    })
    
    # 合并
    combined = pd.merge(gdp_daily, cpi_daily, left_index=True, right_index=True, how='outer')
    
    # 填充
    combined = combined.ffill().dropna(subset=['gdp_nowcast', 'cpi_headline'])
    
    print(f"✅ Final Dataset Shape: {combined.shape}")
    
    return combined

if __name__ == "__main__":
    build_macro_features()