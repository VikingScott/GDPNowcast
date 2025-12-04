# nowcast/export/macro_features.py

import pandas as pd
from nowcast.pipeline.run_gdp_nowcast import run_backtest as run_gdp
from nowcast.pipeline.run_cpi_nowcast import run_cpi_backtest as run_cpi
from nowcast.features.to_daily import to_daily_features

def build_macro_features(start_date: str = "1990-01-01", 
                         end_date: str = None) -> pd.DataFrame:
    """
    [主入口] 并行生成 GDP 和 CPI 的 Nowcast，并合并为单一宽表。
    默认从 1990-01-01 开始，以保证数据质量。
    """
    print(f"\n========== 1. Running GDP Nowcast (from {start_date}) ==========")
    gdp_res = run_gdp(start_date=start_date, end_date=end_date, freq="M")
    
    print(f"\n========== 2. Running CPI Nowcast (from {start_date}) ==========")
    cpi_res = run_cpi(start_date=start_date, end_date=end_date, freq="W-FRI")
    
    print("\n========== 3. Merging & Processing ==========")
    
    # 转换为日频信号 (带前缀)
    # GDP -> gdp_nowcast, gdp_hard_z...
    gdp_daily = to_daily_features(gdp_res, prefix="gdp")
    
    # CPI -> cpi_nowcast, cpi_hard_z...
    cpi_daily = to_daily_features(cpi_res, prefix="cpi")
    
    # 合并 (Outer Join)
    combined = pd.merge(gdp_daily, cpi_daily, left_index=True, right_index=True, how='outer')
    
    # 填充空隙 (ffill) 并去除极早期的空值
    combined = combined.ffill().dropna()
    
    print(f"✅ Final Dataset Shape: {combined.shape}")
    print("Columns:", combined.columns.tolist())
    
    return combined

if __name__ == "__main__":
    df = build_macro_features()
    print(df.tail())