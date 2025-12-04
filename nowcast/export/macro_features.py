# nowcast/export/macro_features.py

import pandas as pd
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
    # 注意：cpi_res 现在包含多列 (cpi_headline, cpi_core, etc.)
    # to_daily_features 会自动识别 'nowcast' 关键字，我们需要微调它或者手动重命名
    # 鉴于 to_daily_features 目前逻辑比较简单，我们这里手动处理多列 CPI
    
    # --- CPI 多列日频化 ---
    cpi_daily = cpi_res.resample('D').ffill()
    
    # 对每一列预测值做平滑和 Z-Score
    cpi_targets = [c for c in cpi_daily.columns if 'cpi_' in c] # 找出所有预测列
    
    for col in cpi_targets:
        # 平滑
        cpi_daily[col] = cpi_daily[col].rolling(window=5, min_periods=1).mean()
        # Z-Score
        rmean = cpi_daily[col].rolling(window=252, min_periods=60).mean()
        rstd = cpi_daily[col].rolling(window=252, min_periods=60).std()
        cpi_daily[f"{col}_z"] = (cpi_daily[col] - rmean) / (rstd + 1e-6)
    
    # 重命名元数据列，防止冲突
    cpi_daily = cpi_daily.rename(columns={
        'data_completeness': 'cpi_data_completeness',
        'hard_data_z': 'cpi_hard_data_z',
        'soft_data_z': 'cpi_soft_data_z'
    })
    
    # 合并
    combined = pd.merge(gdp_daily, cpi_daily, left_index=True, right_index=True, how='outer')
    combined = combined.ffill().dropna()
    
    print(f"✅ Final Dataset Shape: {combined.shape}")
    print("Columns:", combined.columns.tolist())
    
    return combined

if __name__ == "__main__":
    build_macro_features()