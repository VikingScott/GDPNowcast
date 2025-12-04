# nowcast/features/to_daily.py

import pandas as pd
import numpy as np

def to_daily_features(nowcast_df: pd.DataFrame, 
                      prefix: str,
                      market_calendar: pd.DatetimeIndex = None,
                      window: int = 252) -> pd.DataFrame:
    """
    将低频 Nowcast 结果转换为日频信号，并添加指定前缀。
    """
    df = nowcast_df.sort_index()
    
    # 1. 扩展到日频
    if market_calendar is not None:
        daily_df = df.reindex(market_calendar, method='ffill')
    else:
        daily_df = df.resample('D').ffill()
    
    # 2. 智能列重命名
    # 找到所有预测列 (nowcast, cpi_headline, cpi_core...)
    # 和所有 std 列 (nowcast_std, cpi_headline_std...)
    # 以及元数据列
    
    # 核心逻辑：凡是 df 里有的列，都带上前缀（除非已经有前缀）
    rename_map = {}
    for col in daily_df.columns:
        if col.startswith("target_") or col == "date":
            continue
        if col.startswith(prefix): # 已经有前缀
            rename_map[col] = col
        else:
            rename_map[col] = f"{prefix}_{col}"
            
    daily_df = daily_df.rename(columns=rename_map)
    
    # 找到主预测列 (通常是 prefix_nowcast 或者 prefix_cpi_headline)
    # 对于 GDP: gdp_nowcast
    # 对于 CPI: cpi_cpi_headline (注意重名风险，Pipeline里 key 已经是 cpi_headline)
    
    # 修正逻辑：Pipeline 输出的 Key 已经是 cpi_headline, cpi_core 等
    # 所以 prefix="cpi" 时，cpi_cpi_headline 会有点丑，但为了统一先这样，或者 pipeline 里改 key
    # 简单起见，我们假设 Pipeline 输出的 key 是干净的
    
    # 寻找主信号列用于算 Regime
    if f"{prefix}_nowcast" in daily_df.columns:
        main_col = f"{prefix}_nowcast"
    elif f"{prefix}_cpi_headline" in daily_df.columns:
        main_col = f"{prefix}_cpi_headline"
    else:
        # Fallback
        main_col = daily_df.columns[0]

    # 3. 平滑 (只对预测值做，不对 Std 做)
    # 识别 value columns (不含 std, z, completeness)
    value_cols = [c for c in daily_df.columns if "_std" not in c and "_z" not in c and "completeness" not in c and "regime" not in c]
    
    for col in value_cols:
        daily_df[col] = daily_df[col].rolling(window=5, min_periods=1).mean()
    
    # 4. 计算 Z-Score & Regime (基于主信号)
    rolling_mean = daily_df[main_col].rolling(window=window, min_periods=60).mean()
    rolling_std = daily_df[main_col].rolling(window=window, min_periods=60).std()
    
    z_col = f"{prefix}_signal_z"
    daily_df[z_col] = (daily_df[main_col] - rolling_mean) / (rolling_std + 1e-6)
    
    regime_col = f"{prefix}_regime"
    daily_df[regime_col] = np.where(daily_df[z_col] > 0.5, 1, np.where(daily_df[z_col] < -0.5, -1, 0))
    
    return daily_df.dropna()