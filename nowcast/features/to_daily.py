# nowcast/features/to_daily.py

import pandas as pd
import numpy as np

def to_daily_features(nowcast_df: pd.DataFrame, 
                      prefix: str,  # [新增] e.g. "gdp" or "cpi"
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
    
    # 2. 统一重命名核心列 (假设输入列名为 'nowcast' 或 'cpi_nowcast' 等)
    # 先找到那个预测值列 (通常是含 'nowcast' 的列)
    pred_col = [c for c in daily_df.columns if 'nowcast' in c][0]
    
    target_col = f"{prefix}_nowcast"
    daily_df = daily_df.rename(columns={pred_col: target_col})
    
    # 重命名元数据列 (如果存在)
    meta_cols = ['data_completeness', 'hard_data_z', 'soft_data_z']
    rename_map = {c: f"{prefix}_{c}" for c in meta_cols if c in daily_df.columns}
    daily_df = daily_df.rename(columns=rename_map)
    
    # 3. 输出端平滑 (5日均线)
    daily_df[target_col] = daily_df[target_col].rolling(window=5, min_periods=1).mean()
    
    # 4. 计算 Z-Score
    rolling_mean = daily_df[target_col].rolling(window=window, min_periods=60).mean()
    rolling_std = daily_df[target_col].rolling(window=window, min_periods=60).std()
    
    z_col = f"{prefix}_nowcast_z"
    daily_df[z_col] = (daily_df[target_col] - rolling_mean) / (rolling_std + 1e-6)
    
    # 5. 生成 Regime
    regime_col = f"{prefix}_regime"
    conditions = [
        daily_df[z_col] > 0.5,
        daily_df[z_col] < -0.5
    ]
    choices = [1, -1]
    daily_df[regime_col] = np.select(conditions, choices, default=0)
    
    # 只保留相关列
    cols_to_keep = [c for c in daily_df.columns if c.startswith(prefix)]
    return daily_df[cols_to_keep].dropna()