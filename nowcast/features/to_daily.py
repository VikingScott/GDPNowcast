# nowcast/features/to_daily.py

import pandas as pd
import numpy as np

def to_daily_features(nowcast_df: pd.DataFrame, 
                      market_calendar: pd.DatetimeIndex = None,
                      window: int = 252) -> pd.DataFrame:
    """
    将低频的 Nowcast 结果转换为日频交易信号。
    """
    # 1. 确保按时间排序
    df = nowcast_df.sort_index()
    
    # 2. 扩展到日频 (Forward Fill)
    if market_calendar is not None:
        daily_df = df.reindex(market_calendar, method='ffill')
    else:
        # 如果没有提供日历，就按自然日重采样
        daily_df = df.resample('D').ffill()
    
    # 只保留 nowcast 列，重命名方便识别
    daily_df = daily_df[['nowcast']].rename(columns={'nowcast': 'gdp_nowcast'})
    
    # 3. 计算 Rolling Z-Score (标准化信号)
    # 加上 min_periods 防止初期数据不足导致计算错误
    rolling_mean = daily_df['gdp_nowcast'].rolling(window=window, min_periods=60).mean()
    rolling_std = daily_df['gdp_nowcast'].rolling(window=window, min_periods=60).std()
    
    # 加上一个小 epsilon 防止除以 0
    daily_df['gdp_nowcast_z'] = (daily_df['gdp_nowcast'] - rolling_mean) / (rolling_std + 1e-6)
    
    # 4. 生成 Regime 信号
    conditions = [
        daily_df['gdp_nowcast_z'] > 0.5,
        daily_df['gdp_nowcast_z'] < -0.5
    ]
    choices = [1, -1]
    daily_df['growth_regime'] = np.select(conditions, choices, default=0)
    
    # 移除前面的空值（因为 rolling 需要窗口）
    return daily_df.dropna()