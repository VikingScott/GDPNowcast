# nowcast/features/to_daily.py

import pandas as pd
import numpy as np

def to_daily_features(nowcast_df: pd.DataFrame, 
                      market_calendar: pd.DatetimeIndex = None,
                      window: int = 252) -> pd.DataFrame:
    """
    将低频的 Nowcast 结果转换为日频交易信号。
    增加了输出端的平滑处理。
    """
    # 1. 确保按时间排序
    df = nowcast_df.sort_index()
    
    # 2. 扩展到日频 (Forward Fill)
    if market_calendar is not None:
        daily_df = df.reindex(market_calendar, method='ffill')
    else:
        daily_df = df.resample('D').ffill()
    
    # 重命名
    daily_df = daily_df[['nowcast']].rename(columns={'nowcast': 'gdp_nowcast'})
    
    # --- [新增] 输出端平滑 (5日均线) ---
    # 作用：防止单日的预测跳动触发错误的交易信号
    # 比如周三发了个数据导致预测值跳了一下，周四修正了，用5日均线能平滑这个过程
    daily_df['gdp_nowcast'] = daily_df['gdp_nowcast'].rolling(window=5, min_periods=1).mean()
    
    # 3. 计算 Rolling Z-Score (标准化信号)
    # 使用平滑后的 gdp_nowcast 计算 Z-Score
    rolling_mean = daily_df['gdp_nowcast'].rolling(window=window, min_periods=60).mean()
    rolling_std = daily_df['gdp_nowcast'].rolling(window=window, min_periods=60).std()
    
    daily_df['gdp_nowcast_z'] = (daily_df['gdp_nowcast'] - rolling_mean) / (rolling_std + 1e-6)
    
    # 4. 生成 Regime 信号
    # Z > 0.5  -> Growth Strengthening (1)
    # Z < -0.5 -> Growth Slowing (-1)
    conditions = [
        daily_df['gdp_nowcast_z'] > 0.5,
        daily_df['gdp_nowcast_z'] < -0.5
    ]
    choices = [1, -1]
    daily_df['growth_regime'] = np.select(conditions, choices, default=0)
    
    return daily_df.dropna()