# nowcast/data/transforms.py

import numpy as np
import pandas as pd

def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """百分比变化 (e.g. 0.01)"""
    return series.pct_change(periods=periods)

def diff(series: pd.Series, periods: int = 1) -> pd.Series:
    """简单差分 (t - t-1)"""
    return series.diff(periods=periods)

def log_diff(series: pd.Series, periods: int = 1) -> pd.Series:
    """对数差分 (近似增长率，适合波动大的金融数据)"""
    return np.log(series).diff(periods=periods)

def pct_qoq_annualized(series: pd.Series) -> pd.Series:
    """
    GDP 专用：季度环比折年率
    Formula: ((Level_t / Level_{t-1})^4 - 1)
    """
    return (series / series.shift(1)) ** 4 - 1

# 映射表
TRANSFORM_MAP = {
    'pct_mom': lambda s: pct_change(s, 1),
    'pct_yoy': lambda s: pct_change(s, 12),
    'pct_qoq_annualized': pct_qoq_annualized,
    'diff': diff,
    'log_diff': log_diff, 
    'none': lambda s: s
}