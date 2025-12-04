# nowcast/features/targets.py

import pandas as pd
from nowcast.data.base import DataProvider

def get_target_series(provider: DataProvider, 
                      target_name: str = "gdp_real",
                      freq: str = "Q") -> pd.Series:
    """
    获取并处理目标变量 (Ground Truth)。
    支持季度 (GDP) 和 月度 (CPI) 两种频率。
    """
    # 1. 获取数据
    series = provider.get_series(target_name)
    series.index = pd.to_datetime(series.index)

    # 2. 索引对齐
    if freq == "Q":
        # 季度：对齐到季末 (3/31)
        series.index = series.index + pd.tseries.offsets.QuarterEnd(startingMonth=3)
    elif freq == "M":
        # 月度：对齐到月末 (1/31)
        series.index = series.index + pd.tseries.offsets.MonthEnd(0)
    else:
        raise ValueError(f"Unsupported freq: {freq}. Use 'Q' or 'M'.")

    # 3. 数值放大 (Scale by 100)
    # 0.02 -> 2.0，适应模型优化器
    series = series * 100 
    
    series.name = "target"
    return series