# nowcast/features/targets.py

import pandas as pd
from nowcast.data.base import DataProvider

def get_target_series(provider: DataProvider, 
                      target_name: str = "gdp_real",
                      freq: str = "Q") -> pd.Series:
    """
    获取并处理目标变量 (Ground Truth)。
    支持季度 (GDP) 和 月度 (CPI) 两种频率。
    
    Args:
        provider: 数据提供者实例
        target_name: series.yaml 中定义的目标变量名
        freq: 目标频率。
              - 'Q': 季度 (默认, e.g. GDP)，对齐到季度末。
              - 'M': 月度 (e.g. CPI)，对齐到月末。
    
    Returns:
        pd.Series: 经过变换和对齐的目标序列，数值已放大100倍 (e.g. 0.02 -> 2.0)
    """
    # 1. 从 Provider 获取数据 (已做过 yaml 里的 transform)
    series = provider.get_series(target_name)
    
    # 确保索引为 datetime 类型
    series.index = pd.to_datetime(series.index)

    # 2. 索引对齐 (Alignment)
    if freq == "Q":
        # 季度数据：通常 FRED 标记在季初 (1/1)，我们需要对齐到季末 (3/31)
        # 这样在 as_of_dataset 里逻辑更顺：站在 3/31 预测 3/31 的值
        series.index = series.index + pd.tseries.offsets.QuarterEnd(startingMonth=3)
    elif freq == "M":
        # 月度数据：通常 FRED 标记在月初 (1/1)，我们需要对齐到月末 (1/31)
        series.index = series.index + pd.tseries.offsets.MonthEnd(0)
    else:
        raise ValueError(f"Unsupported freq: {freq}. Use 'Q' or 'M'.")

    # 3. 数值放大 (Scaling)
    # 将小数 (0.02) 转换为百分点 (2.0)，帮助线性模型优化器更快收敛
    series = series * 100 
    
    series.name = "target"
    return series