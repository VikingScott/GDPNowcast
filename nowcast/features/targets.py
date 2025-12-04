# Target construction
# nowcast/features/targets.py

import pandas as pd
from nowcast.data.base import DataProvider
from nowcast.data.fred import FredDataProvider

def get_target_series(provider: DataProvider, 
                      target_name: str = "gdp_real",
                      align_to_quarter_end: bool = True) -> pd.Series:
    """
    获取并处理目标变量 (Ground Truth)。
    
    Args:
        provider: 数据提供者实例
        target_name: series.yaml 中定义的目标变量名 (默认 gdp_real)
        align_to_quarter_end: 是否将索引对齐到季度末 (建议 True)
                              e.g. 2023-01-01 -> 2023-03-31
                              这样如果不小心引入了未来数据，更容易察觉。
    
    Returns:
        pd.Series: 经过变换的目标序列 (e.g. Real GDP Growth)
    """
    # 1. 从 Provider 获取数据
    # 注意：Provider 会根据 yaml 配置自动做 pct_qoq_annualized 变换
    series = provider.get_series(target_name)
    
    # 2. 索引对齐到季度末
    if align_to_quarter_end:
        # 确保索引是 datetime 类型
        series.index = pd.to_datetime(series.index)
        # 将季度初 (FRED默认) 调整为季度末
        # offsets.QuarterEnd(0) 会把 1/1 移到 3/31, 4/1 移到 6/30
        series.index = series.index + pd.tseries.offsets.QuarterEnd(startingMonth=3)
    
    series.name = "target"
    return series

# ==========================================
# 自测代码 (Scaffolding Test)
# ==========================================
if __name__ == "__main__":
    # 这里的代码只有在直接运行此文件时才会执行
    # python nowcast/features/targets.py
    
    try:
        print("Testing Target Construction...")
        provider = FredDataProvider()
        
        target = get_target_series(provider)
        
        print(f"\nTarget Name: {target.name}")
        print(f"Frequency  : {pd.infer_freq(target.index)}")
        print(f"Latest Obs : {target.index[-1].date()} -> {target.iloc[-1]:.2%}")
        
        print("\nLast 5 Quarters:")
        print(target.tail(5))
        
        print("\n✅ Target construction successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")