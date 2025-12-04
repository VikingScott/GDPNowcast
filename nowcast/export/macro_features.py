# Main entry point
# nowcast/export/macro_features.py

import pandas as pd
from nowcast.pipeline.run_gdp_nowcast import run_backtest
from nowcast.features.to_daily import to_daily_features

def build_macro_features(start_date: str = "2015-01-01", 
                         end_date: str = None,
                         prices_index: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    [对外接口] 构建全套日频宏观特征。
    
    流程：
    1. 运行 Nowcast Pipeline 生成历史预测序列。
    2. 调用 to_daily_features 转换为日频信号。
    3. 返回清洗好的 DataFrame。
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        prices_index: (可选) 传入 ETF 价格表的 index，用于对齐交易日
    
    Returns:
        pd.DataFrame: 包含 gdp_nowcast_z, growth_regime 等列
    """
    print("🏗️ Building Macro Features...")
    
    # 1. 获取 Nowcast 原始序列 (Month-End)
    # 注意：这里我们复用了 pipeline 里的逻辑，它会自动读取本地缓存
    nowcast_res = run_backtest(start_date=start_date, end_date=end_date, freq="M")
    
    # 2. 转换为日频信号
    daily_features = to_daily_features(nowcast_res, market_calendar=prices_index)
    
    print("✅ Macro Features Ready.")
    print(daily_features.tail())
    
    return daily_features

if __name__ == "__main__":
    # 测试一下接口
    df = build_macro_features(start_date="2018-01-01")
    print(df.describe())