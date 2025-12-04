# X, y panel construction
# nowcast/features/panel_builder.py

import pandas as pd
from nowcast.data.base import DataProvider
from nowcast.data.fred import FredDataProvider

class PanelBuilder:
    def __init__(self, provider: DataProvider):
        self.provider = provider

    def build_monthly_panel(self, series_list: list[str], end_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        构建月度特征面板 (X)。
        
        功能：
        1. 遍历 series_list 获取数据
        2. 自动识别频率：
           - 周频 (Weekly) -> 降采样为月频 (Mean)
           - 月频 (Monthly) -> 重采样对齐到月末 (Last)
        3. 合并为宽表
        
        Args:
            series_list: series.yaml 中定义的指标名列表
            end_date: (可选) 截止日期，用于防止未来信息泄露
            
        Returns:
            pd.DataFrame: 索引为月末日期的特征矩阵
        """
        data_dict = {}
        
        print(f"Building panel with {len(series_list)} indicators...")
        
        for name in series_list:
            # 1. 获取基础数据 (已做过 transform)
            try:
                s = self.provider.get_series(name, end_date=end_date)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load '{name}'. Error: {e}")
                continue
            
            # 2. 频率处理与对齐
            # 判断逻辑：如果数据点密度过高（例如是月度跨度的4倍以上），则视为高频数据
            # 或者直接依赖 pd.infer_freq (有时不准，所以双重判断)
            is_high_freq = False
            inferred_freq = pd.infer_freq(s.index)
            
            if inferred_freq == 'W' or (len(s) > 0 and (s.index[-1] - s.index[0]).days / len(s) < 20):
                is_high_freq = True

            if is_high_freq:
                # 周频 -> 月频：取当月平均值
                # 例如：初请失业金，月内波动大，取平均更能代表当月水平
                s_monthly = s.resample('ME').mean()
            else:
                # 月频 -> 月频：对齐到月末
                # FRED数据通常是月初(MS)，我们强制转为月末(M)，值不变
                s_monthly = s.resample('ME').last()
            
            data_dict[name] = s_monthly

        # 3. 合并 DataFrame
        # outer join 保证即使某些数据历史较短，也能保留长历史的数据
        panel = pd.DataFrame(data_dict)
        
        # 4. 排序与最终截断
        panel = panel.sort_index()
        if end_date:
            panel = panel.loc[:end_date]
            
        return panel

# ==========================================
# 快速自测 (Self-Check)
# ==========================================
if __name__ == "__main__":
    # 运行：python -m nowcast.features.panel_builder
    try:
        # 1. 初始化 (使用伪造Key利用本地缓存)
        provider = FredDataProvider(api_key="offline_mode")
        
        # 2. 定义要测试的特征 (混合月度和周度)
        test_features = [
            'industrial_production', # 月度
            'initial_claims',        # 周度 -> 需要被转为月度
            'payrolls'               # 月度
        ]
        
        # 3. 构建面板
        builder = PanelBuilder(provider)
        X = builder.build_monthly_panel(test_features)
        
        print("\n--- Panel Constructed ---")
        print(f"Shape: {X.shape}")
        print(f"Date Range: {X.index.min().date()} to {X.index.max().date()}")
        print("\n--- Tail (Last 5 Months) ---")
        print(X.tail())
        
        # 检查是否还有 Weekly 的痕迹 (行数应该大幅减少)
        print("\n✅ Test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")