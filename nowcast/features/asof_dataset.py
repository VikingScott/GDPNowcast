# Point-in-time dataset logic
# nowcast/features/asof_dataset.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 定义各指标的发布延迟规则 (天数)
# 这是一个简化的假设，用于模拟 "Vintage Data"
# 例如：delay=40 表示该月数据通常在次月10号左右发布 (30+10)
# delay=0 表示当月发布 (如 Sentiment, Philly Fed)
PUBLICATION_LAGS = {
    'gdp_real': 90,           # GDP 延迟很久，但这主要影响 y，不影响 X
    'industrial_production': 16, # 次月中旬 (15-16号)
    'payrolls': 7,            # 次月首个周五 (约7号)
    'retail_sales_real': 16,  # 次月中旬
    'housing_starts': 19,     # 次月19号左右
    'philly_fed_mfg': 0,      # 当月发布 (第三个周四)
    'consumer_sentiment': 0,  # 当月发布 (初值)
    'initial_claims': 0       # 周频，当月就有数据
}

@dataclass
class QuarterlySample:
    """代表一个训练/预测样本"""
    quarter_label: str       # e.g., "2023Q3"
    as_of_date: pd.Timestamp # 站在哪一天看的
    X: np.ndarray            # 特征向量 (Flattened)
    y: float | None          # 目标值 (GDP Growth)

class AsOfDatasetGenerator:
    def __init__(self, monthly_panel: pd.DataFrame, target_series: pd.Series):
        """
        Args:
            monthly_panel: 月度特征矩阵 (索引为月末)
            target_series: 季度 GDP 目标 (索引为季末)
        """
        self.X_panel = monthly_panel.sort_index()
        self.y_series = target_series.sort_index()
        
    def get_quarter_months(self, quarter_end: pd.Timestamp) -> List[pd.Timestamp]:
        """给定季度末日期，返回该季度的3个月末日期"""
        return [quarter_end - pd.offsets.MonthEnd(i) for i in range(3, 0, -1)]

    def mask_data_by_vintage(self, full_data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        核心函数：根据 as_of_date 模拟数据可见性。
        如果某条数据的 '理论发布时间' 晚于 as_of_date，则将其设为 NaN。
        """
        masked = full_data.copy()
        
        for col in masked.columns:
            # 获取该指标的发布延迟 (默认 30 天)
            lag_days = PUBLICATION_LAGS.get(col, 30)
            
            # 计算每行数据 '理论上' 应该被看到的日期
            # 假设 index 是月末 (MonthEnd)
            # 发布日 = 月末 + lag_days
            publication_dates = masked.index + pd.Timedelta(days=lag_days)
            
            # 找到那些 "未来才发布" 的数据点
            future_mask = publication_dates > as_of_date
            
            # 将其设为 NaN (不可见)
            masked.loc[future_mask, col] = np.nan
            
        return masked

    def create_quarterly_feature_vector(self, 
                                        quarter_months: List[pd.Timestamp], 
                                        vintage_panel: pd.DataFrame) -> np.ndarray:
        """
        将3个月的特征摊平为一个向量。并进行简单的缺失值填补。
        """
        # 截取该季度的3个月数据
        # reindex 确保即使数据缺失也能保留占位符 (NaN)
        q_data = vintage_panel.reindex(quarter_months)
        
        # --- 简单的填补逻辑 (Imputation) ---
        # 1. 优先用 ffill (最近的已知值)
        q_data_filled = q_data.ffill()
        
        # 2. 如果还有 NaN (比如季度第一个月就缺失)，用整体均值填补 (Neutral Value)
        # 注意：这里用的是 vintage_panel 的均值，避免未来信息
        # 更好的做法是用 expanding mean，但 MVP 简化处理
        if q_data_filled.isna().any().any():
            q_data_filled = q_data_filled.fillna(vintage_panel.mean())
            
        # 3. 再次兜底 (防止全空列导致 mean 也是 NaN)
        q_data_filled = q_data_filled.fillna(0.0)
        
        # 摊平：[M1_Feat1, ..., M1_FeatN, M2_Feat1, ..., M3_FeatN]
        return q_data_filled.values.flatten()

    def generate_dataset(self, as_of_dates: List[pd.Timestamp]) -> List[QuarterlySample]:
        """
        生成数据集。
        对每个 as_of_date，它尝试预测 "当时所在的季度" 或 "刚结束的季度" 的 GDP。
        """
        samples = []
        
        for as_of in as_of_dates:
            # 1. 模拟 Vintage 数据 (这一步最关键！)
            # 此时 masked_X 里只有 as_of 当天能看到的数据
            masked_X = self.mask_data_by_vintage(self.X_panel, as_of)
            
            # 2. 确定目标季度 (Target Quarter)
            # 简单策略：预测 "as_of 日期所在的季度"
            # e.g., if as_of is 2023-05-15, target is 2023Q2 (ends 2023-06-30)
            target_q_end = as_of + pd.offsets.QuarterEnd(startingMonth=3)
            
            # 3. 获取目标值 (y)
            # 训练时我们需要 y，预测时 y 可以是 None
            # 注意：这里的 y 是事后真值 (Ground Truth)，用于训练
            if target_q_end in self.y_series.index:
                y_val = self.y_series.loc[target_q_end]
            else:
                # 目标季度还没结束或 GDP 还没出，无法用于训练
                y_val = None
                
            # 4. 构建特征 (X)
            q_months = self.get_quarter_months(target_q_end)
            X_vec = self.create_quarterly_feature_vector(q_months, masked_X)
            
            samples.append(QuarterlySample(
                quarter_label=str(target_q_end.date()),
                as_of_date=as_of,
                X=X_vec,
                y=y_val
            ))
            
        return samples

# ==========================================
# 自测代码
# ==========================================
if __name__ == "__main__":
    # python -m nowcast.features.asof_dataset
    from nowcast.data.fred import FredDataProvider
    from nowcast.features.panel_builder import PanelBuilder
    from nowcast.features.targets import get_target_series
    
    print("Testing AsOfDataset logic...")
    
    # 1. 准备数据
    provider = FredDataProvider(api_key="offline_mode")
    target = get_target_series(provider)
    
    # 使用你确认的 7 个特征 (philly_fed 替换 ism)
    feats = [
        'industrial_production', 'payrolls', 'retail_sales_real',
        'housing_starts', 'philly_fed_mfg', 'consumer_sentiment', 'initial_claims'
    ]
    builder = PanelBuilder(provider)
    panel = builder.build_monthly_panel(feats)
    
    # 2. 初始化生成器
    gen = AsOfDatasetGenerator(panel, target)
    
    # 3. 模拟几个关键的时间点测试 "Ragged Edge"
    test_dates = [
        pd.Timestamp("2020-04-15"), # 疫情初期，Q2刚开始，只有M1部分数据
        pd.Timestamp("2020-05-15"), # Q2中期
        pd.Timestamp("2020-07-30")  # Q2结束，等待GDP公布
    ]
    
    samples = gen.generate_dataset(test_dates)
    
    for s in samples:
        print(f"\n--- As Of: {s.as_of_date.date()} ---")
        print(f"Target Quarter: {s.quarter_label}")
        print(f"y (Truth): {s.y:.4f}" if s.y else "y: None")
        print(f"X shape: {s.X.shape}") 
        # 验证维度: 7个指标 * 3个月 = 21
        print(f"X Sample (First 7 - Month 1): {s.X[:7]}")