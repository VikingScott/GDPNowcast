# nowcast/features/asof_dataset.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

# 发布延迟规则 (Vintage Lag)
PUBLICATION_LAGS = {
    # --- GDP 相关 ---
    'gdp_real': 90,
    'industrial_production': 16,
    'payrolls': 7,
    'retail_sales_real': 16,
    'housing_starts': 19,
    'philly_fed_mfg': 0,      # 当月发布
    'consumer_sentiment': 0,  # 当月发布
    'initial_claims': 0,      # 实时

    # --- CPI 相关 ---
    'cpi_headline': 15,
    'cpi_core': 15,
    'cpi_food': 15,
    'cpi_shelter': 15,
    'ppi_all': 14,
    'cpi_sticky': 15,
    'hourly_earnings': 7,

    # --- 高频/市场数据 (Lag=0) ---
    'oil_wti': 0,
    'gas_price': 0,
    'inflation_breakeven': 0
}

@dataclass
class VintageSample:
    label: str
    as_of_date: pd.Timestamp
    X: np.ndarray
    y: float | None
    completeness: float = 0.0

class AsOfDatasetGenerator:
    def __init__(self, monthly_panel: pd.DataFrame, target_series: pd.Series, target_freq: str = "Q"):
        self.X_panel = monthly_panel.sort_index()
        self.y_series = target_series.sort_index()
        self.target_freq = target_freq
        
    def get_period_months(self, period_end: pd.Timestamp) -> List[pd.Timestamp]:
        """确定特征提取的时间范围"""
        if self.target_freq == "Q":
            # GDP模式：取该季度对应的 3 个月
            return [period_end - pd.offsets.MonthEnd(i) for i in range(3, 0, -1)]
        else:
            # CPI模式：只取该月当月数据
            return [period_end]

    def mask_data_by_vintage(self, full_data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        根据 as_of_date 屏蔽未来数据。
        [关键修正] 对于 Lag=0 的实时指标，允许在当月内可见。
        """
        masked = full_data.copy()
        
        for col in masked.columns:
            lag_days = PUBLICATION_LAGS.get(col, 30) # 默认滞后30天
            
            if lag_days == 0:
                # [Lag=0 特殊逻辑] 允许看到当月截止目前的数据
                # 逻辑：发布日 = 月初 (MonthBegin)
                publication_dates = masked.index - pd.tseries.offsets.MonthBegin(1)
            else:
                # [普通逻辑] 必须等到 月末 + 滞后天数
                publication_dates = masked.index + pd.Timedelta(days=lag_days)
            
            future_mask = publication_dates > as_of_date
            masked.loc[future_mask, col] = np.nan
            
        return masked

    def create_feature_vector(self, 
                              months_to_fetch: List[pd.Timestamp], 
                              vintage_panel: pd.DataFrame) -> tuple[np.ndarray, float]:
        data_slice = vintage_panel.reindex(months_to_fetch)
        
        # 计算完整度
        total_points = data_slice.size
        valid_points = data_slice.count().sum()
        completeness = valid_points / total_points if total_points > 0 else 0.0
        
        # 填补逻辑 (FFill -> 0.0)
        data_filled = data_slice.ffill()
        if data_filled.isna().any().any():
            data_filled = data_filled.fillna(0.0)
            
        return data_filled.values.flatten(), completeness

    def generate_dataset(self, as_of_dates: List[pd.Timestamp]) -> List[VintageSample]:
        samples = []
        for as_of in as_of_dates:
            # 1. 确定目标周期结束日
            if self.target_freq == "Q":
                target_end = as_of + pd.offsets.QuarterEnd(startingMonth=3)
            else:
                target_end = as_of + pd.offsets.MonthEnd(0)

            # 2. 模拟可见数据
            masked_X = self.mask_data_by_vintage(self.X_panel, as_of)
            
            # 3. 获取 y
            if target_end in self.y_series.index:
                y_val = self.y_series.loc[target_end]
            else:
                y_val = None
                
            # 4. 构建 X
            months_to_fetch = self.get_period_months(target_end)
            X_vec, comp_score = self.create_feature_vector(months_to_fetch, masked_X)
            
            samples.append(VintageSample(
                label=str(target_end.date()),
                as_of_date=as_of,
                X=X_vec,
                y=y_val,
                completeness=comp_score
            ))
        return samples