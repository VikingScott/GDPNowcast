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
    'philly_fed_mfg': 0,
    'consumer_sentiment': 0,
    'initial_claims': 0,

    # --- CPI 相关 ---
    'cpi_headline': 15,
    'cpi_core': 15,
    'cpi_food': 15,
    'cpi_shelter': 15,
    'ppi_all': 14,
    'cpi_sticky': 15,
    'hourly_earnings': 7, # 随非农发布，Lag=7

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
        if self.target_freq == "Q":
            return [period_end - pd.offsets.MonthEnd(i) for i in range(3, 0, -1)]
        else:
            return [period_end]

    def mask_data_by_vintage(self, full_data: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """
        根据 as_of_date 屏蔽未来数据。
        [关键修正] 对于 Lag=0 的实时指标，允许在当月内可见。
        """
        masked = full_data.copy()
        
        for col in masked.columns:
            lag_days = PUBLICATION_LAGS.get(col, 30)
            
            if lag_days == 0:
                # [修正逻辑] 实时指标 (Lag=0)
                # 只要 as_of_date 迈入了这个月，哪怕是 1号，我们也认为该月数据（的部分聚合值）可见
                # 逻辑：发布日 = 月初 (MonthBegin)
                publication_dates = masked.index - pd.tseries.offsets.MonthBegin(1)
            else:
                # [原有逻辑] 滞后指标 (Lag > 0)
                # 必须等到 月末 + 滞后天数 才能看到
                publication_dates = masked.index + pd.Timedelta(days=lag_days)
            
            future_mask = publication_dates > as_of_date
            masked.loc[future_mask, col] = np.nan
            
        return masked

    def create_feature_vector(self, 
                              months_to_fetch: List[pd.Timestamp], 
                              vintage_panel: pd.DataFrame) -> tuple[np.ndarray, float]:
        data_slice = vintage_panel.reindex(months_to_fetch)
        
        total_points = data_slice.size
        valid_points = data_slice.count().sum()
        completeness = valid_points / total_points if total_points > 0 else 0.0
        
        data_filled = data_slice.ffill()
        if data_filled.isna().any().any():
            data_filled = data_filled.fillna(0.0)
            
        return data_filled.values.flatten(), completeness

    def generate_dataset(self, as_of_dates: List[pd.Timestamp]) -> List[VintageSample]:
        samples = []
        for as_of in as_of_dates:
            if self.target_freq == "Q":
                target_end = as_of + pd.offsets.QuarterEnd(startingMonth=3)
            else:
                target_end = as_of + pd.offsets.MonthEnd(0)

            masked_X = self.mask_data_by_vintage(self.X_panel, as_of)
            
            if target_end in self.y_series.index:
                y_val = self.y_series.loc[target_end]
            else:
                y_val = None
                
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