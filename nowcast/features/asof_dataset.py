# nowcast/features/asof_dataset.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

# 发布延迟规则 (Vintage Lag)
PUBLICATION_LAGS = {
    'gdp_real': 90,
    'industrial_production': 16,
    'payrolls': 7,
    'retail_sales_real': 16,
    'housing_starts': 19,
    'philly_fed_mfg': 0,
    'consumer_sentiment': 0,
    'initial_claims': 0,
    'cpi_headline': 15,
    'cpi_core': 15,
    'cpi_food': 15,
    'cpi_shelter': 15,
    'ppi_all': 14,
    'cpi_sticky': 15,
    'hourly_earnings': 7,
    'cpi_energy': 15,  # 确保这个也在
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
        masked = full_data.copy()
        
        for col in masked.columns:
            # [关键修改] 如果是滞后特征 (e.g. cpi_headline_lag)，视作"历史已知数据"，Lag=0
            if col.endswith('_lag'):
                lag_days = 0
            else:
                lag_days = PUBLICATION_LAGS.get(col, 30)
            
            if lag_days == 0:
                # 实时指标/滞后指标：当月内可见
                publication_dates = masked.index - pd.tseries.offsets.MonthBegin(1)
            else:
                # 普通指标：下个月可见
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