# nowcast/features/cpi_engine.py

import pandas as pd
import numpy as np
from nowcast.data.fred import FredDataProvider
from nowcast.data.transforms import TRANSFORM_MAP

# --- 1. 专家规则配置 ---
CPI_CONFIG = {
    'cpi_headline': ['oil_wti', 'gas_price', 'cpi_headline_lag1'],
    'cpi_core':     ['cpi_core_lag1', 'hourly_earnings_lag1', 'consumer_sentiment'],
    'cpi_energy':   ['oil_wti', 'gas_price'],
    'cpi_food':     ['cpi_food_lag1', 'ppi_all_lag1'],
    'cpi_shelter':  ['cpi_shelter_lag1'],
    'cpi_sticky':   ['cpi_sticky_lag1']
}

# --- 2. 发布滞后天数 ---
PUBLICATION_LAGS = {
    'cpi_headline': 15, 'cpi_core': 15, 'cpi_food': 15, 
    'cpi_shelter': 15, 'cpi_sticky': 15, 'cpi_energy': 15,
    'ppi_all': 14,
    'hourly_earnings': 7,
    'consumer_sentiment': 0,
    'oil_wti': 0,
    'gas_price': 0
}

class CPIFeatureEngine:
    def __init__(self, provider: FredDataProvider):
        self.provider = provider

    def get_clean_dataset(self, target_name: str, as_of_date: str = None) -> pd.DataFrame:
        if target_name not in CPI_CONFIG:
            raise ValueError(f"Unknown target: {target_name}")

        features = CPI_CONFIG[target_name]
        df_dict = {}

        # --- A. 获取目标 ---
        s_target = self._fetch_monthly_series(target_name)
        df_dict['target'] = s_target

        # --- B. 获取特征 ---
        for feat_key in features:
            if feat_key.endswith('_lag1'):
                raw_name = feat_key.replace('_lag1', '')
                is_lagged = True
            else:
                raw_name = feat_key
                is_lagged = False
                
            s_feat = self._fetch_monthly_series(raw_name)
            
            if is_lagged:
                df_dict[feat_key] = s_feat.shift(1)
            else:
                df_dict[feat_key] = s_feat

        # --- C. 合并 ---
        df = pd.DataFrame(df_dict)
        df = df.sort_index()
        
        # --- D. Vintage Masking (防作弊) ---
        if as_of_date:
            current_date = pd.Timestamp(as_of_date)
            
            for col in df.columns:
                if col == 'target': continue
                
                original_name = col.replace('_lag1', '')
                lag = PUBLICATION_LAGS.get(original_name, 30)
                
                # 计算该行数据的"产生时间" (Reference Date)
                if '_lag1' in col:
                    ref_dates = df.index - pd.DateOffset(months=1)
                else:
                    ref_dates = df.index
                
                # 计算"发布时间" (Publication Date)
                if lag == 0:
                    # [关键修复] 实时指标 (Lag=0)
                    # 逻辑: 只要迈入该月 (MonthBegin)，数据即开始可见
                    # e.g. 2月数据 (index=2/28) 在 2月1日 可见
                    pub_dates = ref_dates - pd.tseries.offsets.MonthBegin(1)
                else:
                    # 滞后指标
                    # e.g. 1月数据 (index=1/31) 在 2月15日 可见
                    pub_dates = ref_dates + pd.Timedelta(days=lag)
                
                # Masking: 如果 发布时间 > as_of_date，说明不可见 -> NaN
                mask = pub_dates > current_date
                df.loc[mask, col] = np.nan

        return df

    def _fetch_monthly_series(self, name: str) -> pd.Series:
        s = self.provider.get_series(name, skip_transform=True)
        s.index = pd.to_datetime(s.index)
        
        freq = pd.infer_freq(s.index)
        if freq == 'W' or (len(s) > 10 and (s.index[1] - s.index[0]).days < 20):
            s_monthly = s.resample('ME').mean()
        else:
            s_monthly = s.resample('ME').last()
            
        cfg = self.provider.series_config.get(name, {})
        func_name = cfg.get('transform', 'none')
        
        if func_name in TRANSFORM_MAP:
            s_final = TRANSFORM_MAP[func_name](s_monthly)
        else:
            s_final = s_monthly
            
        if len(s_final) > 24:
            mean = s_final.mean()
            std = s_final.std()
            s_final = s_final.clip(lower=mean - 4*std, upper=mean + 4*std)
            
        return s_final