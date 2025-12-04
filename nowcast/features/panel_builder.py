import pandas as pd
from nowcast.data.base import DataProvider

class PanelBuilder:
    def __init__(self, provider: DataProvider):
        self.provider = provider

    def build_monthly_panel(self, series_list: list[str], end_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        构建月度特征面板 (X)。
        包含:
        1. 极值截断 (Clipping): ±4 sigma
        2. 频率对齐 (Weekly->Mean, Monthly->Last)
        """
        data_dict = {}
        
        print(f"Building panel with {len(series_list)} indicators (with 4-sigma clipping)...")
        
        for name in series_list:
            try:
                s = self.provider.get_series(name, end_date=end_date)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load '{name}'. Error: {e}")
                continue
            
            # --- Clipping (Winsorization) ---
            if len(s) > 10:
                mean = s.mean()
                std = s.std()
                upper = mean + 4 * std
                lower = mean - 4 * std
                s = s.clip(lower=lower, upper=upper)
            
            # --- Resampling ---
            is_high_freq = False
            inferred_freq = pd.infer_freq(s.index)
            
            if inferred_freq == 'W' or (len(s) > 0 and (s.index[-1] - s.index[0]).days / len(s) < 20):
                is_high_freq = True

            # 使用 'ME' (MonthEnd) 兼容 pandas 2.2+
            if is_high_freq:
                s_monthly = s.resample('ME').mean()
            else:
                s_monthly = s.resample('ME').last()
            
            data_dict[name] = s_monthly

        panel = pd.DataFrame(data_dict)
        panel = panel.sort_index()
        
        if end_date:
            panel = panel.loc[:end_date]
            
        return panel