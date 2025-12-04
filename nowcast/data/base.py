from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    # 新增 skip_transform 参数，默认为 False 以保持兼容性
    def get_series(self, internal_name: str, end_date: pd.Timestamp | None = None, skip_transform: bool = False) -> pd.Series:
        pass
        """
        根据 internal_name (如 'gdp_real') 获取时间序列数据。
        
        Args:
            internal_name: config/series.yaml 中定义的键名
            end_date: 如果提供，截断在此日期之前（包含），用于防止未来数据泄露
            
        Returns:
            pd.Series: 
                - index: pd.DatetimeIndex (freq set if possible)
                - values: float
                - 已完成 transforms (如 log, diff, pct_change)
        """
        pass