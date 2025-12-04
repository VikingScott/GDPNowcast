# nowcast/data/fred.py

import os
import yaml
import pandas as pd
from pathlib import Path
from fredapi import Fred
from .base import DataProvider
from .transforms import TRANSFORM_MAP

class FredDataProvider(DataProvider):
    def __init__(self, api_key: str = None, config_path: str = None, cache_dir: str = None):
        """
        Args:
            api_key: FRED API Key
            config_path: series.yaml 路径
            cache_dir: 本地缓存目录 (默认: nowcast/data/cache)
        """
        # 1. API Key Setup
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("No FRED API key found. Set FRED_API_KEY env var or pass explicitly.")
        
        self.fred = Fred(api_key=self.api_key)
        
        # 2. Config Setup
        if config_path is None:
            # 默认找 ../config/series.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "series.yaml"
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.series_config = yaml.safe_load(f)
            
        # 3. Cache Setup (自动创建目录)
        if cache_dir is None:
            # 默认存放在 nowcast/data/cache
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_series(self, internal_name: str, end_date: pd.Timestamp | None = None) -> pd.Series:
        """
        策略：
        1. 检查本地 cache/{internal_name}.csv 是否存在
        2. 如果有 -> 读取
        3. 如果无 -> 联网下载 -> 存入 cache
        4. 执行 transform 和 end_date 截断
        """
        if internal_name not in self.series_config:
            raise ValueError(f"Series '{internal_name}' not found in config.")
        
        cfg = self.series_config[internal_name]
        fred_code = cfg['code']
        
        # --- A. 获取原始数据 (Raw Data) ---
        cache_file = self.cache_dir / f"{internal_name}.csv"
        
        if cache_file.exists():
            # 命中缓存
            # print(f"[Cache Hit] Loading {internal_name} from {cache_file}")
            series = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # read_csv 有时返回 DataFrame, 确保转为 Series
            if isinstance(series, pd.DataFrame):
                series = series.squeeze("columns")
        else:
            # 缓存未命中，联网下载
            print(f"[Fetching] Downloading {fred_code} from FRED API...")
            try:
                series = self.fred.get_series(fred_code)
                series.name = internal_name
                series.index.name = "date"
                
                # 写入缓存
                series.to_csv(cache_file)
                print(f"[Cached] Saved to {cache_file}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to fetch {fred_code}: {e}")

        # --- B. 数据预处理 ---
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()

        # Transform
        transform_name = cfg.get('transform', 'none')
        if transform_name in TRANSFORM_MAP:
            series = TRANSFORM_MAP[transform_name](series)
        elif transform_name != 'none':
            raise ValueError(f"Unknown transform: {transform_name}")

        # --- C. 信息集截断 (Point-in-Time) ---
        if end_date is not None:
            series = series.loc[:end_date]
            
        return series