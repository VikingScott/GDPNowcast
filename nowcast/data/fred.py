# nowcast/data/fred.py

import os
import yaml
import pandas as pd
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv
from .base import DataProvider
from .transforms import TRANSFORM_MAP

# 自动加载 .env 文件
load_dotenv()

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
        
        # 宽容处理：如果没有 Key，允许初始化（用于读取本地缓存），但在联网时会报错
        if self.api_key:
            self.fred = Fred(api_key=self.api_key)
        else:
            self.fred = None
        
        # 2. Config Setup
        if config_path is None:
            # 默认找 ../config/series.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "series.yaml"
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.series_config = yaml.safe_load(f)
            
        # 3. Cache Setup
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_series(self, internal_name: str, end_date: pd.Timestamp | None = None, skip_transform: bool = False) -> pd.Series:
        """
        获取时间序列数据。
        
        Args:
            internal_name: series.yaml 中的键名
            end_date: 截止日期 (Vintage截断)
            skip_transform: 是否跳过数据变换 (如 pct_change)。
                            对于高频数据 (Weekly/Daily)，通常设为 True，
                            以便在 PanelBuilder 中先聚合为月均值，再做变换。
        """
        if internal_name not in self.series_config:
            raise ValueError(f"Series '{internal_name}' not found in config.")
        
        cfg = self.series_config[internal_name]
        fred_code = cfg['code']
        
        # --- A. 获取原始数据 (Raw Data) ---
        cache_file = self.cache_dir / f"{internal_name}.csv"
        
        if cache_file.exists():
            # 命中缓存
            series = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if isinstance(series, pd.DataFrame):
                series = series.squeeze("columns")
        else:
            # 缓存未命中，需要联网
            if self.fred is None:
                raise ValueError(f"Missing FRED API Key and cache not found for {internal_name}. Please set FRED_API_KEY in .env")
                
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

        # [关键逻辑] 仅当 skip_transform 为 False 时才执行变换
        # 如果 PanelBuilder 传入 True，这里会返回原始的 Level 数据（如汽油绝对价格）
        if not skip_transform:
            transform_name = cfg.get('transform', 'none')
            if transform_name in TRANSFORM_MAP:
                series = TRANSFORM_MAP[transform_name](series)
            elif transform_name != 'none':
                raise ValueError(f"Unknown transform: {transform_name}")

        # --- C. 截断 ---
        if end_date is not None:
            series = series.loc[:end_date]
            
        return series