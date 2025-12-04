# nowcast/data/fred.py

import os
import yaml
import pandas as pd
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv  # <--- 新增导入
from .base import DataProvider
from .transforms import TRANSFORM_MAP

# 自动加载 .env 文件 (如果存在)
# 它会从当前目录向上寻找 .env，直到找到为止
load_dotenv()  # <--- 新增这行，魔法发生的地方

class FredDataProvider(DataProvider):
    def __init__(self, api_key: str = None, config_path: str = None, cache_dir: str = None):
        """
        Args:
            api_key: FRED API Key
            config_path: series.yaml 路径
            cache_dir: 本地缓存目录 (默认: nowcast/data/cache)
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        
        # 为了兼容之前的“离线模式”逻辑：
        # 如果既没有传参，环境变量里也没有，且不是为了离线测试，才报错。
        if not self.api_key:
             # 这里稍微宽容一点，如果没有key，可以先不报错，
             # 等到真正要联网下载时(get_series里的else分支)再报错。
             # 但为了严谨，我们通常还是建议在这里检查。
             # 现在的逻辑是：如果没有 key，self.fred 初始化可能会失败，或者我们需要手动处理。
             pass 

        # 如果有 key，初始化 fred；如果没有，self.fred 设为 None，只允许读缓存
        if self.api_key:
            self.fred = Fred(api_key=self.api_key)
        else:
            self.fred = None
        
        # 2. Config Setup
        if config_path is None:
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

    def get_series(self, internal_name: str, end_date: pd.Timestamp | None = None) -> pd.Series:
        if internal_name not in self.series_config:
            raise ValueError(f"Series '{internal_name}' not found in config.")
        
        cfg = self.series_config[internal_name]
        fred_code = cfg['code']
        
        # --- A. 获取原始数据 ---
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
                series.to_csv(cache_file)
                print(f"[Cached] Saved to {cache_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch {fred_code}: {e}")

        # --- B. 数据预处理 ---
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()

        transform_name = cfg.get('transform', 'none')
        if transform_name in TRANSFORM_MAP:
            series = TRANSFORM_MAP[transform_name](series)
        elif transform_name != 'none':
            raise ValueError(f"Unknown transform: {transform_name}")

        # --- C. 截断 ---
        if end_date is not None:
            series = series.loc[:end_date]
            
        return series