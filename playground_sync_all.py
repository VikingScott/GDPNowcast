# playground_sync_all.py
from src.data.fred_sync import sync_all_series

if __name__ == "__main__":
    sync_all_series(
        config_path="config/series.yaml",
        data_dir="data/raw",
        observation_start="1980-01-01",
        observation_end=None,
    )
