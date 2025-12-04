# scripts/update_data.py

import sys
import os
import shutil
from pathlib import Path

# 确保能找到 nowcast 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nowcast.data.fred import FredDataProvider

def update_all_data():
    print("========================================")
    print("   Starting Data Update (FRED -> Local) ")
    print("========================================")

    # 1. 初始化 Provider
    # 这一步会自动读取 config/series.yaml
    try:
        provider = FredDataProvider()
        print(f"Loaded config from: {provider.series_config.keys()}")
    except Exception as e:
        print(f"❌ Error initializing provider: {e}")
        return

    # 2. 获取缓存目录路径
    cache_dir = provider.cache_dir
    print(f"Cache Directory: {cache_dir.resolve()}\n")

    # 3. 遍历所有指标进行更新
    success_count = 0
    fail_count = 0
    
    for internal_name in provider.series_config.keys():
        print(f"🔄 Updating: {internal_name}...", end=" ", flush=True)
        
        # 核心逻辑：先删除旧缓存，逼迫 FredDataProvider 重新下载
        cache_file = cache_dir / f"{internal_name}.csv"
        if cache_file.exists():
            try:
                os.remove(cache_file)
            except OSError as e:
                print(f"[Error removing cache] {e}")
                fail_count += 1
                continue
        
        # 调用 get_series 会触发下载 + 保存
        try:
            # end_date=None 表示获取最新全量数据
            provider.get_series(internal_name, end_date=None)
            print("✅ Done.")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed! Error: {e}")
            fail_count += 1

    print("\n========================================")
    print(f"Summary: {success_count} Success, {fail_count} Failed.")
    print("========================================")

if __name__ == "__main__":
    update_all_data()