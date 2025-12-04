# nowcast/pipeline/run_gdp_nowcast.py

import pandas as pd
import numpy as np
from tqdm import tqdm

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.bayesian import GDPNowcasterBayesian
from nowcast.evaluation import evaluate_and_print

# --- 指标分类定义 (用于归因分析) ---
HARD_DATA = [
    'industrial_production', 
    'payrolls', 
    'retail_sales_real', 
    'housing_starts', 
    'initial_claims'
]
SOFT_DATA = [
    'philly_fed_mfg', 
    'consumer_sentiment'
]

def run_backtest(start_date="auto", end_date=None, freq="M"):
    """
    运行全量 GDP Nowcast 回测。
    包含：Bayesian 模型、不确定性计算、元数据生成、预计算加速。
    """
    print("🚀 Initializing GDP Nowcast Pipeline (Bayesian)...")
    
    # 1. 准备数据
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 2. 构建目标 (GDP, 季度频率)
    y_full = get_target_series(provider, target_name="gdp_real", freq="Q")
    y_full = y_full.dropna()

    # --- 自动推断开始时间 ---
    if start_date == "auto":
        min_date = y_full.index.min()
        start_date_ts = min_date + pd.DateOffset(years=2) # 2年预热
        print(f"📅 Auto-detected start date: {start_date_ts.date()}")
    else:
        start_date_ts = pd.Timestamp(start_date)

    # 3. 构建特征面板
    features_list = [k for k in provider.series_config.keys() if k != 'gdp_real']
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    # 确定 Hard/Soft 指标索引
    feat_cols = panel_full.columns.tolist()
    hard_indices = [i for i, col in enumerate(feat_cols) if col in HARD_DATA]
    soft_indices = [i for i, col in enumerate(feat_cols) if col in SOFT_DATA]
    
    # 4. 初始化数据集生成器 (季度模式)
    gen = AsOfDatasetGenerator(panel_full, y_full, target_freq="Q")
    
    if end_date is None:
        end_date = pd.Timestamp.now()
    
    eval_dates = pd.date_range(start=start_date_ts, end=end_date, freq=freq)
    
    print(f"📅 Starting Vintage Replay from {start_date_ts.date()} to {str(end_date)[:10]}...")
    
    # ==========================================
    # [加速优化] 预计算历史特征
    # ==========================================
    print("⚡ Pre-computing historical feature vectors...")
    historical_X_map = {}
    for q_date in y_full.index:
        q_months = gen.get_period_months(q_date)
        X_vec, _ = gen.create_feature_vector(q_months, panel_full)
        historical_X_map[q_date] = X_vec
        
    results = []
    
    # ==========================================
    # 主循环
    # ==========================================
    for as_of_date in tqdm(eval_dates):
        # --- A. 准备训练集 ---
        # GDP 发布滞后约 90 天
        training_cutoff = as_of_date - pd.Timedelta(days=90)
        valid_quarters = y_full.index[y_full.index <= training_cutoff]
        
        if len(valid_quarters) < 12: 
            continue
            
        X_train_list = [historical_X_map[q] for q in valid_quarters]
        y_train = y_full.loc[valid_quarters].values
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train)
        
        # --- B. 准备预测样本 ---
        current_sample_list = gen.generate_dataset([as_of_date])
        test_sample = current_sample_list[0]
        X_test = test_sample.X.reshape(1, -1)
        
        # --- C. 计算 Z-Score (归因) ---
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0) + 1e-6
        X_test_z = (test_sample.X - train_mean) / train_std
        # GDP 特征是平铺的 (3个月 * N特征)，需要 Reshape
        X_test_z_reshaped = X_test_z.reshape(3, len(feat_cols))
        avg_z = np.mean(X_test_z_reshaped, axis=0)
        
        hard_z = np.mean(avg_z[hard_indices]) if hard_indices else 0
        soft_z = np.mean(avg_z[soft_indices]) if soft_indices else 0
        
        # --- D. 训练与预测 ---
        model = GDPNowcasterBayesian()
        model.fit(X_train, y_train)
        mean, std = model.predict_uncertainty(X_test)
        
        results.append({
            "date": as_of_date,
            "target_quarter": test_sample.label,
            "nowcast": mean[0],
            "nowcast_std": std[0],
            "data_completeness": test_sample.completeness,
            "hard_data_z": hard_z,
            "soft_data_z": soft_z,
            "train_size": len(y_train)
        })

    # --- 结果处理 ---
    df_res = pd.DataFrame(results).set_index("date")
    print("\n✅ GDP Backtest Complete!")
    
    # 映射真实值并评估
    y_truth_map = y_full.to_dict()
    def get_truth(q_str):
        q_ts = pd.Timestamp(q_str)
        return y_truth_map.get(q_ts, np.nan)
    
    df_res['actual'] = df_res['target_quarter'].apply(get_truth)
    
    # 调用统一评估模块
    evaluate_and_print(df_res, target_col='nowcast', actual_col='actual', label="GDP Real Growth")
    
    return df_res

if __name__ == "__main__":
    run_backtest(start_date="auto")