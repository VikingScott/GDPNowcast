# nowcast/pipeline/run_gdp_nowcast.py

import pandas as pd
import numpy as np
from tqdm import tqdm

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.ridge import GDPNowcasterRidge

# --- 指标分类定义 (用于归因分析) ---
# 硬数据: 实体经济活动
HARD_DATA = [
    'industrial_production', 
    'payrolls', 
    'retail_sales_real', 
    'housing_starts', 
    'initial_claims'
]
# 软数据: 调查/情绪/预期
SOFT_DATA = [
    'philly_fed_mfg', 
    'consumer_sentiment'
]

def run_backtest(start_date="auto", end_date=None, freq="M"):
    """
    运行全量 GDP Nowcast 回测，并生成包含元数据的结果。
    """
    print("🚀 Initializing GDP Nowcast Pipeline...")
    
    # 1. 准备数据
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 2. 构建目标 (GDP, 季度频率)
    # freq='Q' 对齐到季末
    y_full = get_target_series(provider, target_name="gdp_real", freq="Q")
    y_full = y_full.dropna()

    # --- [新增] 自动推断开始时间 ---
    if start_date == "auto":
        # 逻辑：数据最早时间 + 2年预热期 (让Rolling Window有数)
        min_date = y_full.index.min()
        # 加上2年 buffer
        start_date_ts = min_date + pd.DateOffset(years=2)
        print(f"📅 Auto-detected start date: {start_date_ts.date()}")
    else:
        start_date_ts = pd.Timestamp(start_date)
    # -----------------------------

    # 3. 构建特征面板
    features_list = [k for k in provider.series_config.keys() if k != 'gdp_real']
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    # 确定 Hard/Soft 指标在特征向量中的索引位置
    # GDP 特征向量结构是摊平的: [M1_AllFeatures, M2_AllFeatures, M3_AllFeatures]
    # 我们需要找到对应列的索引，以便后续 reshape 和切片
    feat_cols = panel_full.columns.tolist()
    hard_indices = [i for i, col in enumerate(feat_cols) if col in HARD_DATA]
    soft_indices = [i for i, col in enumerate(feat_cols) if col in SOFT_DATA]
    
    # 4. 初始化数据集生成器
    gen = AsOfDatasetGenerator(panel_full, y_full, target_freq="Q")
    
    if end_date is None:
        end_date = pd.Timestamp.now()
    
    # 生成评估日期序列
    eval_dates = pd.date_range(start=start_date_ts, end=end_date, freq=freq)
    
    print(f"📅 Starting Vintage Replay from {start_date_ts.date()} to {str(end_date)[:10]}...")
    
    # ==========================================
    # [加速优化] 预计算历史特征 (Pre-computation)
    # ==========================================
    print("⚡ Pre-computing historical feature vectors...")
    historical_X_map = {}
    # 遍历所有已知的真实 GDP 季度
    for q_date in y_full.index:
        q_months = gen.get_period_months(q_date)
        # 注意：这里 create_feature_vector 返回 (X, score)，我们只取 X
        X_vec, _ = gen.create_feature_vector(q_months, panel_full)
        historical_X_map[q_date] = X_vec
        
    results = []
    
    # ==========================================
    # 主循环 (Time Travel Loop)
    # ==========================================
    for as_of_date in tqdm(eval_dates):
        # --- A. 准备训练集 ---
        # 假设 GDP 发布延迟 90 天，只能用已发布的季度训练
        training_cutoff = as_of_date - pd.Timedelta(days=90)
        valid_quarters = y_full.index[y_full.index <= training_cutoff]
        
        if len(valid_quarters) < 12: 
            continue
            
        # 查表获取训练数据
        X_train_list = [historical_X_map[q] for q in valid_quarters]
        y_train = y_full.loc[valid_quarters].values
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train)
        
        # --- B. 准备预测样本 (Vintage) ---
        current_sample_list = gen.generate_dataset([as_of_date])
        test_sample = current_sample_list[0]
        
        X_test = test_sample.X.reshape(1, -1)
        
        # --- C. 计算 Hard/Soft Z-Score ---
        # 1. 计算训练集的均值和标准差 (用于标准化当前数据)
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0) + 1e-6 # 防止除以0
        
        # 2. 标准化当前样本 (Z-Score)
        X_test_z = (test_sample.X - train_mean) / train_std
        
        # 3. 提取分项 Z 分
        # X 向量长度 = 3 * N_features。结构是 [M1, M2, M3]
        # Reshape 回 (3, N_features)
        n_feats = len(feat_cols)
        X_test_z_reshaped = X_test_z.reshape(3, n_feats)
        
        # 取 3 个月平均的 Z-Score
        avg_z_per_feature = np.mean(X_test_z_reshaped, axis=0)
        
        # 聚合 Hard/Soft
        hard_z = np.mean(avg_z_per_feature[hard_indices]) if hard_indices else 0
        soft_z = np.mean(avg_z_per_feature[soft_indices]) if soft_indices else 0
        
        # --- D. 训练与预测 ---
        model = GDPNowcasterRidge()
        model.fit(X_train, y_train)
        pred_value = model.predict(X_test)[0]
        
        results.append({
            "date": as_of_date,
            "target_quarter": test_sample.label,
            "nowcast": pred_value,
            "data_completeness": test_sample.completeness,
            "hard_data_z": hard_z,
            "soft_data_z": soft_z,
            "train_size": len(y_train)
        })

    # --- 结果处理 ---
    df_res = pd.DataFrame(results).set_index("date")
    print("\n✅ Backtest Complete!")
    
    # 简单映射真实值 (用于调试/绘图，如果不绘图可省略)
    y_truth_map = y_full.to_dict()
    def get_truth(q_str):
        q_ts = pd.Timestamp(q_str)
        return y_truth_map.get(q_ts, np.nan)
    df_res['actual'] = df_res['target_quarter'].apply(get_truth)
    
    return df_res

if __name__ == "__main__":
    # 测试模式：自动寻找最早日期开始
    run_backtest(start_date="auto")