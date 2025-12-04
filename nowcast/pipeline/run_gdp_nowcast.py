# CLI script
# nowcast/pipeline/run_gdp_nowcast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.svr import GDPNowcasterSVR

from sklearn.metrics import r2_score, mean_squared_error

def run_backtest(start_date="2010-01-01", end_date=None, freq="M"):
    """
    运行全量历史回测 (Backtest/Vintage Replay)。
    
    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期 (默认今天)
        freq: 评估频率 ('M'=月末, 'D'=每日, 'W'=每周)
              建议 MVP 阶段用 'M'，速度快且能看清趋势。
    """
    print("🚀 Initializing Nowcast Pipeline...")
    
    # 1. 准备全量数据
    # 使用 'offline_mode' 利用本地缓存 (需确保 update_data.py 已运行)
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 2. 构建目标 (y)
    y_full = get_target_series(provider)
    y_full = y_full.dropna()

    # 3. 构建特征面板 (X)
    # 从 yaml 自动读取所有 features
    features_list = [k for k in provider.series_config.keys() if k != 'gdp_real']
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    # 4. 初始化数据集生成器
    # 这一步只是准备好工具，还没开始生成
    gen = AsOfDatasetGenerator(panel_full, y_full)
    
    # 5. 生成评估时间轴
    if end_date is None:
        end_date = pd.Timestamp.now()
    
    # 生成评估日期序列 (e.g., 2010-01-31, 2010-02-28, ...)
    eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    results = []
    
    print(f"📅 Starting Vintage Replay from {start_date} to {str(end_date)[:10]}...")
    print(f"   Total evaluation points: {len(eval_dates)}")

    # ==========================================
    # 主循环 (Time Travel Loop)
    # ==========================================
    for as_of_date in tqdm(eval_dates):
        # --- A. 准备训练集 (Training Set) ---
        # 规则：只能用 as_of_date 之前已经"完结"且GDP已公布的季度进行训练
        # 假设 GDP 发布延迟 90 天
        training_cutoff = as_of_date - pd.Timedelta(days=90)
        
        # 找到所有结束时间早于 cutoff 的季度
        # 这是一个简化处理，直接利用 y_full 的索引
        train_quarters = y_full.index[y_full.index <= training_cutoff]
        
        if len(train_quarters) < 12: 
            # 如果训练样本太少 (比如刚开始回测时)，跳过或给空值
            continue
            
        X_train_list = []
        y_train_list = []
        
        # 构建历史训练样本 (使用 Full Panel, 忽略 Vintage 问题以加速)
        for q_date in train_quarters:
            q_months = gen.get_quarter_months(q_date)
            # 这里的 X 我们用 full panel (假设历史结构已定型)
            X_vec = gen.create_quarterly_feature_vector(q_months, panel_full)
            X_train_list.append(X_vec)
            y_train_list.append(y_full.loc[q_date])
            
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        # --- B. 准备预测样本 (Test Sample) ---
        # 必须严格使用 Vintage Logic (Ragged Edge)
        # generate_dataset 返回的是一个列表，我们这里只取当前这一个点
        current_sample_list = gen.generate_dataset([as_of_date])
        test_sample = current_sample_list[0]
        
        X_test = test_sample.X.reshape(1, -1) # SVR 要求 2D array
        
        # --- C. 训练与预测 ---
        # 每次都重新初始化模型 (Expanding Window)
        model = GDPNowcasterSVR(C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        
        pred_value = model.predict(X_test)[0]
        
        # --- D. 记录结果 ---
        results.append({
            "date": as_of_date,
            "target_quarter": test_sample.quarter_label,
            "nowcast": pred_value,
            "train_size": len(y_train)
        })

    # ==========================================
    # 结果整合与可视化
    # ==========================================
    df_res = pd.DataFrame(results).set_index("date")
    
    print("\n✅ Backtest Complete!")
    
    # --- [新增] 1. 数据对齐与评估 ---
    # 我们需要把 df_res 中的预测值，和 y_full 中的真实值对应起来
    # df_res 有 'target_quarter' (str), y_full index 是 quarter end (timestamp)
    
    # 把 y_full 转成简单的查找表
    y_truth_map = y_full.to_dict()
    # 注意：y_full 的 index 是 timestamp，需要转成和 df_res 一样的字符串格式对比，或者反之
    # 这里我们假设 df_res['target_quarter'] 格式是 'YYYY-MM-DD'
    
    def get_truth(q_str):
        q_ts = pd.Timestamp(q_str)
        return y_truth_map.get(q_ts, np.nan)

    # 将真实值映射回结果表
    df_res['actual'] = df_res['target_quarter'].apply(get_truth)
    
    # 移除没有真实值对应的行 (可能是最近一个季度 GDP 还没出)
    df_eval = df_res.dropna(subset=['actual']).copy()
    
    if len(df_eval) > 0:
        # --- [新增] 2. 计算核心指标 ---
        r2 = r2_score(df_eval['actual'], df_eval['nowcast'])
        mse = mean_squared_error(df_eval['actual'], df_eval['nowcast'])
        var_pred = np.var(df_eval['nowcast'])
        var_actual = np.var(df_eval['actual'])
        
        print("\n📊 Model Performance Metrics:")
        print(f"   R² Score         : {r2:.4f} (越接近1越好，负数说明不如瞎猜)")
        print(f"   MSE              : {mse:.6f}")
        print(f"   Variance (Pred)  : {var_pred:.6f}")
        print(f"   Variance (Actual): {var_actual:.6f}")
        print(f"   Var Ratio (P/A)  : {var_pred/var_actual:.2f} (过低说明欠拟合/死鱼线)")
        
        # --- [新增] 3. 残差分析 ---
        df_eval['residual'] = df_eval['nowcast'] - df_eval['actual']
        
        # 打印残差最大的 5 个时间点 (看看是在哪栽跟头的)
        print("\n📉 Top 5 Largest Errors (Residuals):")
        print(df_eval['residual'].abs().sort_values(ascending=False).head(5))
        
        # 绘图增加残差子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：预测 vs 真实
        ax1.plot(df_res.index, df_res['nowcast'], label='GDP Nowcast', color='blue', linewidth=1.5)
        # 绘制真实值 (红点)
        y_truth_plot = y_full[y_full.index >= pd.to_datetime(start_date)]
        ax1.plot(y_truth_plot.index, y_truth_plot, 'ro', label='Actual GDP', markersize=4)
        ax1.step(y_truth_plot.index, y_truth_plot, where='post', color='red', alpha=0.3, linestyle='--')
        ax1.set_title(f"US Real GDP Nowcast (SVR) | R²={r2:.3f}")
        ax1.set_ylabel("QoQ Annualized Growth")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 下图：残差
        ax2.bar(df_eval.index, df_eval['residual'], color='gray', alpha=0.6, width=20, label='Residual (Pred - Actual)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Error")
        ax2.set_title("Residual Analysis")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        output_path = "nowcast_result_with_metrics.png"
        plt.savefig(output_path)
        print(f"\n📊 Chart saved to {output_path}")
        
    else:
        print("⚠️ Not enough data points to calculate metrics.")

    return df_res

if __name__ == "__main__":
    # 运行回测 (从 2015 年开始，避免早期数据缺失问题)
    # 如果想跑更长，可以改 start_date，但需确保本地 cache 有足够早的数据
    run_backtest(start_date="2015-01-01", freq="M")