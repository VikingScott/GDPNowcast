# nowcast/pipeline/run_gdp_nowcast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator

from nowcast.models.ridge import GDPNowcasterRidge
from nowcast.models.svr import GDPNowcasterSVR

from sklearn.metrics import r2_score, mean_squared_error

def run_backtest(start_date="2015-01-01", end_date=None, freq="M"):
    """
    运行全量历史回测 (Backtest/Vintage Replay)。
    
    集成特性：
    1. 预计算 (Pre-computation): 将复杂度从 O(N^2) 降为 O(N)，极大加速回测。
    2. 自动调优 (Auto-tune): 调用 RandomizedSearchCV 寻找最佳 C 和 epsilon。
    3. 完整评估: 包含 R2, MSE, Variance Ratio 和残差分析。
    """
    print("🚀 Initializing Nowcast Pipeline...")
    
    # 1. 准备全量数据
    # 使用 'offline_mode' 利用本地缓存 (需确保 update_data.py 已运行)
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 2. 构建目标 (y)
    y_full = get_target_series(provider)
    # [关键修复] 删除因计算增长率产生的首行 NaN，否则 SVR 会报错
    y_full = y_full.dropna()

    # 3. 构建特征面板 (X)
    # 从 yaml 自动读取所有 features
    features_list = [k for k in provider.series_config.keys() if k != 'gdp_real']
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    # 4. 初始化数据集生成器
    gen = AsOfDatasetGenerator(panel_full, y_full)
    
    # 5. 生成评估时间轴
    if end_date is None:
        end_date = pd.Timestamp.now()
    
    eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    print(f"📅 Starting Vintage Replay from {start_date} to {str(end_date)[:10]}...")
    
    # ==========================================
    # [加速优化] 1. 预计算历史训练样本 (Pre-computation)
    # ==========================================
    print("⚡ Pre-computing historical feature vectors to speed up training...")
    
    historical_X_map = {}
    
    # 遍历所有已知的真实 GDP 季度，预先生成对应的特征向量
    # 因为训练集使用的是 Revised Data，这部分是固定的，不需要在循环里重复计算
    for q_date in y_full.index:
        q_months = gen.get_quarter_months(q_date)
        X_vec = gen.create_quarterly_feature_vector(q_months, panel_full)
        historical_X_map[q_date] = X_vec
        
    results = []
    print(f"   Total evaluation points: {len(eval_dates)}")

    # ==========================================
    # 主循环 (Time Travel Loop)
    # ==========================================
    for as_of_date in tqdm(eval_dates):
        # --- A. 准备训练集 (Training Set) ---
        # 规则：只能用 as_of_date 之前已经"完结"且GDP已公布的季度进行训练
        # 假设 GDP 发布延迟 90 天
        training_cutoff = as_of_date - pd.Timedelta(days=90)
        
        # 筛选合法的训练季度
        valid_quarters = y_full.index[y_full.index <= training_cutoff]
        
        if len(valid_quarters) < 12: 
            # 如果训练样本太少 (比如刚开始回测时)，跳过
            continue
            
        # [极速模式] 直接查表获取 X_train，不再重复计算
        # 使用列表推导式从预计算的字典中提取，速度极快
        X_train_list = [historical_X_map[q] for q in valid_quarters]
        # y_train 直接切片
        y_train_list = y_full.loc[valid_quarters].values
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        # --- B. 准备预测样本 (Test Sample) ---
        # Test Sample 必须保持 Vintage Logic (Ragged Edge)，不能预计算
        # 因为每个 as_of_date 看到的数据缺失情况都不一样
        current_sample_list = gen.generate_dataset([as_of_date])
        test_sample = current_sample_list[0]
        
        X_test = test_sample.X.reshape(1, -1) # SVR 要求 2D array
        
        # --- C. 训练与预测 ---

        # 选择切换模型
        model = GDPNowcasterRidge()
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
    
    # --- 1. 数据对齐与评估 ---
    # 构建真实值查找表
    y_truth_map = y_full.to_dict()
    
    def get_truth(q_str):
        q_ts = pd.Timestamp(q_str)
        return y_truth_map.get(q_ts, np.nan)

    # 映射真实值
    df_res['actual'] = df_res['target_quarter'].apply(get_truth)
    
    # [关键修复] 使用 .copy() 避免 SettingWithCopyWarning
    df_eval = df_res.dropna(subset=['actual']).copy()
    
    if len(df_eval) > 0:
        # --- 2. 计算核心指标 ---
        r2 = r2_score(df_eval['actual'], df_eval['nowcast'])
        mse = mean_squared_error(df_eval['actual'], df_eval['nowcast'])
        var_pred = np.var(df_eval['nowcast'])
        var_actual = np.var(df_eval['actual'])
        
        print("\n📊 Model Performance Metrics:")
        print(f"   R² Score         : {r2:.4f} (越接近1越好，负数说明不如瞎猜)")
        print(f"   MSE              : {mse:.6f}")
        print(f"   Variance (Pred)  : {var_pred:.6f}")
        print(f"   Variance (Actual): {var_actual:.6f}")
        # 如果 y 被放大了100倍，这里不需要再调整单位
        print(f"   Var Ratio (P/A)  : {var_pred/var_actual:.2f} (过低说明欠拟合/死鱼线)")
        
        # --- 3. 残差分析 ---
        df_eval['residual'] = df_eval['nowcast'] - df_eval['actual']
        
        print("\n📉 Top 5 Largest Errors (Residuals):")
        print(df_eval['residual'].abs().sort_values(ascending=False).head(5))
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：预测 vs 真实
        ax1.plot(df_res.index, df_res['nowcast'], label='GDP Nowcast', color='blue', linewidth=1.5)
        
        # 绘制真实值 (红点)
        y_truth_plot = y_full[y_full.index >= pd.to_datetime(start_date)]
        ax1.plot(y_truth_plot.index, y_truth_plot, 'ro', label='Actual GDP', markersize=4)
        ax1.step(y_truth_plot.index, y_truth_plot, where='post', color='red', alpha=0.3, linestyle='--')
        
        ax1.set_title(f"US Real GDP Nowcast (SVR Auto-Tuned) | R²={r2:.3f}")
        ax1.set_ylabel("QoQ Annualized Growth (%)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 下图：残差
        ax2.bar(df_eval.index, df_eval['residual'], color='gray', alpha=0.6, width=20, label='Residual (Pred - Actual)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylabel("Error (%)")
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
    # 运行回测
    run_backtest(start_date="2015-01-01", freq="M")