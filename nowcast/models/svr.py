# SVR implementation
# nowcast/models/svr.py

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .base import NowcastModel

class GDPNowcasterSVR(NowcastModel):
    def __init__(self, 
                 kernel='rbf', 
                 C=10.0, 
                 epsilon=0.005, 
                 gamma='scale'):
        """
        SVR 模型包装器。
        内部使用 StandardScaler + SVR 的 Pipeline，
        因为 SVR 对特征的尺度非常敏感。
        """
        self.target_name = "gdp_real"
        # 自动做标准化 (StandardScaler) 是 SVR 的必选项
        self.model = make_pipeline(
            StandardScaler(),
            SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型。
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测。
        """
        return self.model.predict(X)

# ==========================================
# 快速自测
# ==========================================
if __name__ == "__main__":
    # python -m nowcast.models.svr
    
    # 造一点假数据测试流程
    print("Testing SVR Model wrapper...")
    X_dummy = np.random.randn(50, 21) # 50个样本，21个特征
    y_dummy = np.random.randn(50) * 0.02 + 0.02 # 模拟 2% 左右的增长
    
    model = GDPNowcasterSVR(C=10.0)
    model.fit(X_dummy, y_dummy)
    
    pred = model.predict(X_dummy[:5])
    print(f"Predictions: {pred}")
    print("✅ SVR Model test passed!")