# nowcast/models/ridge.py

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base import NowcastModel

class GDPNowcasterRidge(NowcastModel):
    def __init__(self, alphas=None):
        """
        Ridge 回归模型 (自带交叉验证选择最佳正则化力度 alpha)。
        
        Args:
            alphas: 正则化强度的候选列表。
                    如果为 None，默认尝试从 0.1 到 1000.0 的广泛范围。
        """
        self.target_name = "gdp_real"
        
        # 默认尝试的 alpha 范围 (对数刻度)
        if alphas is None:
            alphas = np.logspace(-1, 3, 50) # 0.1, ..., 1000
            
        self.alphas = alphas
        self._build_model()

    def _build_model(self):
        # 使用 RidgeCV，它会自动通过 Leave-One-Out Cross-Validation 选出最好的 alpha
        # 不需要手写 GridSearch，速度飞快
        self.model = make_pipeline(
            StandardScaler(), # 线性模型必须做标准化
            RidgeCV(alphas=self.alphas, scoring='neg_mean_squared_error')
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        # 调试时可以打印选出来的最佳 alpha
        # best_alpha = self.model.named_steps['ridgecv'].alpha_
        # print(f"Best Ridge Alpha: {best_alpha:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)