# nowcast/models/bayesian.py

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from .base import NowcastModel

class GDPNowcasterBayesian(NowcastModel):
    def __init__(self):
        """
        贝叶斯岭回归模型。
        不需要调参 (Auto-tune)，它会自动推断正则化参数 (alpha/lambda)。
        """
        self.target_name = "gdp_real"
        self._build_model()

    def _build_model(self):
        # BayesianRidge 自身带有正则化机制，不需要 CV
        # 使用 RobustScaler 防止 2020 年数据破坏分布假设
        self.model = make_pipeline(
            RobustScaler(),
            BayesianRidge(compute_score=True) # compute_score=True 方便看模型质量
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        标准预测接口，只返回均值 (Mean)，保持兼容性。
        """
        return self.model.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        [专用接口] 返回均值和标准差。
        Returns:
            mean: 预测均值
            std: 预测标准差 (衡量不确定性)
        """
        # return_std=True 是 BayesianRidge 的特权
        mean, std = self.model.predict(X, return_std=True)
        return mean, std