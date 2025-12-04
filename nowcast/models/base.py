# Abstract Model class
# nowcast/models/base.py

from abc import ABC, abstractmethod
import numpy as np

class NowcastModel(ABC):
    """所有 Nowcast 模型的抽象基类"""
    
    target_name: str = "gdp_real"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass