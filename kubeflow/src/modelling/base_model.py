from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import pandas as pd


class BaseModelStrategy(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_test: pd.DataFrame, y_test: pd.DataFrame,
              find_best_parameters: bool) -> Tuple[Any, Dict]:
        pass

    @abstractmethod
    def get_feature_importance(self, model: Any,
                               features: list) -> pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        pass
