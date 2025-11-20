import numpy as np
from typing import Optional

class MinMaxScaler:
    """Scale data to [0,1] range."""

    def __init__(self):
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.data_min) / (self.data_max - self.data_min + 1e-8)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Scaler has not been fitted yet.")
        return data * (self.data_max - self.data_min + 1e-8) + self.data_min
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)
    
class StandardScaler:
    """Scale data to zero mean and unit variance."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.mean) / (self.std + 1e-8)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return data * (self.std + 1e-8) + self.mean
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)