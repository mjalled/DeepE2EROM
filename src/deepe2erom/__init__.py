"""
DeepE2EROM: End-to-End Reduced Order Models using Autoencoders
"""

__version__ = "0.1.0"
__author__ = "Ali Mjalled"
__email__ = "ali.mjalled@ruhr-uni-bochum.de"

from .data import DataPreprocessor, DataConfig, MinMaxScaler, StandardScaler, SequenceDataset
from .utils import animate_1Dsimulation, animate_compare_1Dsimulations, animate_2Dsimulation, animate_compare_2Dsimulations
from .models import DeepE2EROM, ModelConfig, ControlAffineDynamics, LSTMcDynamics, LinearDynamics, create_model
from .training import Trainer

__all__ = [
    "DeepE2EROM", "ModelConfig", "ControlAffineDynamics", "LSTMcDynamics", "LinearDynamics", "create_model",
    "DataPreprocessor", "DataConfig", "MinMaxScaler", "StandardScaler", "SequenceDataset",
    "animate_1Dsimulation", "animate_compare_1Dsimulations", "animate_2Dsimulation", "animate_compare_2Dsimulations",
    "Trainer",
]

def get_version():
    return __version__