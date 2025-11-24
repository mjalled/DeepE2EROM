"""
DeepE2EROM: End-to-End Reduced Order Models using Autoencoders
"""

__version__ = "0.1.0"
__author__ = "Ali Mjalled"
__email__ = "ali.mjalled@ruhr-uni-bochum.de"

from .data import DataPreprocessor
from .utils import animate_1Dsimulation, animate_2Dsimulation
from .models import DeepE2EROM, ModelConfig, ControlAffineDynamics, create_model
#from .training import Trainer

__all__ = [
    "DeepE2EROM", "ModelConfig", "ControlAffineDynamics", "create_model",
    "DataProcessor", "DataConfig", "MinMaxScaler", "StandardScaler", "SequenceDataset",
    "animate_simulation", "animate_comparison", "animate_1Dsimulation"
]

def get_version():
    return __version__