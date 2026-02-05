from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Data splitting
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2

    # Sequence parameters
    lookback: int = 10
    pred_horizon: int = 1
    stride: int = 1

    # scaling options
    scale_states: bool = True
    scale_controls: bool = False
    scale_states_collectively: bool = True # If True, scale all state dimensions together
    scale_controls_collectively: bool = False # If True, scale all control dimensions together
    scaling_method: str = "minmax" # "minmax", "standard" or "none"

    # device configuration
    device: Optional[torch.device] = None

    # Data dimensions
    state_dim: Optional[Tuple] = None
    control_dim: Optional[int] = None

    def __post_init__(self):

        if self.device is None:
            self.device = torch.device('cpu')

        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")