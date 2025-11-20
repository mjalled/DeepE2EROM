import torch
from torch.utils.data import Dataset
from typing import Tuple

class SequenceDataset(Dataset):
    """Pytorch Dataset for sequence data."""

    def __init__(self, states: torch.Tensor, controls: torch.Tensor, targets: torch.Tensor):
        self.states = states
        self.controls = controls
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.controls[idx], self.targets[idx]

class SimulationDataset:
    """Container for original simulation data."""

    def __init__(self, states: torch.Tensor, controls: torch.Tensor,
                 states_scaled: torch.Tensor, controls_scaled: torch.Tensor):
        
        self.states = states
        self.controls = controls
        self.states_scaled = states_scaled
        self.controls_scaled = controls_scaled

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.states.shape