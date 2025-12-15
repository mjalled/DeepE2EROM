from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    """Configuration class for model creation."""

    # Input dimensions
    state_dim: Union[int, Tuple[int, ...]]
    control_dim: int
    lookback: int
    pred_horizon: int

    # Latent dimensions
    latent_dim: int
    control_latent_dim: Optional[int] = None  # For input autoencoder

    # State autoencoder options
    # Option A: Architecture specifications
    encoder_arch: Optional[List[Dict]] = None
    decoder_arch: Optional[List[Dict]] = None
    # Option B: Pre-built modules
    encoder: Optional[nn.Module] = None
    decoder: Optional[nn.Module] = None

    # Dynamics module 
    dynamics: Optional[nn.Module] = None

    # Input autoencoder options
    # Option A: Architecture specifications
    control_encoder_arch: Optional[List[Dict]] = None
    control_decoder_arch: Optional[List[Dict]] = None
    # Option B: Pre-built modules
    control_encoder: Optional[nn.Module] = None
    control_decoder: Optional[nn.Module] = None

    # Options
    use_control_autoencoder: bool = False

    # device
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cpu')
        
        if self.use_control_autoencoder and self.control_latent_dim is None:
            raise ValueError("input_latent_dim must be specified when using input autoencoder.")
        
        # Validate other components
        components = [
            ('encoder', self.encoder_arch, self.encoder),
            ('decoder', self.decoder_arch, self.decoder),
        ]
        
        if self.use_control_autoencoder:
            components.extend([
                ('input_encoder', self.control_encoder_arch, self.control_encoder),
                ('input_decoder', self.control_decoder_arch, self.control_decoder),
            ])
        
        for name, arch, module in components:
            if arch is None and module is None:
                raise ValueError(f"Either {name}_arch or {name} must be provided")
            if arch is not None and module is not None:
                raise ValueError(f"Provide either {name}_arch OR {name}, not both")
