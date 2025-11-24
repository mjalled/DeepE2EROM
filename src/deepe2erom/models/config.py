from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
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
    input_latent_dim: Optional[int] = None  # For input autoencoder

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
    input_encoder_arch: Optional[List[Dict]] = None
    input_decoder_arch: Optional[List[Dict]] = None
    # Option B: Pre-built modules
    input_encoder: Optional[nn.Module] = None
    input_decoder: Optional[nn.Module] = None

    # Trainin weights
    rec_weight: float = 1.0
    pred_weight: float = 1.0
    latent_weight: float = 1.0
    input_rec_weight: float = 1.0

    # Options
    use_input_autoencoder: bool = False

    def __post_init__(self):
        if self.use_input_autoencoder and self.input_latent_dim is None:
            raise ValueError("input_latent_dim must be specified when using input autoencoder.")
        
        # Validate that dynamics is provided
        if self.dynamics is None:
            raise ValueError("dynamics module must be provided")
        
        # Validate other components
        components = [
            ('encoder', self.encoder_arch, self.encoder),
            ('decoder', self.decoder_arch, self.decoder),
        ]
        
        if self.use_input_autoencoder:
            components.extend([
                ('input_encoder', self.input_encoder_arch, self.input_encoder),
                ('input_decoder', self.input_decoder_arch, self.input_decoder),
            ])
        
        for name, arch, module in components:
            if arch is None and module is None:
                raise ValueError(f"Either {name}_arch or {name} must be provided")
            if arch is not None and module is not None:
                raise ValueError(f"Provide either {name}_arch OR {name}, not both")
