import torch.nn as nn
from typing import List, Dict, Optional
from .config import ModelConfig
from .core import DeepE2EROM

def build_sequential(arch: List[Dict], component_name: str) -> nn.Sequential:
    """Build sequential network from architecture specification."""
    layers = []
    
    for i, layer_spec in enumerate(arch):
        layer_type = layer_spec['type']
        params = layer_spec.get('params', {})
        
        if layer_type == 'Linear':
            layer = nn.Linear(**params)
        elif layer_type == 'Conv2d':
            layer = nn.Conv2d(**params)
        elif layer_type == 'ConvTranspose2d':
            layer = nn.ConvTranspose2d(**params)
        elif layer_type == 'ReLU':
            layer = nn.ReLU()
        elif layer_type == 'Sigmoid':
            layer = nn.Sigmoid()
        elif layer_type == 'Tanh':
            layer = nn.Tanh()
        elif layer_type == 'Flatten':
            layer = nn.Flatten()
        elif layer_type == 'Unflatten':
            layer = nn.Unflatten(**params)
        elif layer_type == 'BatchNorm1d':
            layer = nn.BatchNorm1d(**params)
        elif layer_type == 'BatchNorm2d':
            layer = nn.BatchNorm2d(**params)
        elif layer_type == 'Dropout':
            layer = nn.Dropout(**params)
        else:
            raise ValueError(f"Unknown layer type {layer_type} in {component_name}")
            
        layers.append(layer)
    
    return nn.Sequential(*layers)

def create_model_from_config(config: ModelConfig) -> DeepE2EROM:
    """Create model from configuration."""
    return DeepE2EROM(config)

def create_model(
    state_dim: int,
    control_dim: int, 
    lookback: int,
    pred_horizon: int,
    latent_dim: int,
    dynamics: nn.Module,
    # Core components
    encoder_arch: Optional[List[Dict]] = None,
    decoder_arch: Optional[List[Dict]] = None,
    encoder: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    # Input autoencoder
    use_input_autoencoder: bool = False,
    input_latent_dim: Optional[int] = None,
    input_encoder_arch: Optional[List[Dict]] = None,
    input_decoder_arch: Optional[List[Dict]] = None,
    input_encoder: Optional[nn.Module] = None,
    input_decoder: Optional[nn.Module] = None,
    # Training weights
    rec_weight: float = 1.0,
    pred_weight: float = 1.0,
    latent_weight: float = 1.0,
    input_rec_weight: float = 1.0
) -> DeepE2EROM:
    """Convenience function to create model with flexible dynamics options."""
    
    config = ModelConfig(
        state_dim=state_dim,
        control_dim=control_dim,
        lookback=lookback,
        pred_horizon=pred_horizon,
        latent_dim=latent_dim,
        input_latent_dim=input_latent_dim,
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        dynamics=dynamics,
        input_encoder_arch=input_encoder_arch,
        input_decoder_arch=input_decoder_arch,
        encoder=encoder,
        decoder=decoder,
        input_encoder=input_encoder,
        input_decoder=input_decoder,
        rec_weight=rec_weight,
        pred_weight=pred_weight,
        latent_weight=latent_weight,
        input_rec_weight=input_rec_weight,
        use_input_autoencoder=use_input_autoencoder
    )
    
    return create_model_from_config(config)

