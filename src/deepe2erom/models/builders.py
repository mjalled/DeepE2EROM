import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
from .config import ModelConfig
from .core import DeepE2EROM
from .dynamics import ControlAffineDynamics, LSTMcDynamics, LinearDynamics  # Updated import

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

def _create_dynamics(dynamics_type: str, config: ModelConfig, **kwargs) -> nn.Module:
    """Factory function to create dynamics based on type."""
    if dynamics_type == "control_affine":
        drift_arch = kwargs.get("drift_arch")
        input_arch = kwargs.get("input_arch")
        if drift_arch is None or input_arch is None:
            raise ValueError("drift_arch and input_arch must be provided for control_affine dynamics")
        return ControlAffineDynamics(config, drift_arch, input_arch)
    elif dynamics_type == "lstm":
        lstm_hidden_size = kwargs.get("lstm_hidden_size", 128)
        lstm_layers = kwargs.get("lstm_layers", 1)
        return LSTMcDynamics(config, lstm_hidden_size, lstm_layers)
    elif dynamics_type == "linear":
        return LinearDynamics(config)
    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}. Supported: 'control_affine', 'lstm', 'linear'")

def create_model(
    state_dim: Union[int, Tuple[int, ...]],
    control_dim: int, 
    lookback: int,
    pred_horizon: int,
    latent_dim: int,
    # Dynamics options: either specify type + archs, or pass pre-built module
    dynamics_type: Optional[str] = None,  # e.g., "control_affine", "lstm", "linear"
    dynamics: Optional[nn.Module] = None,  # Pre-built dynamics (for custom/advanced use)
    drift_arch: Optional[List[Dict]] = None,  # For control_affine
    input_arch: Optional[List[Dict]] = None,  # For control_affine
    lstm_hidden_size: Optional[int] = 128,  # For lstm
    lstm_layers: Optional[int] = 1,  # For lstm
    # Core components
    encoder_arch: Optional[List[Dict]] = None,
    decoder_arch: Optional[List[Dict]] = None,
    encoder: Optional[nn.Module] = None,
    decoder: Optional[nn.Module] = None,
    # Control autoencoder
    use_control_autoencoder: bool = False,
    control_latent_dim: Optional[int] = None,
    control_encoder_arch: Optional[List[Dict]] = None,
    control_decoder_arch: Optional[List[Dict]] = None,
    control_encoder: Optional[nn.Module] = None,
    control_decoder: Optional[nn.Module] = None,
    # device
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> DeepE2EROM:
    """Convenience function to create model with flexible dynamics options."""
    
    if seed is not None:
        torch.manual_seed(seed)

    # Validate dynamics specification
    if dynamics_type is not None and dynamics is not None:
        raise ValueError("Provide either dynamics_type OR dynamics, not both")
    if dynamics_type is None and dynamics is None:
        raise ValueError("Either dynamics_type or dynamics must be provided")
    
    # Build config without dynamics first
    config = ModelConfig(
        state_dim=state_dim,
        control_dim=control_dim,
        lookback=lookback,
        pred_horizon=pred_horizon,
        latent_dim=latent_dim,
        control_latent_dim=control_latent_dim,
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        dynamics=None,  # Set later
        control_encoder_arch=control_encoder_arch,
        control_decoder_arch=control_decoder_arch,
        encoder=encoder,
        decoder=decoder,
        control_encoder=control_encoder,
        control_decoder=control_decoder,
        use_control_autoencoder=use_control_autoencoder,
        device=device
    )
    
    # Create dynamics based on type or use provided
    if dynamics_type is not None:
        dynamics = _create_dynamics(dynamics_type, config, 
                                    drift_arch=drift_arch, input_arch=input_arch,
                                    lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers)
    
    # Set dynamics in config
    config.dynamics = dynamics
    
    return create_model_from_config(config)

