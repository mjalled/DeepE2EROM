from typing import Dict, List
import torch
import torch.nn as nn
from .config import ModelConfig

class ControlAffineDynamics(nn.Module):
    """
    Control-affine dynamics module for DeepE2EROM.
    
    Implements: z_next = drift(z_sequence, u_sequence) + input(z_sequence, u_sequence) * u_current
    
    Note: The current control is taken as the last element of u_sequence for control-affine structure.
    """
    
    def __init__(self, config: ModelConfig,
                 drift_arch: List[Dict],
                 input_arch: List[Dict]):
        """
        Initialize control-affine dynamics.
        
        Args:
            config: Model configuration
            drift_arch: Architecture specification for drift network
            input_arch: Architecture specification for input network
        """
        super().__init__()
        self.config = config
        
        # Input dimensions for dynamics
        if config.use_input_autoencoder:
            if config.input_latent_dim is None:
                raise ValueError("input_latent_dim must be provided when use_input_autoencoder is True")
            control_latent_dim = config.input_latent_dim
        else:
            if config.control_dim is None:
                raise ValueError("control_dim must be provided")
            control_latent_dim = config.control_dim
            
        self.dynamics_input_dim = (config.latent_dim * config.lookback + 
                                 control_latent_dim * (config.lookback - 1))
        
        # Build drift and input networks
        from .builders import build_sequential
        self.drift_net = build_sequential(drift_arch, "drift_net")
        self.input_net = build_sequential(input_arch, "input_net")
        
        # Validate output dimensions
        self._validate_network_outputs()
    
    def _validate_network_outputs(self):
        """Validate that networks produce correct output dimensions."""
        # Test with dummy input to check output shapes
        test_input = torch.randn(2, self.dynamics_input_dim)
        
        drift_output = self.drift_net(test_input)
        if drift_output.shape[1] != self.config.latent_dim:
            raise ValueError(f"Drift network output dimension {drift_output.shape[1]} "
                           f"does not match latent_dim {self.config.latent_dim}")
        
        input_output = self.input_net(test_input)
        
        control_dim_val = self.config.input_latent_dim if self.config.use_input_autoencoder else self.config.control_dim
        if control_dim_val is None:
             # This should be caught by init checks, but needed for type checker
             raise ValueError("Control dimension is None")

        expected_input_dim = self.config.latent_dim * control_dim_val
        if input_output.shape[1] != expected_input_dim:
            raise ValueError(f"Input network output dimension {input_output.shape[1]} "
                           f"does not match expected {expected_input_dim}")
    
    def forward(self, latent_window: torch.Tensor, control_window: torch.Tensor) -> torch.Tensor:
        """
        Control-affine dynamics forward pass.
        
        Args:
            latent_window: (batch, lookback, latent_dim) - sequence of latent states
            control_window: (batch, lookback, control_latent_dim) - sequence of controls  
            
        Returns:
            next_latent: (batch, latent_dim) - predicted next latent state
        """
        batch_size = latent_window.shape[0]
        
        # For control-affine: use all latent states but only previous controls (lookback-1)
        latent_flat = latent_window.reshape(batch_size, -1)  # (batch, latent_dim * lookback)
        control_prev_flat = control_window[:, :-1].reshape(batch_size, -1)  # (batch, control_latent_dim * (lookback-1))
        
        # Current control for affine term (last element of control window)
        current_control = control_window[:, -1]  # (batch, control_latent_dim)
        
        # Concatenate for dynamics input
        dynamics_input = torch.cat([latent_flat, control_prev_flat], dim=1)
        
        # Compute drift and input terms
        drift = self.drift_net(dynamics_input)  # (batch, latent_dim)
        input_matrix = self.input_net(dynamics_input)  # (batch, latent_dim * control_latent_dim)
        
        # Reshape input matrix and compute affine term
        input_matrix = input_matrix.view(batch_size, self.config.latent_dim, -1)  # (batch, latent_dim, control_latent_dim)
        input_effect = torch.bmm(input_matrix, current_control.unsqueeze(-1)).squeeze(-1)  # (batch, latent_dim)
        
        return drift + input_effect
    
class LSTMcDynamics(nn.Module):
    """Example custom dynamics using LSTM."""
    
    def __init__(self, config: ModelConfig, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.config = config
        
        if config.use_input_autoencoder:
            control_latent_dim = config.input_latent_dim
        else:
            control_latent_dim = config.control_dim
            
        assert control_latent_dim is not None, "Control latent dimension must be defined"
        input_size = config.latent_dim + control_latent_dim

        # LSTM that processes concatenated sequences
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_net = nn.Linear(hidden_size, config.latent_dim)
    
    def forward(self, latent_window: torch.Tensor, control_window: torch.Tensor) -> torch.Tensor:
        # Concatenate along feature dimension
        combined = torch.cat([latent_window, control_window], dim=-1)  # (batch, lookback, latent_dim + control_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Use last hidden state to predict next latent
        next_latent = self.output_net(lstm_out[:, -1])  # (batch, latent_dim)
        
        return next_latent
