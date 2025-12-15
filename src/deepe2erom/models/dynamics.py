from typing import Dict, List
import torch
import torch.nn as nn
import warnings
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
        if config.use_control_autoencoder:
            if config.control_latent_dim is None:
                raise ValueError("input_latent_dim must be provided when use_input_autoencoder is True")
            control_latent_dim = config.control_latent_dim
        else:
            if config.control_dim is None:
                raise ValueError("control_dim must be provided")
            control_latent_dim = config.control_dim
            
        self.dynamics_input_dim = (config.latent_dim * config.lookback + 
                                 control_latent_dim * (config.lookback - 1))
        
        # Expected output dims
        self.drift_output_dim = config.latent_dim
        self.input_output_dim = config.latent_dim * control_latent_dim
        
        # Adjust and validate architectures
        drift_arch = self._adjust_architecture(drift_arch, self.dynamics_input_dim, self.drift_output_dim, "drift_arch")
        input_arch = self._adjust_architecture(input_arch, self.dynamics_input_dim, self.input_output_dim, "input_arch")
        
        # Build drift and input networks
        from .builders import build_sequential
        self.drift_net = build_sequential(drift_arch, "drift_net")
        self.input_net = build_sequential(input_arch, "input_net")
        
        # Validate output dimensions
        self._validate_network_outputs()
    
    def _adjust_architecture(self, arch: List[Dict], expected_in: int, expected_out: int, arch_name: str) -> List[Dict]:
        """Adjust architecture if input/output dims don't match, and warn."""
        adjusted_arch = arch.copy()
        
        # Check and adjust first layer in_features
        if adjusted_arch and adjusted_arch[0]["type"] == "Linear":
            if adjusted_arch[0]["params"]["in_features"] != expected_in:
                warnings.warn(f"{arch_name}: First layer in_features {adjusted_arch[0]['params']['in_features']} does not match expected {expected_in}. Correcting to {expected_in}.")
                adjusted_arch[0]["params"]["in_features"] = expected_in
        
        # Check and adjust last layer out_features
        if adjusted_arch and adjusted_arch[-1]["type"] == "Linear":
            if adjusted_arch[-1]["params"]["out_features"] != expected_out:
                warnings.warn(f"{arch_name}: Last layer out_features {adjusted_arch[-1]['params']['out_features']} does not match expected {expected_out}. Correcting to {expected_out}.")
                adjusted_arch[-1]["params"]["out_features"] = expected_out
        
        return adjusted_arch
    
    def _validate_network_outputs(self):
        """Validate that networks produce correct output dimensions."""
        # Test with dummy input to check output shapes
        test_input = torch.randn(2, self.dynamics_input_dim)
        
        drift_output = self.drift_net(test_input)
        if drift_output.shape[1] != self.config.latent_dim:
            raise ValueError(f"Drift network output dimension {drift_output.shape[1]} "
                           f"does not match latent_dim {self.config.latent_dim}")
        
        input_output = self.input_net(test_input)
        
        control_dim_val = self.config.control_latent_dim if self.config.use_control_autoencoder else self.config.control_dim
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
        
        if config.use_control_autoencoder:
            control_latent_dim = config.control_latent_dim
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
    
class LinearDynamics(nn.Module):
    """
    Linear dynamics module for DeepE2EROM.
    
    Implements: next_latent = A * ksi + B * u_current
    
    Where ksi is the current latent state (last in window), and u_current is the current control.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize linear dynamics.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Determine control dimension
        self.control_latent_dim = (config.control_latent_dim 
                              if config.use_control_autoencoder 
                              else config.control_dim)
        if self.control_latent_dim is None:
            raise ValueError("Control dimension must be defined")
        
        # Compute ksi dimension
        self.ksi_dim = config.latent_dim * config.lookback + self.control_latent_dim * (config.lookback - 1)
        
        # Linear layers: A maps ksi to latent, B maps u_current to latent
        self.A = nn.Linear(self.ksi_dim, config.latent_dim)
        self.B = nn.Linear(self.control_latent_dim, config.latent_dim)
        
        # Validate output dimensions (should match latent_dim)
        self._validate_outputs()
    
    def _validate_outputs(self):
        """Validate that linear layers produce correct output dimensions."""
        if self.control_latent_dim is None:
            raise ValueError("Control latent dimension must be defined")
        test_ksi = torch.randn(self.ksi_dim)
        test_u = torch.randn(self.control_latent_dim)
        
        output = self.A(test_ksi) + self.B(test_u)
        if output.shape[0] != self.config.latent_dim:
            raise ValueError(f"Linear dynamics output dimension {output.shape[0]} "
                           f"does not match latent_dim {self.config.latent_dim}")
    
    def forward(self, latent_window: torch.Tensor, control_window: torch.Tensor) -> torch.Tensor:
        """
        Linear dynamics forward pass.
        
        Args:
            latent_window: (batch, lookback, latent_dim) - sequence of latent states
            control_window: (batch, lookback, control_latent_dim) - sequence of controls
            
        Returns:
            next_latent: (batch, latent_dim) - predicted next latent state
        """
        # Reshape latent
        latent = latent_window.view(latent_window.size(0), -1)  # (batch, lookback * latent_dim)
        # Use last latent state and last control
        u_current = control_window[:, -1]  # (batch, control_latent_dim)
        u = control_window[:, :-1,:].reshape(control_window.size(0), -1)
        # Concatenate the latent space and the input signals
        ksi = torch.cat((latent, u), dim=-1)
        
        # Linear dynamics: next_latent = A @ ksi + B @ u_current
        next_latent = self.A(ksi) + self.B(u_current)  # (batch, latent_dim)
        
        return next_latent
