import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from .config import ModelConfig

class DeepE2EROM(nn.Module):
    """
    End-to-End Reduced-Order Model with flexible architecture.

    Features:
        - Flexible state autoencoder (1D or 2D)
        - Custom dynamics module (control-affine or any architecture)
        - Optional input autoencoder for control inputs
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Validate that dynamics is provided (moved from ModelConfig)
        if self.config.dynamics is None:
            raise ValueError("dynamics module must be provided")

        # Build or assign components
        self.encoder = config.encoder or self._build_component(config.encoder_arch, "encoder")
        self.decoder = config.decoder or self._build_component(config.decoder_arch, "decoder")
        
        # Build dynamics based on configuration
        self.dynamics = config.dynamics
        
        if config.use_control_autoencoder:
            self.control_encoder = config.control_encoder or self._build_component(config.control_encoder_arch, "control_encoder")
            self.control_decoder = config.control_decoder or self._build_component(config.control_decoder_arch, "control_encoder")
        else:
            self.control_encoder = None
            self.control_decoder = None
            
        # Move model to device
        self.to(config.device)
            
        # Loss function
        self.mse_loss = nn.MSELoss()

    def _build_component(self, arch: Optional[List[Dict]], component_name: str) -> nn.Module:
        """Build component from architecture specification."""
        if arch is None:
            raise ValueError(f"Architecture for {component_name} is not specified.")
        from .builders import build_sequential
        return build_sequential(arch, component_name)
    
    def forward_autoencoder(self, x: torch.Tensor) -> torch.Tensor:
        """Autoencoder forward pass only."""
        batch_size, lookback, *state_dims = x.shape
        
        # Flatten batch and sequence dimensions
        x_flat = x.reshape(-1, *state_dims)
        
        # Encode and decode
        latent = self.encoder(x_flat)
        reconstructed = self.decoder(latent)
        
        # Reshape back to sequence
        return reconstructed.reshape(batch_size, lookback, *state_dims)
    
    def forward_control_autoencoder(self, u: torch.Tensor) -> torch.Tensor:
        """Control input autoencoder forward pass only."""
        if self.control_encoder is None or self.control_decoder is None:
            raise ValueError("Control autoencoder components are not defined.")
        
        batch_size, seq_length, control_dim = u.shape
        
        # Flatten batch and sequence dimensions
        u_flat = u.reshape(-1, control_dim)
        
        # Encode and decode
        latent = self.control_encoder(u_flat)
        reconstructed = self.control_decoder(latent)
        
        # Reshape back to sequence
        return reconstructed.reshape(batch_size, seq_length, control_dim)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple:
        """
        Full forward pass with dynamics prediction.
        
        Args:
            x: Input states of shape (batch, lookback, *state_dims)
            u: Input controls of shape (batch, lookback + pred_horizon - 1, control_dim)
            
        Returns:
            Tuple containing reconstructed and predicted sequences
        """
        batch_size, lookback, *state_dims = x.shape
        u_seq_length = u.shape[1]
        
        assert lookback == self.config.lookback
        assert u_seq_length >= lookback + self.config.pred_horizon - 1, "Control sequence length is short and cannot predict future states for prediction horizon."
        assert self.dynamics is not None, "Dynamics module must be provided."
        
        # Autoencode input sequence
        decoded_sequence = self.forward_autoencoder(x)
        
        # Encode input sequence to latent space
        x_flat = x.reshape(-1, *state_dims)
        latents_flat = self.encoder(x_flat)
        latents = latents_flat.reshape(batch_size, lookback, -1)
        
        # Process controls
        if self.control_encoder is not None:
            assert self.control_decoder is not None, "Control decoder must be provided when using control autoencoder."

            # Autoencode control inputs
            decoded_input_sequence = self.forward_control_autoencoder(u)

            # Encode inputs
            u_flat = u.reshape(-1, self.config.control_dim)
            input_latents_flat = self.control_encoder(u_flat)
            input_latents = input_latents_flat.reshape(batch_size, u_seq_length, -1)
            control_data = input_latents
        else:
            decoded_input_sequence = None
            control_data = u
        
        # Rolling prediction over horizon
        latent_window = latents.clone()
        pred_latents = []
        pred_states = []
        
        for t in range(self.config.pred_horizon):
            # Get state window for dynamics
            current_latent_window = latent_window  # (batch, lookback, latent_dim)
            # Get control window
            current_control_window = control_data[:, t:t + lookback]  # (batch, lookback, control_dim or input_latent_dim)
            
            # Let dynamics module handle the prediction
            next_latent = self.dynamics(current_latent_window, current_control_window)
            
            # Decode to state space
            next_state = self.decoder(next_latent)
            
            pred_latents.append(next_latent.unsqueeze(1))
            pred_states.append(next_state.unsqueeze(1))
            
            # Update latent window (shift and append)
            latent_window = torch.cat([latent_window[:, 1:], next_latent.unsqueeze(1)], dim=1)
        
        pred_latents = torch.cat(pred_latents, dim=1)  # (batch, pred_horizon, latent_dim)
        pred_states = torch.cat(pred_states, dim=1)    # (batch, pred_horizon, *state_dims)
        
        # Return appropriate outputs based on configuration
        if self.control_encoder is not None:
            return decoded_sequence, decoded_input_sequence, pred_latents, pred_states
        else:
            return decoded_sequence, pred_latents, pred_states
    
    
    def count_parameters(self, verbose: bool = True) -> Dict[str, int]:
        """
        Count parameters for each component of the model.
        
        Args:
            verbose: If True, print detailed parameter counts
            
        Returns:
            Dictionary with parameter counts for each component
        """
        param_counts = {}
        
        # Count parameters for each component
        components = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'dynamics': self.dynamics,
        }
        
        if self.control_encoder is not None:
            assert self.control_encoder is not None, "Input decoder must be provided when using input autoencoder."
            components['control_encoder'] = self.control_encoder
            components['control_decoder'] = self.control_decoder
        
        for name, module in components.items():
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[name] = {
                'total': total_params,
                'trainable': trainable_params
            }
        
        # Calculate totals
        total_all = sum(info['total'] for info in param_counts.values())
        trainable_all = sum(info['trainable'] for info in param_counts.values())
        param_counts['total'] = {'total': total_all, 'trainable': trainable_all}
        
        if verbose:
            self._print_parameter_counts(param_counts)
        
        return param_counts
    
    def _print_parameter_counts(self, param_counts: Dict):
        """Print formatted parameter counts."""
        print("=" * 60)
        print("MODEL PARAMETER COUNT")
        print("=" * 60)
        
        for name, counts in param_counts.items():
            if name == 'total':
                continue
            print(f"{name:15} | {counts['total']:>8,} total | {counts['trainable']:>8,} trainable")
        
        print("-" * 60)
        total = param_counts['total']
        print(f"{'TOTAL':15} | {total['total']:>8,} total | {total['trainable']:>8,} trainable")
        print("=" * 60)