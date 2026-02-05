import numpy as np
import torch
from .config import DataConfig
from .scalers import MinMaxScaler, StandardScaler
from .datasets import SequenceDataset, SimulationDataset
from typing import Dict, Optional, Tuple, Any, Union

class DataPreprocessor:
    """Process raw simulations for training control-affine ROMs."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.state_scaler = None
        self.control_scaler = None
        self._processed = False

    def process_simulations(self,
                             states: np.ndarray,
                             controls: np.ndarray,
                             split_data: bool = True) -> Dict[str, Any]:
        
        """
        Process raw simulation data into training sequences and preserves original simulations.

        Args:
            states (np.ndarray): Array of shape (n_simulations, n_timesteps, *state_dims)
            controls (np.ndarray): Array of shape (n_simulations, n_timesteps, control_dim)
            split_data (bool): Whether to split data into train/val/test sets
        
        Returns:
            Dictionary containing:
            - 'datasets': PyTorch datasets for training (sequences)
            - 'simulations': Original simulations for evaluation
            - 'metadata': Processing information     
        """

        # validate input shapes
        self._validate_inputs(states, controls)

        # Split data
        if split_data:
            data_splits = self._split_simulations(states, controls)
        else:
            data_splits = {"full": (states, controls)}

        # Scale data
        scaled_data = self._scale_and_preserve_originals(data_splits)

        # create sequences
        sequence_splits = self._create_sequences(scaled_data["scaled"])

        # Create Pytorch datasets
        dataset_splits = self._create_datasets(sequence_splits)

        # prepare simulation data
        simulation_splits = self._prepare_simulation_data(scaled_data)

        self._processed = True

        return {
            "datasets": dataset_splits, # sequences for training
            "simulations": simulation_splits, # original simulations for evaluation
            "metadata": {
                "state_scaler": self.state_scaler,
                "control_scaler": self.control_scaler,
                "config": self.config
            }
        }
    
    def _validate_inputs(self, states: np.ndarray, controls: np.ndarray) -> None:
        """Validate input data shapes and dimensions."""
        
        if states.ndim not in [3,5]:
            raise ValueError("States must be 3D (n_simulations, n_timesteps, state_dim) for 1D simulations or 5D (n_simulations, n_timesteps, C, H, W).")
        if controls.ndim != 3:
            raise ValueError("Controls must be 3D (n_simulations, n_timesteps, control_dim).")
        if states.shape[0] != controls.shape[0]:
            raise ValueError("Number of simulations in states and controls must match.")
        if states.shape[1] != controls.shape[1]:
            raise ValueError("Number of timesteps in states and controls must match.")
        
    def _split_simulations(self, states: np.ndarray, controls: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split simulations into train/val/test sets."""
        
        n_simulations = states.shape[0]
        splits = {}

        # If we have only one simulation, split along the time axis
        if n_simulations == 1:
            n_timesteps = states.shape[1]
            train_end = int(n_timesteps * self.config.train_split)
            val_end = train_end + int(n_timesteps * self.config.val_split)

            splits["train"] = (states[:, :train_end], controls[:, :train_end])
            splits["val"] = (states[:, train_end:val_end], controls[:, train_end:val_end])
            splits["test"] = (states[:, val_end:], controls[:, val_end:])
            
            print(f"Single simulation detected. Splitting along time axis: Train={train_end}, Val={val_end-train_end}, Test={n_timesteps-val_end} steps.")

        else:
            train_end = int(n_simulations * self.config.train_split)
            val_end = train_end + int(n_simulations * self.config.val_split)

            splits["train"] = (states[:train_end], controls[:train_end])
            splits["val"] = (states[train_end:val_end], controls[train_end:val_end])
            splits["test"] = (states[val_end:], controls[val_end:])

        return splits

    def _scale_and_preserve_originals(self, data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Scale  data while preserving original simulations."""
        
        scaled_data = {"original": {},
                       "scaled": {}}
        
        for split, (states, controls) in data_splits.items():
            scaled_data["original"][split] = (states.copy(), controls.copy())

        reference_split = "train" if "train" in data_splits else next(iter(data_splits))

        # Scale states if required
        if self.config.scale_states and self.config.scaling_method != "none":
            if self.config.scaling_method == "minmax":
                self.state_scaler = MinMaxScaler()
            elif self.config.scaling_method == "standard":
                self.state_scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {self.config.scaling_method}")
            
            print("Scaling states using: ", self.config.scaling_method)
            ref_states = data_splits[reference_split][0]
            ref_states_scaled = self._apply_scaling_to_sequences(ref_states, self.state_scaler, fit=True)

            for split, (states, controls) in data_splits.items():
                states_scaled = ref_states_scaled if split == reference_split else self._apply_scaling_to_sequences(states, self.state_scaler, fit=False)
                scaled_data["scaled"][split] = (states_scaled, controls.copy())
        
        # No scaling - use original
        else:   
            for split, (states, controls) in scaled_data["original"].items():
                scaled_data["scaled"][split] = (states.copy(), controls.copy())

        # Scale controls if needed
        if self.config.scale_controls and self.config.scaling_method != "none":
            if self.config.scaling_method == "minmax":
                self.control_scaler = MinMaxScaler()
            elif self.config.scaling_method == "standard":
                self.control_scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {self.config.scaling_method}")

            print("Scaling controls using: ", self.config.scaling_method)
            ref_controls = data_splits[reference_split][1]
            
            # Determine reshape shape based on collective scaling config
            reshape_dim = 1 if self.config.scale_controls_collectively else ref_controls.shape[-1]
            
            self.control_scaler.fit(ref_controls.reshape(-1, reshape_dim))

            for split, (states, controls) in scaled_data["scaled"].items():
                controls_scaled = self.control_scaler.transform(controls.reshape(-1, reshape_dim)).reshape(controls.shape)
                scaled_data["scaled"][split] = (states, controls_scaled)
        
        return scaled_data

    def _apply_scaling_to_sequences(self, data: np.ndarray, scaler, fit: bool = False) -> np.ndarray:
        """Apply scaling to sequence data while preserving structure."""

        original_shape = data.shape
        ndim = data.ndim

        if ndim == 3:
            # 1D Simulation (N, T, D)
            if self.config.scale_states_collectively:
                # Scale collectively (all dimensions share statistics)
                flattened = data.reshape(-1, 1)
            else:
                # Scale independently (each dimension has own statistics)
                flattened = data.reshape(-1, original_shape[-1])
            
            if fit:
                scaled_flat = scaler.fit_transform(flattened)
            else:
                scaled_flat = scaler.transform(flattened)
            
            out = scaled_flat.reshape(original_shape)

        elif ndim == 5:
            # 2D Simulation (N, T, C, H, W): Scale per channel
            # Transpose to (N, T, H, W, C) to scale each channel C independently over the grid
            permuted = np.transpose(data, (0, 1, 3, 4, 2))
            permuted_shape = permuted.shape
            
            flattened = permuted.reshape(-1, permuted_shape[-1])

            if fit:
                scaled_flat = scaler.fit_transform(flattened)
            else:
                scaled_flat = scaler.transform(flattened)
            
            scaled_permuted = scaled_flat.reshape(permuted_shape)
            # Transpose back to (N, T, C, H, W)
            out = np.transpose(scaled_permuted, (0, 1, 4, 2, 3))

        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}. Expected 3 or 5.")

        return out
        
    def _create_sequences(self, data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create input-target sequences from scaled data."""
        
        sequences_splits = {}

        for split, (states, controls) in data_splits.items():
            if states.ndim ==3: # 1D simulation
                X, Y, U = self._create_1d_sequences(states, controls)
            else: # 2D simulation
                X, Y, U = self._create_2d_sequences(states, controls)

            sequences_splits[split] = (X, Y, U)

        return sequences_splits

    def _create_1d_sequences(self, states: np.ndarray, controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for 1D simulation data."""
        
        lookback = self.config.lookback
        pred_horizon = self.config.pred_horizon
        stride = self.config.stride

        n_simulations, n_timesteps, state_dim = states.shape
        n_samples = n_simulations * ((n_timesteps - lookback - pred_horizon) // stride + 1)

        X = np.zeros((n_samples, lookback, state_dim))
        Y = np.zeros((n_samples, pred_horizon, state_dim))
        U = np.zeros((n_samples, lookback + pred_horizon - 1, controls.shape[-1]))

        idx = 0
        for sim in range(n_simulations):
            for sample in range(0, n_timesteps - lookback - pred_horizon + 1, stride):
                X[idx] = states[sim, sample:sample + lookback]
                Y[idx] = states[sim, sample + lookback:sample + lookback + pred_horizon]
                U[idx] = controls[sim, sample:sample + lookback + pred_horizon - 1]
                idx += 1
        
        return X, Y, U
    
    def _create_2d_sequences(self, states: np.ndarray, controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for 2D simulation data."""
        
        lookback = self.config.lookback
        pred_horizon = self.config.pred_horizon
        stride = self.config.stride

        n_simulations, n_timesteps, C, H, W = states.shape
        n_samples = n_simulations * ((n_timesteps - lookback - pred_horizon) // stride + 1)

        X = np.zeros((n_samples, lookback, C, H, W))
        Y = np.zeros((n_samples, pred_horizon, C, H, W))
        U = np.zeros((n_samples, lookback + pred_horizon - 1, controls.shape[-1]))

        idx = 0
        for sim in range(n_simulations):
            for sample in range(0, n_timesteps - lookback - pred_horizon + 1, stride):
                X[idx] = states[sim, sample:sample + lookback]
                Y[idx] = states[sim, sample + lookback:sample + lookback + pred_horizon]
                U[idx] = controls[sim, sample:sample + lookback + pred_horizon - 1]
                idx += 1
        
        return X, Y, U
    
    def _create_datasets(self, sequence_splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, SequenceDataset]:
        """Create PyTorch datasets from sequences."""
        
        dataset = {}

        for split, (X, Y, U) in sequence_splits.items():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.config.device)
            U_tensor = torch.tensor(U, dtype=torch.float32).to(self.config.device)
            Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self.config.device)

            dataset[split] = SequenceDataset(X_tensor, U_tensor, Y_tensor)
        
        return dataset
    
    def _prepare_simulation_data(self, scaled_data: Dict[str, Any]) -> Dict[str, SimulationDataset]:
        """Prepare SimulationData objects for evaluation."""
        
        simulation_splits = {}

        for split in scaled_data["original"].keys():
            states_orig, controls_orig = scaled_data["original"][split]
            states_scaled, controls_scaled = scaled_data["scaled"][split]

            # convert to tensors
            states_orig_tensor = torch.tensor(states_orig, dtype=torch.float32).to(self.config.device)
            controls_orig_tensor = torch.tensor(controls_orig, dtype=torch.float32).to(self.config.device)
            states_scaled_tensor = torch.tensor(states_scaled, dtype=torch.float32).to(self.config.device)
            controls_scaled_tensor = torch.tensor(controls_scaled, dtype=torch.float32).to(self.config.device)

            simulation_splits[split] = SimulationDataset(states = states_orig_tensor,
                                                         controls = controls_orig_tensor,
                                                         states_scaled = states_scaled_tensor,
                                                         controls_scaled = controls_scaled_tensor)
            
        return simulation_splits
    
    def scale_trajectories(self, trajectories: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Scale new trajectories using the fitted scalers."""
        if not self._processed:
            raise ValueError("DataPreprocessor has not processed any data yet.")
        
        is_tensor = torch.is_tensor(trajectories)
        if is_tensor:
            device = trajectories.device
            dtype = trajectories.dtype
            data = trajectories.detach().cpu().numpy()
        else:
            data = trajectories

        scaled_states = data
        if self.state_scaler is not None:
            scaled_states = self._apply_scaling_to_sequences(data, self.state_scaler, fit=False)
        
        if is_tensor:
            return torch.tensor(scaled_states, device=device, dtype=dtype)
        return scaled_states
    
    def inverse_scale_trajectories(self, states: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse scale the states using the fitted state scaler."""
        if self.state_scaler is None:
            raise ValueError("State scaler has not been fitted yet.")
        
        is_tensor = torch.is_tensor(states)
        if is_tensor:
            device = states.device
            dtype = states.dtype
            data = states.detach().cpu().numpy()
        else:
            data = states
        
        original_shape = data.shape
        ndim = data.ndim

        if ndim == 3:
            # 1D Simulation (N, T, D)
            if self.config.scale_states_collectively:
                flattened = data.reshape(-1, 1)
            else:
                flattened = data.reshape(-1, data.shape[-1])
                
            inv_scaled_flat = self.state_scaler.inverse_transform(flattened)
            out = inv_scaled_flat.reshape(original_shape)

        elif ndim == 5:
            # 2D Simulation (N, T, C, H, W): Scale per channel
            permuted = np.transpose(data, (0, 1, 3, 4, 2))
            permuted_shape = permuted.shape
            
            flattened = permuted.reshape(-1, permuted_shape[-1])
            inv_scaled_flat = self.state_scaler.inverse_transform(flattened)
            inv_scaled_permuted = inv_scaled_flat.reshape(permuted_shape)
            out = np.transpose(inv_scaled_permuted, (0, 1, 4, 2, 3))

        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}. Expected 3 or 5.")

        if is_tensor:
            return torch.tensor(out, device=device, dtype=dtype)
        return out
    
    def scale_controls(self, controls: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Scale new controls using the fitted control scaler."""
        if not self._processed:
            raise ValueError("DataPreprocessor has not processed any data yet.")
        
        if self.control_scaler is None:
            return controls
        
        is_tensor = torch.is_tensor(controls)
        if is_tensor:
            device = controls.device
            dtype = controls.dtype
            data = controls.detach().cpu().numpy()
        else:
            data = controls
        
        reshape_dim = 1 if self.config.scale_controls_collectively else data.shape[-1]
        
        flattened = data.reshape(-1, reshape_dim)
        scaled_flat = self.control_scaler.transform(flattened)
        out = scaled_flat.reshape(data.shape)

        if is_tensor:
            return torch.tensor(out, device=device, dtype=dtype)
        return out
    
    def inverse_scale_controls(self, controls: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse scale the controls using the fitted control scaler."""
        if self.control_scaler is None:
            raise ValueError("Control scaler has not been fitted yet.")
        
        is_tensor = torch.is_tensor(controls)
        if is_tensor:
            device = controls.device
            dtype = controls.dtype
            data = controls.detach().cpu().numpy()
        else:
            data = controls
        
        reshape_dim = 1 if self.config.scale_controls_collectively else data.shape[-1]
        
        flattened = data.reshape(-1, reshape_dim)
        inv_scaled_flat = self.control_scaler.inverse_transform(flattened)
        out = inv_scaled_flat.reshape(data.shape)

        if is_tensor:
            return torch.tensor(out, device=device, dtype=dtype)
        return out


