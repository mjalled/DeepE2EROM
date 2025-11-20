import numpy as np
import torch
from .config import DataConfig
from .scalers import MinMaxScaler, StandardScaler
from .datasets import SequenceDataset, SimulationDataset
from typing import Dict, Optional, Tuple, Any

class DataPreprocessor:
    """Process raw simulations for training control-affine ROMs."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.state_scaler = None
        self.control_scaler = None
        self._processed = False

    def process_simualations(self,
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

        train_end = int(n_simulations * self.config.train_split)
        val_end = train_end + int(n_simulations * self.config.val_split)

        splits = {}
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
    
    def get_simulation(self, processed_data: Dict[str, Any], split: str, index: int, sim_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Retrieve a specific simulation by index from processed data."""
        
        if not self._processed:
            raise ValueError("DataPreprocessor has not been processed yet. Process simulations first.")

        simulations: Dict[str, SimulationDataset] = processed_data["simulations"]
        
        if split not in simulations:
            raise ValueError(f"Split '{split}' not found in processed data.")
        
        simulation_data = simulations[split]
        
        if index < 0 or index >= simulation_data.states.shape[0]:
            raise IndexError(f"Index {index} out of bounds for split '{split}' with {simulation_data.states.shape[0]} simulations.")
        
        # Get the full simulation
        states = simulation_data.states[index]
        controls = simulation_data.controls[index]
        states_scaled = simulation_data.states_scaled[index]
        controls_scaled = simulation_data.controls_scaled[index]
        
        # Truncate to specified length if provided
        if sim_len is not None:
            if sim_len <= 0:
                raise ValueError(f"sim_len must be positive, got {sim_len}")
            if sim_len > states.shape[0]:
                raise ValueError(f"sim_len {sim_len} exceeds simulation length {states.shape[0]}")
            
            states = states[:sim_len]
            controls = controls[:sim_len]
            states_scaled = states_scaled[:sim_len]
            controls_scaled = controls_scaled[:sim_len]
        
        return {
            "states": states,
            "controls": controls,
            "states_scaled": states_scaled,
            "controls_scaled": controls_scaled
        }
    
    def inverse_scale_states(self, states: np.ndarray) -> np.ndarray:
        """Inverse scale the states using the fitted state scaler."""
        if self.state_scaler is None:
            raise ValueError("State scaler has not been fitted yet.")
        
        original_shape = states.shape
        ndim = states.ndim

        if ndim == 3:
            # 1D Simulation (N, T, D)
            if self.config.scale_states_collectively:
                flattened = states.reshape(-1, 1)
            else:
                flattened = states.reshape(-1, states.shape[-1])
                
            inv_scaled_flat = self.state_scaler.inverse_transform(flattened)
            out = inv_scaled_flat.reshape(original_shape)

        elif ndim == 5:
            # 2D Simulation (N, T, C, H, W): Scale per channel
            permuted = np.transpose(states, (0, 1, 3, 4, 2))
            permuted_shape = permuted.shape
            
            flattened = permuted.reshape(-1, permuted_shape[-1])
            inv_scaled_flat = self.state_scaler.inverse_transform(flattened)
            inv_scaled_permuted = inv_scaled_flat.reshape(permuted_shape)
            out = np.transpose(inv_scaled_permuted, (0, 1, 4, 2, 3))

        return out
    
    def inverse_scale_controls(self, controls: np.ndarray) -> np.ndarray:
        """Inverse scale the controls using the fitted control scaler."""
        if self.control_scaler is None:
            raise ValueError("Control scaler has not been fitted yet.")
        
        reshape_dim = 1 if self.config.scale_controls_collectively else controls.shape[-1]
        
        flattened = controls.reshape(-1, reshape_dim)
        inv_scaled_flat = self.control_scaler.inverse_transform(flattened)
        out = inv_scaled_flat.reshape(controls.shape)

        return out


