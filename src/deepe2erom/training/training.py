import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import copy
import pickle
from typing import Dict, Tuple, Union, Optional
from torch.utils.data import DataLoader

class Trainer:
    """
    Trainer module for DeepE2EROM models.
    Handles training loop, validation, restarts, and plotting.
    """
    def __init__(self, model, 
                 rec_weight: float = 1.0,
                 pred_weight: float = 1.0,
                 latent_weight: float = 1.0,
                 input_rec_weight: float = 1.0,
                 learning_rate: float = 1e-3,
                 restarts_ae: int = 1,
                 restarts_e2e: int = 1,
                 device: Optional[torch.device] = None):
        self.model = model
        self.rec_weight = rec_weight
        self.pred_weight = pred_weight
        self.latent_weight = latent_weight
        self.input_rec_weight = input_rec_weight
        self.learning_rate = learning_rate
        self.restarts_ae = restarts_ae
        self.restarts_e2e = restarts_e2e
        
        # Determine device
        if device is not None:
            self.device = device
        elif hasattr(model, 'config') and model.config.device is not None:
            self.device = model.config.device
        else:
            self.device = torch.device('cpu')
            
        self.model.to(self.device)
        self.mse_loss = nn.MSELoss()

        # initial AEs - capture current state (e.g. pretrained weights)
        self.initial_aes = {
            'encoder': copy.deepcopy(self.model.encoder.state_dict()),
            'decoder': copy.deepcopy(self.model.decoder.state_dict()),
        }
        if self.model.control_encoder is not None:
            self.initial_aes['control_encoder'] = copy.deepcopy(self.model.control_encoder.state_dict())
            self.initial_aes['control_decoder'] = copy.deepcopy(self.model.control_decoder.state_dict())

    def loss_function(self, outputs: Tuple, targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss function based on model outputs and targets.

        Args:
            outputs: Outputs from forward pass
            targets: Dictionary containing:
                - 'input_states': Original input states
                - 'target_states': Future states to predict
                - ['input_controls']: Original input controls (if using input autoencoder)
                
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
        """
        loss_dict = {}

        # If input control reconstruction exists
        if self.model.control_encoder is not None:
            decoded_sequence, decoded_input_sequence, pred_latents, pred_states = outputs
            input_controls = targets['input_controls']

            # Standard MSE without batch-variance scaling for stability
            input_rec_loss = self.mse_loss(decoded_input_sequence, input_controls)
            loss_dict['input_rec_loss'] = input_rec_loss.item()
        else:
            decoded_sequence, pred_latents, pred_states = outputs
            input_rec_loss = torch.tensor(0.0, device=self.device)

        # State reconstruction loss
        input_states = targets['input_states']
        rec_loss = self.mse_loss(decoded_sequence, input_states)
        loss_dict['rec_loss'] = rec_loss.item()

        # Prediction loss
        target_states = targets['target_states']
        pred_loss = self.mse_loss(pred_states, target_states)
        loss_dict['pred_loss'] = pred_loss.item()

        # Latent loss
        target_states_flat = target_states.reshape(-1, *target_states.shape[2:])
        
        # Detach target latents to prevent encoder from collapsing to match predictions (! Dangerous with Stochastic encoders)
        with torch.no_grad():
            true_latents_flat = self.model.encoder(target_states_flat)
            true_latents = true_latents_flat.reshape_as(pred_latents)
            
        latent_loss = self.mse_loss(pred_latents, true_latents)
        loss_dict['latent_loss'] = latent_loss.item()

        # Combine with weights
        total_loss = (self.rec_weight * rec_loss +
                    self.pred_weight * pred_loss +
                    self.latent_weight * latent_loss)

        if self.model.control_encoder is not None:
            total_loss += self.input_rec_weight * input_rec_loss

        return total_loss, loss_dict

    def _process_batch(self, batch):
        """Unpack batch and move to device."""
        # Assuming batch is (input_state, input_control, target_state)
        x, u, y = batch
        x = x.to(self.device)
        u = u.to(self.device)
        y = y.to(self.device)
        
        targets = {
            'input_states': x,
            'target_states': y,
            'input_controls': u 
        }
        return x, u, targets

    def _weight_reset(self, m):
        """Reset model weights to initial state."""
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    def train_autoencoders(self, train_loader: DataLoader, epochs: int, ae_optimizer: torch.optim.Optimizer, control_ae_optimizer: Optional[torch.optim.Optimizer] = None, valid_data: Optional[Union[DataLoader, Tuple[torch.Tensor, ...]]] = None, seed: Optional[int] = None):
        """Pre-train the autoencoder."""

        best_ae_states_val_loss = float('inf')
        for restart in range(self.restarts_ae):
            if seed is not None:
                torch.manual_seed(seed + restart)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + restart)

            print(f"  AE Restart {restart + 1}/{self.restarts_ae}")
            print(f"    Training States Autoencoder for {epochs} epochs...")
            
            # Only reset weights for restarts > 0 to preserve pretrained weights on first run
            if restart > 0:
                self.model.encoder.apply(self._weight_reset)
                self.model.decoder.apply(self._weight_reset)
                # Clear optimizer state to ensure fresh start
                ae_optimizer.state.clear()
                
            for epoch in range(epochs):
                self.model.train()
                total_ae_loss = 0
                for batch in train_loader:
                    x, u, targets = self._process_batch(batch)
                    ae_optimizer.zero_grad()
                    
                    # State AE
                    decoded_x = self.model.forward_autoencoder(x)
                    ae_loss = self.mse_loss(decoded_x, x)
                    ae_loss.backward()
                    ae_optimizer.step()
                    total_ae_loss += ae_loss.item()
                    
                #avg_ae_loss = total_ae_loss / len(train_loader)

                # Validate after each epoch
                if valid_data is not None:
                    _, components = self._validate_autoencoders(valid_data)
                    # extract state and control AE losses
                    ae_val_loss = components['rec_loss']
                    if ae_val_loss < best_ae_states_val_loss:
                        print(f"Epoch {epoch+1} - New best state AE validation loss: {ae_val_loss:.6f}")
                        best_ae_states_val_loss = ae_val_loss
                        self.initial_aes.update({
                            'encoder': copy.deepcopy(self.model.encoder.state_dict()),
                            'decoder': copy.deepcopy(self.model.decoder.state_dict()),
                        })
            
        if self.model.control_encoder is not None and control_ae_optimizer is not None:
            best_ae_control_val_loss = float('inf')
            for restart in range(self.restarts_ae):
                if seed is not None:
                    torch.manual_seed(seed + 1000 + restart)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + 1000 + restart)

                print(f"\n  AE Restart {restart + 1}/{self.restarts_ae}")
                print(f"    Training Control Autoencoder for {epochs} epochs...")
                
                # Only reset weights for restarts > 0
                if restart > 0:
                    self.model.control_encoder.apply(self._weight_reset)
                    self.model.control_decoder.apply(self._weight_reset)
                    # Clear optimizer state
                    control_ae_optimizer.state.clear()
                    
                for epoch in range(epochs):
                    self.model.train()
                    total_control_ae_loss = 0
                    for batch in train_loader:
                        x, u, targets = self._process_batch(batch)
            
                        # Control AE (if exists)
                        decoded_u = self.model.forward_control_autoencoder(u)
                        control_ae_optimizer.zero_grad()
                        control_ae_loss = self.mse_loss(decoded_u, u)
                        control_ae_loss.backward()
                        control_ae_optimizer.step()
                        total_control_ae_loss += control_ae_loss.item()

                    #avg_control_ae_loss = total_control_ae_loss / len(train_loader)

                    # Validate after each epoch
                    if valid_data is not None:
                        _, components = self._validate_autoencoders(valid_data)
                        # extract control AE loss
                        control_ae_val_loss = components['input_rec_loss']
                        if control_ae_val_loss < best_ae_control_val_loss:
                            print(f"Epoch {epoch+1} - New best control AE validation loss: {control_ae_val_loss:.6f}")
                            best_ae_control_val_loss = control_ae_val_loss
                            if self.initial_aes is None:
                                self.initial_aes = {}
                            self.initial_aes.update({
                                'control_encoder': copy.deepcopy(self.model.control_encoder.state_dict()),
                                'control_decoder': copy.deepcopy(self.model.control_decoder.state_dict()),
                            })
        
    def validate(self, valid_data: Union[DataLoader, Tuple[torch.Tensor, ...]]) -> Tuple[float, Dict[str, float]]:
        """Compute validation loss and components."""
        self.model.eval()
        total_val_loss = 0
        total_val_components = {}
        
        if isinstance(valid_data, DataLoader):
            num_batches = len(valid_data)
            with torch.no_grad():
                for batch in valid_data:
                    x, u, targets = self._process_batch(batch)
                    outputs = self.model(x, u)
                    loss, loss_dict = self.loss_function(outputs, targets)
                    total_val_loss += loss.item()
                    for k, v in loss_dict.items():
                        total_val_components[k] = total_val_components.get(k, 0) + v
            
            avg_loss = total_val_loss / num_batches
            avg_components = {k: v / num_batches for k, v in total_val_components.items()}
        else:
            # Tensors (x_val, u_val, y_val)
            x_val, u_val, y_val = valid_data
            x = x_val.to(self.device)
            u = u_val.to(self.device)
            y = y_val.to(self.device)
            targets = {'input_states': x, 'target_states': y, 'input_controls': u}
            
            with torch.no_grad():
                outputs = self.model(x, u)
                loss, loss_dict = self.loss_function(outputs, targets)
            avg_loss = loss.item()
            avg_components = loss_dict
            
        self.model.train()
        return avg_loss, avg_components
    
    def _validate_autoencoders(self, valid_data: Union[DataLoader, Tuple[torch.Tensor, ...]]) -> Tuple[float, Dict[str, float]]:
        """Compute validation loss for autoencoders only."""
        self.model.eval()
        total_ae_loss = 0
        total_control_ae_loss = 0
        avg_ae_components = {}
        
        if isinstance(valid_data, DataLoader):
            num_batches = len(valid_data)
            with torch.no_grad():
                for batch in valid_data:
                    x, u, targets = self._process_batch(batch)
                    
                    # State AE
                    decoded_x = self.model.forward_autoencoder(x)
                    ae_loss = self.mse_loss(decoded_x, x)
                    total_ae_loss += ae_loss.item()
                    
                    # Control AE (if exists)
                    if self.model.control_encoder is not None:
                        decoded_u = self.model.forward_control_autoencoder(u)
                        control_ae_loss = self.mse_loss(decoded_u, u)
                        total_control_ae_loss += control_ae_loss.item()
            
            avg_loss = total_ae_loss / num_batches
            avg_ae_components['rec_loss'] = avg_loss
            if self.model.control_encoder is not None:
                avg_control_ae_loss = total_control_ae_loss / num_batches
                avg_ae_components['input_rec_loss'] = avg_control_ae_loss

        else:
            # Tensors (x_val, u_val, y_val)
            x_val, u_val, _ = valid_data
            x = x_val.to(self.device)
            u = u_val.to(self.device)
            targets = {'input_states': x, 'input_controls': u}
            
            with torch.no_grad():
                # State AE
                decoded_x = self.model.forward_autoencoder(x)
                ae_loss = self.mse_loss(decoded_x, x)
                
                avg_ae_components['rec_loss'] = ae_loss.item()
                
                # Control AE (if exists)
                if self.model.control_encoder is not None:
                    decoded_u = self.model.forward_control_autoencoder(u)
                    control_ae_loss = self.mse_loss(decoded_u, u)
                    avg_ae_components['input_rec_loss'] = control_ae_loss.item()
                
                avg_loss = ae_loss.item()
                
        self.model.train()
        return avg_loss, avg_ae_components

    def train(self, train_loader: DataLoader, 
              valid_data: Union[DataLoader, Tuple[torch.Tensor, ...]],
              epochs: int = 100,
              ae_epochs: int = 5,
              patience_early_stopping: int = 10,
              patience_lr_scheduler: int = 5,
              save_path: str = "training_results",
              seed: Optional[int] = None) -> float:
        """
        Main training loop.
        """
        os.makedirs(save_path, exist_ok=True)
        
        best_overall_loss = float('inf')
        best_overall_model_state = None

        # Phase 1: AE Training
        if ae_epochs > 0:
            # Optimizers for autoencoders only
            ae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
            ae_optimizer = optim.Adam(ae_params, lr=self.learning_rate)
            
            control_ae_optimizer = None
            if self.model.control_encoder is not None and self.model.control_decoder is not None:
                control_ae_params = list(self.model.control_encoder.parameters()) + list(self.model.control_decoder.parameters())
                control_ae_optimizer = optim.Adam(control_ae_params, lr=self.learning_rate)
            
            # Train AEs
            self.train_autoencoders(train_loader, ae_epochs, ae_optimizer, control_ae_optimizer, valid_data, seed=seed)

        for run in range(self.restarts_e2e):
            if seed is not None:
                torch.manual_seed(seed + 2000 + run)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 2000 + run)

            print(f"\nStarting run {run + 1}/{self.restarts_e2e}")
            
            # Reset model weights
            self.model.apply(self._weight_reset)
            
            # Reload best trained autoencoder weights
            self.model.encoder.load_state_dict(self.initial_aes['encoder'])
            self.model.decoder.load_state_dict(self.initial_aes['decoder'])
            if 'control_encoder' in self.initial_aes and self.model.control_encoder is not None and self.model.control_decoder is not None:
                self.model.control_encoder.load_state_dict(self.initial_aes['control_encoder'])
                self.model.control_decoder.load_state_dict(self.initial_aes['control_decoder'])

            # evaluate initial validation loss
            #_, components = self.validate(valid_data)
            #print(f"  Initial validation loss before full training: {components['rec_loss']:.6f}")
            
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_lr_scheduler, min_lr=1e-6, verbose=True)
            
            # Phase 2: Full Training
            history = {
                'avg_epoch_total_loss_train': [],
                'avg_epoch_rec_loss_train': [],
                'avg_epoch_rec_in_loss_train': [],
                'avg_epoch_pred_loss_train': [],
                'avg_epoch_lat_loss_train': [],
                'epoch_total_loss_valid': [],
                'epoch_rec_loss_valid': [],
                'epoch_rec_in_loss_valid': [],
                'epoch_pred_loss_valid': [],
                'epoch_lat_loss_valid': []
            }
            
            best_run_loss = float('inf')
            patience_counter = 0
            best_run_state = None
            is_best_run = False
            
            print(f"Starting full training for {epochs} epochs...")
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                train_components = {}
                
                for batch in train_loader:
                    x, u, targets = self._process_batch(batch)
                    optimizer.zero_grad()
                    outputs = self.model(x, u)
                    loss, loss_dict = self.loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    
                    for k, v in loss_dict.items():
                        train_components[k] = train_components.get(k, 0) + v
                
                avg_train_loss = train_loss / len(train_loader)
                avg_train_components = {k: v / len(train_loader) for k, v in train_components.items()}
                
                avg_val_loss, avg_val_components = self.validate(valid_data)
                
                scheduler.step(avg_val_loss)
                
                # Update history
                history['avg_epoch_total_loss_train'].append(avg_train_loss)
                history['avg_epoch_rec_loss_train'].append(avg_train_components.get('rec_loss', 0))
                history['avg_epoch_rec_in_loss_train'].append(avg_train_components.get('input_rec_loss', 0))
                history['avg_epoch_pred_loss_train'].append(avg_train_components.get('pred_loss', 0))
                history['avg_epoch_lat_loss_train'].append(avg_train_components.get('latent_loss', 0))
                
                history['epoch_total_loss_valid'].append(avg_val_loss)
                history['epoch_rec_loss_valid'].append(avg_val_components.get('rec_loss', 0))
                history['epoch_rec_in_loss_valid'].append(avg_val_components.get('input_rec_loss', 0))
                history['epoch_pred_loss_valid'].append(avg_val_components.get('pred_loss', 0))
                history['epoch_lat_loss_valid'].append(avg_val_components.get('latent_loss', 0))
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_run_loss:
                    best_run_loss = avg_val_loss
                    best_run_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    
                    # Check if this is the best overall model
                    if best_run_loss < best_overall_loss:
                        best_overall_loss = best_run_loss
                        best_overall_model_state = best_run_state
                        is_best_run = True
                        print(f"New best model found with validation loss: {best_overall_loss:.6f}")
                        torch.save(best_overall_model_state, os.path.join(save_path, "best_model.pth"))
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience_early_stopping:
                    print("Early stopping triggered.")
                    break
            
            # If this run is currently the best run, save its history and plots
            if is_best_run:
                with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
                    pickle.dump(history, f)
                
                # Plot the learning curves
                fig, axs = plt.subplots()
                axs.set_yscale('log')
                axs.set_xlabel('Epoch')
                axs.set_ylabel('MSE loss')
                axs.plot(history['avg_epoch_total_loss_train'], color='blue', linestyle='-', label='total train loss')
                axs.plot(history['avg_epoch_rec_loss_train'], color='orange', linestyle='-', label='rec train loss')
                if self.model.control_encoder is not None:
                    axs.plot(history['avg_epoch_rec_in_loss_train'], color='red', linestyle='-', label='rec in train loss')
                axs.plot(history['avg_epoch_pred_loss_train'], color='green', linestyle='-', label='pred train loss')
                axs.plot(history['avg_epoch_lat_loss_train'], color='black', linestyle='-', label='lat train loss')
                
                axs.plot(history['epoch_total_loss_valid'], color='blue', linestyle='--', label='total valid loss')
                axs.plot(history['epoch_rec_loss_valid'], color='orange', linestyle='--', label='rec valid loss')
                if self.model.control_encoder is not None:
                    axs.plot(history['epoch_rec_in_loss_valid'], color='red', linestyle='--', label='rec in valid loss')
                axs.plot(history['epoch_pred_loss_valid'], color='green', linestyle='--', label='pred valid loss')
                axs.plot(history['epoch_lat_loss_valid'], color='black', linestyle='--', label='lat valid loss')
                axs.legend()
                fig.savefig(os.path.join(save_path, 'learning_curve.pdf'))
                plt.close(fig)

        # Load best model into current model instance
        if best_overall_model_state is not None:
            self.model.load_state_dict(best_overall_model_state)
            print(f"Training completed. Best validation loss: {best_overall_loss:.6f}")
        
        return best_overall_loss
    
    def evaluate(self, valid_data: Union[DataLoader, Tuple[torch.Tensor, ...]]) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on validation data."""
        return self.validate(valid_data)
