# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import pickle

# Define functions for simulation
def simulate_heat_equation(
    L=1.0,           # Length of the spatial domain [0, L]
    T=0.5,           # Total time to simulate
    alpha=0.01,      # Thermal diffusivity
    Nx=101,          # Number of spatial grid points
    stability_r=0.4, # Stability constant (must be <= 0.5)
    initial_condition_func=None, # Function u(x, 0)
    input_source_func=None,      # Function f(x, t)
    bc_left=0.0,     # Boundary condition u(0, t)
    bc_right=0.0     # Boundary condition u(L, t)
):
    """
    Solves the 1D heat equation using the FTCS finite difference method.
    
    Returns:
    - U_history (np.ndarray): Array of shape (Nt+1, Nx) holding the temperature field
                             at every time step.
    - F_history (np.ndarray): Array of shape (Nt+1, Nx) holding the input field
                             at every time step.
    - x (np.ndarray): The spatial grid.
    - t (np.ndarray): The time grid.
    """
    
    # --- 1. Setup Grid ---
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    
    # Calculate dt based on stability
    dt = stability_r * dx**2 / alpha
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)
    
    # --- 2. Initialize Arrays ---
    # U_history[j, i] will be u(t_j, x_i)
    U_history = np.zeros((Nt + 1, Nx))
    F_history = np.zeros((Nt + 1, Nx))
    
    # --- 3. Set Initial Condition (IC) ---
    if initial_condition_func:
        U_history[0, :] = initial_condition_func(x)
    # Ensure IC respects boundary conditions
    U_history[0, 0] = bc_left
    U_history[0, -1] = bc_right
    
    # --- 4. Prepare Input Source (f(x,t)) ---
    # We pre-calculate the input for all x and t for efficiency
    if input_source_func:
        for j in range(Nt + 1):
            F_history[j, :] = input_source_func(x, t[j])
            
    # --- 5. Time-Stepping Loop ---
    u_now = U_history[0, :].copy() # Current time step (j)
    
    for j in range(Nt):
        # Get the input for the *current* time step
        f_now = F_history[j, :] 
        
        # Calculate the next time step (j+1)
        u_next = np.zeros(Nx)
        
        # Interior points
        u_next[1:-1] = u_now[1:-1] + stability_r * (
            u_now[2:] - 2 * u_now[1:-1] + u_now[:-2]
        ) + dt * f_now[1:-1]
        
        # Apply Boundary Conditions (BCs)
        u_next[0] = bc_left
        u_next[-1] = bc_right
        
        # Store the result and update for next iteration
        U_history[j+1, :] = u_next
        u_now = u_next.copy()
        
    return U_history, F_history, x, t

# Define an example initial condition: a sine wave
def example_ic(x):
    return np.sin(np.pi * x)

# Generate a database of simulations with varying inputs
def create_random_input_source_func(T_end):
    """
    Creates a function f(x, t) with randomized parameters.
    This function *returns another function*.
    """
    
    # Random parameters for a time-varying Gaussian bump
    num_bumps = np.random.randint(1, 3)
    
    params = []
    for _ in range(num_bumps):
        param_set = {
            'amplitude': np.random.uniform(5.0, 20.0),
            'center': np.random.uniform(0.2, 0.8),
            'width': np.random.uniform(0.01, 0.1),
            't_start': np.random.uniform(0.0, 0.3),
            't_end': np.random.uniform(0.3, T_end),
        }
        params.append(param_set)

    def generated_input_func(x_grid, t_value, t_grid_full=None):
        """The actual f(x, t) function that will be called by the simulator."""
        f_total = np.zeros_like(x_grid)
        for p in params:
            spatial_profile = p['amplitude'] * np.exp(
                -((x_grid - p['center'])**2) / p['width']
            )
            if p['t_start'] <= t_value <= p['t_end']:
                f_total += spatial_profile
        return f_total
        
    return generated_input_func

def generate_database(num_simulations,
                      L=1.0,
                      T=0.5,
                      Nx=101,
                      alpha=0.1):
    """
    Runs multiple simulations with random ICs and Inputs,
    and saves the data to a compressed .npz file.
    """
    all_U_histories = []
    all_F_histories = []

    print(f"Generating {num_simulations} simulations...")
    for _ in range(num_simulations):
        # 1. Create random functions for this run
        input_func_generator = create_random_input_source_func(T_end=T)
        
        # 2. Run simulation
        U, F, _, _ = simulate_heat_equation(
            L=L, T=T, Nx=Nx,
            initial_condition_func=example_ic,
            input_source_func=input_func_generator,
            alpha=alpha
        )
        
        # 3. Store results
        all_U_histories.append(U)
        all_F_histories.append(F)

    return np.array(all_U_histories), np.array(all_F_histories)

all_T_histories, all_U_histories = generate_database(num_simulations=2000)

# reduce temporal resolution
T_reduced = all_T_histories[:,::5, :]
U_reduced = all_U_histories[:,::5, :]

# save to dictionary
Data = {}
Data['all_T_histories'] = T_reduced
Data['all_U_histories'] = U_reduced

# save the data
os.makedirs('./data/', exist_ok=True)
with open('./data/heat_equation_data.pkl', 'wb') as f:
    pickle.dump(Data, f)

