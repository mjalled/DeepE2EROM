# Import modules
import os
import numpy as np
import pickle

# Define parameters
beta = 1 / 200
gamma = 0.79
k = 1 / 4
r = 0.25            # Radius for pixel intensity calculation
time_step = 0.3     # Time step for RK4 integration
steps = 100         # Number of time steps
sigma_y = 0.204     # Std deviation of the noise

# Define the ODEs for the ball's dynamics
def dynamics(px, py, vx, vy, ux, uy):
    ax = beta * ((1 / (px**2)) - (1 / (1 - px)**2)) - gamma * vx + k * ux
    ay = beta * ((1 / (py**2)) - (1 / (1 - py)**2)) - gamma * vy + k * uy
    return ax, ay

# Runge-Kutta 4th order method for integration
def rk4_step(px, py, vx, vy, ux, uy, dt):
    ax1, ay1 = dynamics(px, py, vx, vy, ux, uy)
    k1_vx, k1_vy = ax1 * dt, ay1 * dt
    k1_px, k1_py = vx * dt, vy * dt

    ax2, ay2 = dynamics(px + 0.5 * k1_px, py + 0.5 * k1_py, vx + 0.5 * k1_vx, vy + 0.5 * k1_vy, ux, uy)
    k2_vx, k2_vy = ax2 * dt, ay2 * dt
    k2_px, k2_py = (vx + 0.5 * k1_vx) * dt, (vy + 0.5 * k1_vy) * dt

    ax3, ay3 = dynamics(px + 0.5 * k2_px, py + 0.5 * k2_py, vx + 0.5 * k2_vx, vy + 0.5 * k2_vy, ux, uy)
    k3_vx, k3_vy = ax3 * dt, ay3 * dt
    k3_px, k3_py = (vx + 0.5 * k2_vx) * dt, (vy + 0.5 * k2_vy) * dt

    ax4, ay4 = dynamics(px + k3_px, py + k3_py, vx + k3_vx, vy + k3_vy, ux, uy)
    k4_vx, k4_vy = ax4 * dt, ay4 * dt
    k4_px, k4_py = (vx + k3_vx) * dt, (vy + k3_vy) * dt

    px_new = px + (k1_px + 2 * k2_px + 2 * k3_px + k4_px) / 6
    py_new = py + (k1_py + 2 * k2_py + 2 * k3_py + k4_py) / 6
    vx_new = vx + (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) / 6
    vy_new = vy + (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6

    return px_new, py_new, vx_new, vy_new

# Generate the video image based on pixel intensity
Nx=64
Ny=64
def generate_image(px, py, noise_amplitude):
    image = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            x, y = i/(Nx-1), j/(Ny-1)
            intensity = max(0, 1 - ((x - px)**2 + (y - py)**2) / r**2)
            image[i, j] = intensity + np.random.normal(0, (noise_amplitude*sigma_y)**2)
    return image

# Perform a single simulation
def simulate(num_frames, noise_amplitude=0.1):
    px, py = 0.5, 0.5  # Initial position
    vx, vy = 0.0, 0.0  # Initial velocity

    ux = np.random.uniform(-1, 1, num_frames)
    uy = np.random.uniform(-1, 1, num_frames)

    images = []

    for t in range(num_frames):
        images.append(generate_image(px, py, noise_amplitude))
        px, py, vx, vy = rk4_step(px, py, vx, vy, ux[t], uy[t], time_step)

    return np.array(images), ux, uy

# simulate autonomous dynamics until steady state
def simulate_autonomous(num_simulations, noise_amplitude=0.1):
    # random initial conditions
    px_in = np.random.uniform(0.25, 0.75, num_simulations)
    py_in = np.random.uniform(0.25, 0.75, num_simulations)
    vx_in = np.random.uniform(-0.25, 0.25, num_simulations)
    vy_in = np.random.uniform(-0.25, 0.25, num_simulations)

    steps = 100
    images = np.zeros((num_simulations, steps, 64, 64))

    for i in range(num_simulations):
        px, py = px_in[i], py_in[i]
        vx, vy = vx_in[i], vy_in[i]
        for t in range(steps):
            images[i, t] = generate_image(px, py, noise_amplitude)
            px, py, vx, vy = rk4_step(px, py, vx, vy, 0, 0, time_step)
    
    return images

if __name__ == "__main__":
    # Simulate
    images_train, ux_train, uy_train = simulate(10000, noise_amplitude=1)
    controls = np.array([ux_train, uy_train]).T

    # simulate autonomous dynamics 
    autonomous_dynamics = simulate_autonomous(500, noise_amplitude=1)

    # Save the generated data
    Data = {}
    Data['Forced'] = [images_train, controls]
    Data['Autonomous'] = autonomous_dynamics

    # save the data
    os.makedirs('./data/', exist_ok=True)
    with open('./data/ball_in_box_data.pkl', 'wb') as f:
        pickle.dump(Data, f)
