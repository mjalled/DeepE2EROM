# %% Import packages 
import numpy as np
import torch
from AE_Dyn import AE_DisAffine_seq, InputAffine
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pickle
import time
import os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties
import Utils
#from control.matlab import dlqr

# train on the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")

times_font_path = '/usr/share/fonts/truetype/times.ttf'
times_font = FontProperties(fname=times_font_path)
times_font_leg = FontProperties(fname=times_font_path)
times_font.set_size(8)
times_font_leg.set_size(8)

# Conversion factor
pt_to_inch = 1 / 72.27
# Text width and column width in points
text_width_pt = 469.75499
# Convert to inches
text_width_inch = text_width_pt * pt_to_inch

# %% Load the data and prepare the tensors

with open('../Data/Ball-in-Box.pkl', 'rb') as f:
    data = pickle.load(f)

# get each data type
[images_train, ux_train, uy_train] = data['Training']
[images_valid, ux_valid, uy_valid] = data['Validtion']
[images_test, ux_test, uy_test] = data['Testing']

# transform to numpy arrays
images_train = np.array(images_train)
u_train = np.concatenate((ux_train.reshape(-1,1), uy_train.reshape(-1,1)),axis=1)
Nb_train_samples = 5000
images_train = images_train[:Nb_train_samples]
u_train = u_train[:Nb_train_samples]

images_valid = np.array(images_valid)
u_valid = np.concatenate((ux_valid.reshape(-1,1), uy_valid.reshape(-1,1)),axis=1)
Nb_valid_samples = 1000
#Nb_valid_samples = 5000
images_valid = images_valid[:Nb_valid_samples]
u_valid = u_valid[:Nb_valid_samples]

images_test = np.array(images_test)
u_test = np.concatenate((ux_test.reshape(-1,1), uy_test.reshape(-1,1)),axis=1)

# scale the data
images_train_scaled, max_scaler, min_scaler = Utils.scaler_fit_transform(images_train)
images_valid_scaled = Utils.scaler_transform(images_valid, max_scaler, min_scaler)
images_test_scaled = Utils.scaler_transform(images_test, max_scaler, min_scaler)

# scale the input signals
scale_u = False
if scale_u:
    scaler_u = MinMaxScaler()
    u_train_scaled = scaler_u.fit_transform(u_train)
    u_valid_scaled = scaler_u.transform(u_valid)
    u_test_scaled = scaler_u.transform(u_test)
else:
    u_train_scaled = u_train
    u_valid_scaled = u_valid
    u_test_scaled = u_test

# %% Prepare the sequences for training and validation 

# prepare the data for training
seq_length = 5
X_train = np.zeros((Nb_train_samples-seq_length, seq_length, 1, images_train.shape[1], images_train.shape[2]))
Y_train = np.zeros((Nb_train_samples-seq_length, 1, images_train.shape[1], images_train.shape[2]))

U_train = np.zeros((Nb_train_samples-seq_length, seq_length, u_train.shape[1]))
for i in range(Nb_train_samples-seq_length):
    X_train[i] = images_train_scaled[i:i+seq_length].reshape(seq_length, 1, images_train.shape[1], images_train.shape[2])
    Y_train[i] = images_train_scaled[i+seq_length].reshape(1, images_train.shape[1], images_train.shape[2])
    U_train[i] = u_train_scaled[i:i+seq_length]
# convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32, requires_grad=True).to(device)
U_train = torch.tensor(U_train, dtype=torch.float32, requires_grad=True).to(device)

# prepare the data for validation
X_valid = np.zeros((Nb_valid_samples-seq_length, seq_length, 1, images_valid.shape[1], images_valid.shape[2]))
Y_valid = np.zeros((Nb_valid_samples-seq_length, 1, images_valid.shape[1], images_valid.shape[2]))
U_valid = np.zeros((Nb_valid_samples-seq_length, seq_length, u_valid.shape[1]))
for i in range(Nb_valid_samples-seq_length):
    X_valid[i] = images_valid_scaled[i:i+seq_length].reshape(seq_length, 1, images_valid.shape[1], images_valid.shape[2])
    Y_valid[i] = images_valid_scaled[i+seq_length].reshape(1, images_valid.shape[1], images_valid.shape[2])
    U_valid[i] = u_valid_scaled[i:i+seq_length]
# convert to torch tensors
X_valid = torch.tensor(X_valid, dtype=torch.float32, requires_grad=True).to(device)
Y_valid = torch.tensor(Y_valid, dtype=torch.float32, requires_grad=True).to(device)
U_valid = torch.tensor(U_valid, dtype=torch.float32, requires_grad=True).to(device)

# preapare the test data
Nb_test_samples = images_test_scaled.shape[0]
X_test = np.zeros((Nb_test_samples-seq_length, seq_length, 1, images_test.shape[1], images_test.shape[2]))
Y_test = np.zeros((Nb_test_samples-seq_length, 1, images_test.shape[1], images_test.shape[2]))
U_test = np.zeros((Nb_test_samples-seq_length, seq_length, u_test.shape[1]))
for i in range(Nb_test_samples-seq_length):
    X_test[i] = images_test_scaled[i:i+seq_length].reshape(seq_length, 1, images_test.shape[1], images_test.shape[2])
    Y_test[i] = images_test_scaled[i+seq_length].reshape(1, images_test.shape[1], images_test.shape[2])
    U_test[i] = u_test_scaled[i:i+seq_length]
# convert to torch tensors
X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32, requires_grad=True).to(device)
U_test = torch.tensor(U_test, dtype=torch.float32, requires_grad=True).to(device)

# %% Load the model

folder_name = 'Ball-in-Box_AEAffine'+str(seq_length)+'_Lat2_weight100'

# create the model
model = AE_DisAffine_seq(seq_length=seq_length, state_dim=64, latent_dim=2,
                        input_dim=2, dyn_nb_layers=4,
                        dyn_neurons=256, rec_weight=0.0,
                        pred_weight=0.0, latent_weight=0.0).to(device)

# load the model
model.load_state_dict(torch.load('../Models/' + folder_name + '/DynSeq_iter.pth'))

# %% Generate the reference trajectory and Control the input affine model using P controller 

latent_dim = 2
ref_timesteps = 100
encoded_ref = torch.ones((ref_timesteps, latent_dim)).to(device)
encoded_ref[:50,0] = 0.4
encoded_ref[:50,1] = 0.4
encoded_ref[50:,0] = 0.1
encoded_ref[50:,1] = 0.1

# visualize the reference trajectory 
frames_ref = model.decoder(encoded_ref)
plt.imshow(frames_ref[0,0].detach().cpu().numpy(), cmap='gray')
plt.imshow(frames_ref[50,0].detach().cpu().numpy(), cmap='gray')

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_P = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_P[:seq_length-1] = u0

# Proportional gain - P controller
n = model.latent_dim
Kp = 0.5 * torch.eye(n, device=device)

# loop over the timesteps
for t in range(seq_length, ref_timesteps):
    # build the augmented state/input vector
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)

    # evaluate the model
    a_x = model.drift_net(ksi).T                          
    B_x = model.input_net(ksi).T.view(n, model.input_dim)   

    # 2) The linearized system will be controlled with a P controller
    e_k = encoded_ref[t].view(n,1) - x0[-1].view(n,1)          
    v_k = Kp @ e_k                                      

    # 3) solve for u_k
    rank_B_x = torch.linalg.matrix_rank(B_x)
    if rank_B_x < n:
        print(f"Warning: B_x is not full rank at timestep {t}.")
        # Use pseudo-inverse to handle the case where B_x is not full rank
        B_x_inv = torch.linalg.pinv(B_x)
    else:
        # If B_x is full rank, use the inverse
        B_x_inv = torch.linalg.inv(B_x)
    # the ansatz is given as: z_k+1 = z_k + v_k
    u_k = B_x_inv @ ((x0[-1].view(n,1) + v_k) - a_x)
    u_max= 1.0
    u_k = torch.clamp(u_k, min=-u_max, max=u_max)

    # store u
    u_history_P[t] = u_k.T

    # 4) propagate one step
    next_state = a_x + B_x @ u_k
    x_history[t] = next_state.T

    # 5) roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)

# plot the results 
fig, axs = plt.subplots(2,1, figsize=(10,8))
axs[0].plot(x_history[:,0].detach().cpu().numpy(), label='z1', color='black')
axs[0].plot(x_history[:,1].detach().cpu().numpy(), label='z2', color='red')
axs[0].plot(encoded_ref[:,0].detach().cpu().numpy(), label='ref z1', linestyle='--', color='black')
axs[0].plot(encoded_ref[:,1].detach().cpu().numpy(), label='ref z2', linestyle='--', color='red')
axs[1].plot(u_history_P[:,0].detach().cpu().numpy(),label='u1')
axs[1].plot(u_history_P[:,1].detach().cpu().numpy(),label='u2')
axs[0].set_title('Latent space trajectory')
axs[1].set_title('Input trajectory')
axs[0].legend()
axs[1].legend()
plt.show()
fig.savefig('../Models/' + folder_name + '/P_controller_latent.pdf', bbox_inches='tight')

# decode the trajectory 
controlled_frames = model.decoder(x_history)

# plot the results
fig, axs = plt.subplots(figsize=(10,5))
def update(frame):
    axs.imshow(controlled_frames[frame, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs.set_title("Prediction")
    axs.axis('off')
ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=1)
plt.tight_layout()
writer = animation.PillowWriter(fps=15)
ani.save('../Models/' + folder_name + '/P_controller.gif', writer=writer)
plt.show()

# %% Evaluate multiple reference changes

latent_dim = 2
ref_timesteps = 100
encoded_ref = torch.ones((ref_timesteps, latent_dim)).to(device)
encoded_ref[:25,0] = 0.2
encoded_ref[:25,1] = 0.4
encoded_ref[25:50,0] = 0.6
encoded_ref[25:50,1] = 0.2
encoded_ref[50:75,0] = 0.4
encoded_ref[50:75,1] = 0.7
encoded_ref[75:,0] = 0.1
encoded_ref[75:,1] = 0.1

# visualize the reference trajectory 
frames_ref = model.decoder(encoded_ref)
#plt.imshow(frames_ref[0,0].detach().cpu().numpy(), cmap='gray')
#plt.imshow(frames_ref[50,0].detach().cpu().numpy(), cmap='gray')

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_P = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_P[:seq_length-1] = u0

# Proportional gain - P controller
n = model.latent_dim
Kp = 0.5 * torch.eye(n, device=device)

# loop over the timesteps
for t in range(seq_length, ref_timesteps):
    # build the augmented state/input vector
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)

    # evaluate the model
    a_x = model.drift_net(ksi).T                          
    B_x = model.input_net(ksi).T.view(n, model.input_dim)   

    # 2) The linearized system will be controlled with a P controller
    e_k = encoded_ref[t].view(n,1) - x0[-1].view(n,1)          
    v_k = Kp @ e_k                                      

    # 3) solve for u_k
    rank_B_x = torch.linalg.matrix_rank(B_x)
    if rank_B_x < n:
        print(f"Warning: B_x is not full rank at timestep {t}.")
        # Use pseudo-inverse to handle the case where B_x is not full rank
        B_x_inv = torch.linalg.pinv(B_x)
    else:
        # If B_x is full rank, use the inverse
        B_x_inv = torch.linalg.inv(B_x)
    # the ansatz is given as: z_k+1 = z_k + v_k
    u_k = B_x_inv @ ((x0[-1].view(n,1) + v_k) - a_x)
    u_max= 1.0
    u_k = torch.clamp(u_k, min=-u_max, max=u_max)

    # store u
    u_history_P[t] = u_k.T

    # 4) propagate one step
    next_state = a_x + B_x @ u_k
    x_history[t] = next_state.T

    # 5) roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)

# plot the results 
fig, axs = plt.subplots(2,1, figsize=(10,8))
axs[0].plot(x_history[:,0].detach().cpu().numpy(), label='z1', color='black')
axs[0].plot(x_history[:,1].detach().cpu().numpy(), label='z2', color='red')
axs[0].plot(encoded_ref[:,0].detach().cpu().numpy(), label='ref z1', linestyle='--', color='black')
axs[0].plot(encoded_ref[:,1].detach().cpu().numpy(), label='ref z2', linestyle='--', color='red')
axs[1].plot(u_history_P[:,0].detach().cpu().numpy(),label='u1')
axs[1].plot(u_history_P[:,1].detach().cpu().numpy(),label='u2')
axs[0].set_title('Latent space trajectory')
axs[1].set_title('Input trajectory')
axs[0].legend()
axs[1].legend()
plt.show()
fig.savefig('../Models/' + folder_name + '/P_controller_latent_multsteps.pdf', bbox_inches='tight')

# decode the trajectory 
controlled_frames = model.decoder(x_history)

# plot the results
fig, axs = plt.subplots(figsize=(10,5))
def update(frame):
    axs.imshow(controlled_frames[frame, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs.set_title("Prediction")
    axs.axis('off')
ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=1)
plt.tight_layout()
writer = animation.PillowWriter(fps=15)
ani.save('../Models/' + folder_name + '/P_controller_multsteps.gif', writer=writer)
plt.show()

# %% Evaluate with trajectory tracking

latent_dim = 2
ref_timesteps = 100

# Generate a Lissajous curve in [0,1]x[0,1]
t = torch.linspace(0, 1, ref_timesteps).to(device)
encoded_ref = torch.zeros((ref_timesteps, latent_dim), device=device)
encoded_ref[:, 0] = 0.5 + 0.4 * torch.sin(2 * torch.pi * t)         # z1
encoded_ref[:, 1] = 0.5 + 0.4 * torch.sin(4 * torch.pi * t + 0.5)   # z2

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_P = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_P[:seq_length-1] = u0

# Proportional gain - P controller
n = model.latent_dim
Kp = 0.5 * torch.eye(n, device=device)

# loop over the timesteps
for t in range(seq_length, ref_timesteps):
    # build the augmented state/input vector
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)

    # evaluate the model
    a_x = model.drift_net(ksi).T                          
    B_x = model.input_net(ksi).T.view(n, model.input_dim)   

    # 2) The linearized system will be controlled with a P controller
    e_k = encoded_ref[t].view(n,1) - x0[-1].view(n,1)          
    v_k = Kp @ e_k                                      

    # 3) solve for u_k
    rank_B_x = torch.linalg.matrix_rank(B_x)
    if rank_B_x < n:
        print(f"Warning: B_x is not full rank at timestep {t}.")
        # Use pseudo-inverse to handle the case where B_x is not full rank
        B_x_inv = torch.linalg.pinv(B_x)
    else:
        # If B_x is full rank, use the inverse
        B_x_inv = torch.linalg.inv(B_x)
    # the ansatz is given as: z_k+1 = z_k + v_k
    u_k = B_x_inv @ ((x0[-1].view(n,1) + v_k) - a_x)
    u_max= 1.0
    u_k = torch.clamp(u_k, min=-u_max, max=u_max)

    # store u
    u_history_P[t] = u_k.T

    # 4) propagate one step
    next_state = a_x + B_x @ u_k
    x_history[t] = next_state.T

    # 5) roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)

# plot the results 
fig, axs = plt.subplots(2,1, figsize=(10,8))
axs[0].plot(x_history[:,0].detach().cpu().numpy(), label='z1', color='black')
axs[0].plot(x_history[:,1].detach().cpu().numpy(), label='z2', color='red')
axs[0].plot(encoded_ref[:,0].detach().cpu().numpy(), label='ref z1', linestyle='--', color='black')
axs[0].plot(encoded_ref[:,1].detach().cpu().numpy(), label='ref z2', linestyle='--', color='red')
axs[1].plot(u_history_P[:,0].detach().cpu().numpy(),label='u1')
axs[1].plot(u_history_P[:,1].detach().cpu().numpy(),label='u2')
axs[0].set_title('Latent space trajectory')
axs[1].set_title('Input trajectory')
axs[0].legend()
axs[1].legend()
plt.show()

# decode the trajectory 
controlled_frames = model.decoder(x_history)

# plot the results
fig, axs = plt.subplots(figsize=(10,5))
def update(frame):
    axs.imshow(controlled_frames[frame, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs.set_title("Prediction")
    axs.axis('off')
ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=1)
plt.tight_layout()
writer = animation.PillowWriter(fps=15)
plt.show()

# %% Evaluate on a circular trajectory

# model parameters
beta = 1 / 200
gamma = 0.79
k = 1 / 4
r = 0.25  # Radius for pixel intensity calculation
time_step = 0.3  # Time step for RK4 integration
steps = 100  # Number of time steps
sigma_y = 0.204 # Std deviation of the noise


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

    ux = u_history_P[:,0].detach().cpu().numpy()
    uy = u_history_P[:,1].detach().cpu().numpy()

    images = []

    for t in range(num_frames):
        images.append(generate_image(px, py, noise_amplitude))
        px, py, vx, vy = rk4_step(px, py, vx, vy, ux[t], uy[t], time_step)

    return images

# New simulator that forces the center to move on a circle
def simulate_circle(num_frames,
                    cx=0.6, cy=0.4,            # circle center (off-center by default)
                    rcirc=None,                # circle radius (if None computed safely)
                    noise_amplitude=0.1,
                    start_angle=0.0,
                    omega=2*np.pi/100.0):      # angular speed (radians per frame)
    """
    Produce frames where the ball center follows a circle:
      px(t) = cx + rcirc * cos(theta(t))
      py(t) = cy + rcirc * sin(theta(t))
    Ensures the center stays at least `r` away from domain borders.
    """
    # safety margin: make sure the center of the ball always stays in [r, 1-r]
    # compute a safe max radius if rcirc not provided
    max_r_x = min(cx - r, 1 - r - cx)
    max_r_y = min(cy - r, 1 - r - cy)
    safe_max = max(0.0, min(max_r_x, max_r_y))
    if rcirc is None:
        # choose rcirc somewhat smaller than the safe max to not be "too close"
        rcirc = max(0.0, 0.8 * safe_max)

    if rcirc <= 0:
        raise ValueError("rcirc computed as <= 0; pick a center (cx,cy) further from boundaries "
                         "or reduce ball radius `r`.")

    images = []
    theta = start_angle
    for t in range(num_frames):
        px = cx + rcirc * np.cos(theta)
        py = cy + rcirc * np.sin(theta)
        images.append(generate_image(px, py, noise_amplitude))
        theta += omega  # advance angle by omega radians per frame
    return images

latent_dim = 2
ref_timesteps = 200

# Example usage:
frames = simulate_circle(num_frames=ref_timesteps, cx=0.6, cy=0.4, noise_amplitude=0.0,
                         start_angle=0.0, omega=4*np.pi/100.0)

# encode the frames and get the reference trajectory
frames = np.array(frames)
frames_scaled = Utils.scaler_transform(frames, max_scaler, min_scaler)
frames_scaled = torch.tensor(frames_scaled, dtype=torch.float32).to(device)
encoded_ref = model.encoder(frames_scaled.unsqueeze(1)).squeeze(1)

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_P = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_P[:seq_length-1] = u0

# Proportional gain - P controller
n = model.latent_dim
Kp = 0.5 * torch.eye(n, device=device)

# loop over the timesteps
for t in range(seq_length, ref_timesteps):
    # build the augmented state/input vector
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)

    # evaluate the model
    a_x = model.drift_net(ksi).T                          
    B_x = model.input_net(ksi).T.view(n, model.input_dim)   

    # 2) The linearized system will be controlled with a P controller
    e_k = encoded_ref[t].view(n,1) - x0[-1].view(n,1)          
    v_k = Kp @ e_k                                      

    # 3) solve for u_k
    rank_B_x = torch.linalg.matrix_rank(B_x)
    if rank_B_x < n:
        print(f"Warning: B_x is not full rank at timestep {t}.")
        # Use pseudo-inverse to handle the case where B_x is not full rank
        B_x_inv = torch.linalg.pinv(B_x)
    else:
        # If B_x is full rank, use the inverse
        B_x_inv = torch.linalg.inv(B_x)
    # the ansatz is given as: z_k+1 = z_k + v_k
    u_k = B_x_inv @ ((x0[-1].view(n,1) + v_k) - a_x)
    u_max= 1.0
    u_k = torch.clamp(u_k, min=-u_max, max=u_max)

    # store u
    u_history_P[t] = u_k.T

    # 4) propagate one step
    next_state = a_x + B_x @ u_k
    x_history[t] = next_state.T

    # 5) roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)

# plot the results
fig_width = text_width_inch
fig_height = text_width_inch
fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height-3))
axs[0].plot(x_history[:, 0].detach().cpu().numpy(), label='$z_1$', color='C0', linewidth=0.75)
axs[0].plot(x_history[:, 1].detach().cpu().numpy(), label='$z_2$', color='C1', linewidth=0.75)
axs[0].plot(encoded_ref[:, 0].detach().cpu().numpy(), label='$z^{ref}_1$', linestyle='--', color='C0',linewidth=0.75)
axs[0].plot(encoded_ref[:, 1].detach().cpu().numpy(), label='$z^{ref}_2$', linestyle='--', color='C1',linewidth=0.75)
axs[1].plot(u_history_P[:, 0].detach().cpu().numpy(), label='$u_1$', color='black', linewidth=0.75)
axs[1].plot(u_history_P[:, 1].detach().cpu().numpy(), label='$u_2$', color='salmon', linewidth=0.75)
axs[0].legend(prop=times_font_leg, loc='upper right')
axs[1].legend(prop=times_font_leg, loc='upper right')
# grid
axs[0].grid(color='lightgray', linestyle='--', linewidth=0.5)
axs[1].grid(color='lightgray', linestyle='--', linewidth=0.5)
# Set axis labels
axs[0].set_ylabel('z', fontproperties=times_font)
axs[1].set_ylabel('u', fontproperties=times_font)
axs[1].set_xlabel('time step', fontproperties=times_font)
# Only show x-axis on lower plot
axs[0].set_xticklabels([])
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
for ax in axs:
    for label in ax.get_yticklabels():
        label.set_fontproperties(times_font)
    for label in ax.get_xticklabels():
        label.set_fontproperties(times_font)
plt.show()
plt.tight_layout()
fig.savefig('../Models/' + folder_name + '/P_controller_latent_trajectory.pdf', bbox_inches='tight')

# decode the trajectory 
controlled_frames = model.decoder(x_history)

# plot the results
fig, axs = plt.subplots(figsize=(10,5))
def update(frame):
    axs.imshow(controlled_frames[frame, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs.set_title("Prediction")
    axs.axis('off')
ani = animation.FuncAnimation(fig, update, frames=np.arange(100), interval=1)
plt.tight_layout()
writer = animation.PillowWriter(fps=15)
ani.save('../Models/' + folder_name + '/P_controller_trajectory.gif', writer=writer)
plt.show()

# %% For comparison, contol the model without feedback linearization

latent_dim = 2
ref_timesteps = 100
encoded_ref = torch.ones((ref_timesteps, latent_dim)).to(device)
encoded_ref[:25,0] = 0.2
encoded_ref[:25,1] = 0.4
encoded_ref[25:50,0] = 0.6
encoded_ref[25:50,1] = 0.2
encoded_ref[50:75,0] = 0.4
encoded_ref[50:75,1] = 0.7
encoded_ref[75:,0] = 0.1
encoded_ref[75:,1] = 0.1

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_PID = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_PID[:seq_length-1] = u0

# PID gains
n = model.latent_dim
Kp = 1 * torch.eye(n, device=device)
Ki = 0.1 * torch.eye(n, device=device)
Kd = 0.1 * torch.eye(n, device=device)

# Initialize PID terms
integral_error = torch.zeros((n, 1), device=device)
prev_error = torch.zeros((n, 1), device=device)

# loop over the timesteps
for t in range(seq_length, ref_timesteps):
    # Current error
    e_k = encoded_ref[t].view(n,1) - x0[-1].view(n,1)
    integral_error += e_k
    derivative_error = e_k - prev_error

    # PID control law in latent space
    v_k = Kp @ e_k + Ki @ integral_error + Kd @ derivative_error

    # Use v_k directly as the control input (no feedback linearization)
    u_k = v_k
    u_max = 1.0
    u_k = torch.clamp(u_k, min=-u_max, max=u_max)

    # store u
    u_history_PID[t] = u_k.T

    # propagate one step using the nonlinear model
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)
    next_state = model.drift_net(ksi).T + model.input_net(ksi).T.view(n, model.input_dim) @ u_k
    x_history[t] = next_state.T

    # roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)
    prev_error = e_k

# plot the results 
fig, axs = plt.subplots(2,1, figsize=(10,8))
axs[0].plot(x_history[:,0].detach().cpu().numpy(), label='z1', color='black')
axs[0].plot(x_history[:,1].detach().cpu().numpy(), label='z2', color='red')
axs[0].plot(encoded_ref[:,0].detach().cpu().numpy(), label='ref z1', linestyle='--', color='black')
axs[0].plot(encoded_ref[:,1].detach().cpu().numpy(), label='ref z2', linestyle='--', color='red')
axs[1].plot(u_history_P[:,0].detach().cpu().numpy(),label='u1')
axs[1].plot(u_history_P[:,1].detach().cpu().numpy(),label='u2')
axs[0].set_title('Latent space trajectory')
axs[1].set_title('Input trajectory')
axs[0].legend()
axs[1].legend()
plt.show()

# %% Control the input affine model using LQR controller 

# history values
x_history = torch.zeros((ref_timesteps, model.latent_dim)).to(device)
u_history_lqr = torch.zeros((ref_timesteps, model.input_dim)).to(device)

# define the initial conditions 
x0 = model.encoder(X_test[0])
u0 = U_test[0,:-1,:]

# store the initial conditions
x_history[:seq_length] = x0
u_history_lqr[:seq_length-1] = u0

# dims
n = model.latent_dim
m = model.input_dim

# cost matrices
Q = torch.eye(n, device=device) * 1
R = torch.eye(m, device=device) * 1

# Simple fixed‐point iteration for P
P = Q.clone()
B = torch.eye(2).to(device)
for _ in range(100):
    # A = I, B = I
    BT_P = P  # since B^T P = P
    P_new = Q + P - P @ torch.linalg.inv(R + P) @ P
    if torch.norm(P_new - P) < 1e-6:
        break
    P = P_new

# steady‐state LQR gain
K_lqr = torch.linalg.inv(R + P) @ P  # shape (m,n)
#K_lqr_matlab, S, E = dlqr(torch.eye(n).numpy(), B.cpu().numpy(), Q.cpu().numpy(), R.cpu().numpy())

# now run your loop using K_lqr:
for t in range(seq_length, ref_timesteps-1):
    # build the augmented state/input vector
    ksi = torch.cat((x0.view(1,-1), u0.view(1,-1)), dim=1)

    # evaluate the model
    a_x = model.drift_net(ksi).T                          
    B_x = model.input_net(ksi).T.view(n, model.input_dim)

    # error in z‐space:
    z_current = x0[-1].view(n,1)
    z_ref = encoded_ref[t].view(n,1)
    e_k = z_current - z_ref

    # LQR “delta‐control”:
    v_k = -K_lqr @ e_k + (encoded_ref[t+1].view(n,1) - z_ref)         # shape (n,1)

    # 3) solve for u_k
    rank_B_x = torch.linalg.matrix_rank(B_x)
    if rank_B_x < n:
        print(f"Warning: B_x is not full rank at timestep {t}.")
        # Use pseudo-inverse to handle the case where B_x is not full rank
        B_x_inv = torch.linalg.pinv(B_x)
    else:
        # If B_x is full rank, use the inverse
        B_x_inv = torch.linalg.inv(B_x)

    u_k = B_x_inv @ ((x0[-1].view(n,1) + v_k) - a_x)  # shape (m,1)
    #u_max= 1.0
    #u_k = torch.clamp(u_k, min=-u_max, max=u_max)

     # store u
    u_history_lqr[t] = u_k.T

    # 4) propagate one step
    next_state = a_x + B_x @ u_k
    x_history[t] = next_state.T

    # 5) roll the histories forward
    x0 = torch.cat((x0[1:], next_state.T), dim=0)
    u0 = torch.cat((u0[1:], u_k.T), dim=0)

# plot the results 
fig, axs = plt.subplots(2,1, figsize=(10,8))
axs[0].plot(x_history[:-1,0].detach().cpu().numpy(), label='z1', color='black')
axs[0].plot(x_history[:-1,1].detach().cpu().numpy(), label='z2', color='red')
axs[0].plot(encoded_ref[:-1,0].detach().cpu().numpy(), label='ref z1', linestyle='--', color='black')
axs[0].plot(encoded_ref[:-1,1].detach().cpu().numpy(), label='ref z2', linestyle='--', color='red')
axs[1].plot(u_history_lqr[:-1,0].detach().cpu().numpy(),label='u1')
axs[1].plot(u_history_lqr[:-1,1].detach().cpu().numpy(),label='u2')
axs[0].set_title('Latent space trajectory')
axs[1].set_title('Input trajectory')
axs[0].legend()
axs[1].legend()
plt.show()

# %% Simulate the controlled trajectory on the real model 

simulation_P = simulate(100, noise_amplitude=0)

# Create and save the animation
fig, ax = plt.subplots()
def update(frame):
    ax.clear()
    ax.imshow(simulation_P[frame], cmap="gray", origin="lower", extent=[0, 1, 0, 1])
    ax.set_title(f"Time Step: {frame}")
anim = animation.FuncAnimation(fig, update, frames=steps, interval=50)
from IPython.display import HTML
HTML(anim.to_jshtml())

