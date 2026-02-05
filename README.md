# DeepE2EROM
DeepE2EROM is a PyTorch-based framework for building fully end-to-end, data-driven reduced order models (ROMs) using autoencoders with learnable latent dynamics. The framework supports general nonlinear ROMs and includes an implementation of control-affine latent dynamics for systems with control inputs as shown in the figure below.

![Control-affine ROM](https://github.com/mjalled/framework.png)

## Installation

To install `DeepE2EROM`, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mjalled/DeepE2EROM.git
   cd DeepE2EROM
   ```

2. **(Optional) Create a virtual environment:**
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venvName
   source venvName/bin/activate  # On Windows use `venvName\Scripts\activate`
   ```

3. **Install dependencies:**
   Install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   Install the local package.
   ```bash
   pip install .
   ```

## Structure

- **`deepe2erom/`** – Main Python package.
  - **`models/`** – Autoencoders and latent dynamics modules (including control-affine variants and general nonlinear dynamics).
  - **`data/`** (or equivalent loaders) – Utilities to build datasets from simulation snapshots / trajectories and to handle batching and normalization.
  - **`training/`** – Training and evaluation loops, loss definitions (reconstruction and prediction losses), and logging hooks.
  - **`utils/`** – Common utilities (e.g. I/O helpers, plotting).
- **`examples/`** – Example scripts or notebooks showing how to:
  - Prepare datasets from raw simulations,
  - Define an autoencoder + control-affine latent dynamics configuration,
  - Control using feedback linearization


