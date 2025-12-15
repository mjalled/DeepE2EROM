import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from typing import Union, Optional, List, Tuple
import os

def animate_1Dsimulation(data: Union[np.ndarray, torch.Tensor],
                       x: Optional[Union[np.ndarray, torch.Tensor]] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[float, float] = (10, 5),
                       fps: int = 15,
                       title: str = "",
                       dpi: int = 100) -> None:
    
    """Animate 1D simulation data and save as GIF.

    Args:
        data: Array of shape (Nb_timesteps, N)
        save_path: Path to save the GIF file. If None, the animation is save in the local directory
        figsize: Figure size for the animation (width, height)
        fps: Frames per second for the animation
        title: Title of the animation
        dpi: Dots per inch for the saved animation
    """

    # Convert to numpy if input is a torch tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    n_timesteps, n_points = data.shape

    # Handle x values
    if x is None:
        x_vals = np.arange(n_points)
    elif isinstance(x, torch.Tensor):
        x_vals = x.detach().cpu().numpy()
    else:
        x_vals = x

    # create figure and axis
    fig, ax = plt.subplots(1,1,figsize=figsize)

    line, = ax.plot(x_vals, data[0], color='blue')
    ax.set_ylim(np.min(data), np.max(data))
    ax.set_title(f"{title} - Timestep: 0/{n_timesteps-1}")
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.grid()

    # Update function for animation
    def update(frame: int):
        line.set_ydata(data[frame])
        ax.set_title(f"{title} - Timestep: {frame}/{n_timesteps-1}")
        return [line]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, blit=True, interval=1000/fps)

    # Ensure save directory exists
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = os.path.join(os.getcwd(), "simulation_animation.gif")
    
    # Save animation as GIF
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)

def animate_compare_1Dsimulations(data1: Union[np.ndarray, torch.Tensor],
                                  data2: Union[np.ndarray, torch.Tensor],
                                  labels: Optional[List[str]] = None,
                                  x: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[float, float] = (10, 5),
                                  fps: int = 15,
                                  title: str = "",
                                  dpi: int = 100) -> None:
    
    """Animate two 1D simulation data arrays and save as GIF.

    Args:
        data1: First array of shape (Nb_timesteps, N)
        data2: Second array of shape (Nb_timesteps, N)
        labels: List of labels for the two datasets (default: ["Ground truth", "Prediction"])
        x: Optional x-values for plotting
        save_path: Path to save the GIF file. If None, the animation is saved in the local directory
        figsize: Figure size for the animation (width, height)
        fps: Frames per second for the animation
        title: Title of the animation
        dpi: Dots per inch for the saved animation
    """

    # Convert to numpy if input is a torch tensor
    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()

    n_timesteps, n_points = data1.shape
    if data2.shape != data1.shape:
        raise ValueError("data1 and data2 must have the same shape")

    # Handle labels
    if labels is None:
        labels = ["Ground truth", "Prediction"]
    elif len(labels) != 2:
        raise ValueError("labels must contain exactly 2 strings")

    # Handle x values
    if x is None:
        x_vals = np.arange(n_points)
    elif isinstance(x, torch.Tensor):
        x_vals = x.detach().cpu().numpy()
    else:
        x_vals = x

    # create figure and axis
    fig, ax = plt.subplots(1,1,figsize=figsize)

    line1, = ax.plot(x_vals, data1[0], color='blue', label=labels[0])
    line2, = ax.plot(x_vals, data2[0], color='red', linestyle='--', label=labels[1])
    ax.set_ylim(min(np.min(data1), np.min(data2)), max(np.max(data1), np.max(data2)))
    ax.set_title(f"{title} - Timestep: 0/{n_timesteps-1}")
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid()

    # Update function for animation
    def update(frame: int):
        line1.set_ydata(data1[frame])
        line2.set_ydata(data2[frame])
        ax.set_title(f"{title} - Timestep: {frame}/{n_timesteps-1}")
        return [line1, line2]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, blit=True, interval=1000/fps)

    # Ensure save directory exists
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = os.path.join(os.getcwd(), "simulation_comparison.gif")

    # Save animation as GIF
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Comparison animation saved to: {save_path}")


def animate_2Dsimulation(data: Union[np.ndarray, torch.Tensor],
                       save_path: Optional[str] = None,
                       channel: int = 0,
                       cmap: str = 'viridis',
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       figsize: Tuple[float, float] = (10, 5),
                       fps: int = 15,
                       title: str = "",
                       show_colorbar: bool = True,
                       dpi: int = 100) -> None:
    
    """Animate 2D simulation data and save as GIF.
    
    Args:
        data: Array of shape (Nb_timesteps, channels, H, W) or (Nb_timesteps, H, W)
        save_path: Path to save the GIF file. If None, the animation is save in the local directory
        channel: Channel index to visualize if data has multiple channels
        cmap: Colormap for visualization
        vmin: Minimum value for color scaling (if None, uses min of data)
        vmax: Maximum value for color scaling (if None, uses max of data)
        figsize: Figure size for the animation (width, height)
        fps: Frames per second for the animation
        title: Title of the animation
        show_colorbar: Whether to display a colorbar
        dpi: Dots per inch for the saved animation
    """

    # Convert to numpy if input is a torch tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Handle data shape
    if data.ndim == 3:
        frames = data
    elif data.ndim == 4:
        frames = data[:, channel, :, :]
    else:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")
    
    n_timesteps = len(frames)

    # create figure and axis
    fig, ax = plt.subplots(1,1,figsize=figsize)

    # set vmin and vmax if not provided
    if vmin is None:
        vmin = np.min(frames)
    if vmax is None:
        vmax = np.max(frames)

    # Create the initial frame
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} - Timestep: 0/{n_timesteps-1}")
    ax.axis('off')

    # Add colorbar if needed
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Intensity')
    
    # Update function for animation
    def update(frame: int):
        im.set_array(frames[frame])
        ax.set_title(f"{title} - Timestep: {frame}/{n_timesteps-1}")
        return [im]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, blit=True, interval=1000/fps)

    # Ensure save directory exists
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = os.path.join(os.getcwd(), "simulation_animation.gif")

    # Save animation as GIF
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Animation saved to: {save_path}")

def animate_compare_2Dsimulations(data1: Union[np.ndarray, torch.Tensor],
                                  data2: Union[np.ndarray, torch.Tensor],
                                  labels: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  channel: int = 0,
                                  cmap: str = 'viridis',
                                  vmin: Optional[float] = None,
                                  vmax: Optional[float] = None,
                                  figsize: Tuple[float, float] = (12, 5),
                                  fps: int = 15,
                                  show_colorbar: bool = True,
                                  dpi: int = 100) -> None:
    
    """Animate two 2D simulation data arrays and save as GIF with shared colorbar.

    Args:
        data1: First array of shape (Nb_timesteps, channels, H, W) or (Nb_timesteps, H, W)
        data2: Second array of shape (Nb_timesteps, channels, H, W) or (Nb_timesteps, H, W)
        labels: List of labels for the two datasets (default: ["Data 1", "Data 2"])
        save_path: Path to save the GIF file. If None, the animation is save in the local directory
        channel: Channel index to visualize if data has multiple channels
        cmap: Colormap for visualization
        vmin: Minimum value for color scaling (if None, uses min of both data)
        vmax: Maximum value for color scaling (if None, uses max of both data)
        figsize: Figure size for the animation (width, height)
        fps: Frames per second for the animation
        show_colorbar: Whether to display a shared colorbar
        dpi: Dots per inch for the saved animation
    """

    # Convert to numpy if input is a torch tensor
    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()

    # Handle data shape
    if data1.ndim == 3:
        frames1 = data1
    elif data1.ndim == 4:
        frames1 = data1[:, channel, :, :]
    else:
        raise ValueError(f"Expected 3D or 4D data for data1, got {data1.ndim}D")

    if data2.ndim == 3:
        frames2 = data2
    elif data2.ndim == 4:
        frames2 = data2[:, channel, :, :]
    else:
        raise ValueError(f"Expected 3D or 4D data for data2, got {data2.ndim}D")

    n_timesteps = len(frames1)
    if len(frames2) != n_timesteps:
        raise ValueError("data1 and data2 must have the same number of timesteps")

    # Handle labels
    if labels is None:
        labels = ["Ground truth", "Prediction"]
    elif len(labels) != 2:
        raise ValueError("labels must contain exactly 2 strings")

    # create figure and axis
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Ensure axs is always a numpy array
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    # set vmin and vmax if not provided
    if vmin is None:
        vmin = min(np.min(frames1), np.min(frames2))
    if vmax is None:
        vmax = max(np.max(frames1), np.max(frames2))

    # Create the initial frames
    im1 = axs[0].imshow(frames1[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"{labels[0]} - Timestep: 0/{n_timesteps-1}")
    axs[0].axis('off')

    im2 = axs[1].imshow(frames2[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"{labels[1]} - Timestep: 0/{n_timesteps-1}")
    axs[1].axis('off')

    # Add shared colorbar if needed
    if show_colorbar:
        cbar = fig.colorbar(im1, ax=axs, shrink=0.8, location='right')
        cbar.set_label('Intensity')

    # Update function for animation
    def update(frame: int):
        im1.set_array(frames1[frame])
        axs[0].set_title(f"{labels[0]} - Timestep: {frame}/{n_timesteps-1}")
        im2.set_array(frames2[frame])
        axs[1].set_title(f"{labels[1]} - Timestep: {frame}/{n_timesteps-1}")
        return [im1, im2]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, blit=True, interval=1000/fps)

    # Ensure save directory exists
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = os.path.join(os.getcwd(), "simulation_2D_comparison.gif")

    # Save animation as GIF
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Comparison 2D animation saved to: {save_path}")