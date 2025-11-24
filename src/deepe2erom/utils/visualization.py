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