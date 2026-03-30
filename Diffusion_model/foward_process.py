import torch
import matplotlib.pyplot as plt
import numpy as np

def cosine_schedule(num_timesteps, s=0.008):
  def f(t):
    return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
  x = torch.linspace(0, num_timesteps, num_timesteps + 1)
  alphas_cumprod = f(x) / f(torch.tensor([0]))
  betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  betas = torch.clip(betas,  0.0001, 0.02)
  return betas

def sample_by_t(tensor_to_sample, timesteps, x_shape, device='cpu'):
  batch_size = timesteps.shape[0]
  tensor_to_sample= tensor_to_sample.to(device)
  sampled_tensor = tensor_to_sample.gather(-1, timesteps.to(device))
  sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
  return sampled_tensor.to(timesteps.device)



def sample_q(x0, t, schedule, noise=None, gamma=2, noise_perturb=0, add_random_noise=False):
    if noise is None:
        noise = torch.randn_like(x0)

    ab_t  = schedule.alphas_bar[t].view(-1, 1, 1)
    sq    = torch.sqrt(ab_t)
    sq1m  = torch.sqrt(1. - ab_t)

    if noise_perturb > 0:
        sq1m = sq1m * (1.0 + noise_perturb * torch.randn_like(sq1m))

    x_t = sq * x0 + sq1m * noise

    if add_random_noise:
        x_t += torch.randn_like(x_t) * torch.sqrt(1. - ab_t)

    snr_t = ab_t / (1. - ab_t + 1e-20)
    w_t   = torch.minimum(
        torch.tensor(gamma, device=x0.device, dtype=x0.dtype) / snr_t,
        torch.tensor(1.0,   device=x0.device, dtype=x0.dtype)
    )

    return x_t, w_t

# def sample_q(x0, t, noise=None, gamma=2, noise_perturb=0, add_random_noise=False):
#     """
#     Hybrid noise schedule with controlled randomness.
#     """
#     device = x0.device  # Use the device of the input tensor
    
#     if noise is None:
#         noise = torch.randn_like(x0, device=device)  # Standard Gaussian noise on correct device
#     # Compute cosine noise schedule inside the function
#     num_timesteps = 1000 # Adjust if needed
#     betas_t = cosine_schedule(num_timesteps).to(device)  # Ensure schedule is on correct device
#     alphas_t = 1.0 - betas_t
#     alphas_bar_t = torch.cumprod(alphas_t, dim=0)  # Compute cumulative product

#     # Ensure t is on the same device
#     t = t.to(device)
    
#     # Sample alpha values for current t
#     sqrt_alphas_bar_t_sampled = torch.sqrt(alphas_bar_t[t])
#     sqrt_1_minus_alphas_bar_t_sampled = torch.sqrt(1.0 - alphas_bar_t[t])
    
#     # Introduce slight randomness to noise variance
#     perturb_factor = 1.0 + noise_perturb * torch.randn_like(sqrt_1_minus_alphas_bar_t_sampled, device=device)
#     sqrt_1_minus_alphas_bar_t_sampled *= perturb_factor
    
#     # Generate noisy sample
#     x_t = sqrt_alphas_bar_t_sampled * x0 + sqrt_1_minus_alphas_bar_t_sampled * noise
    
#     # Option to add random noise at each step
#     if add_random_noise:
#         x_t += torch.randn_like(x_t, device=device) * torch.sqrt(1 - alphas_bar_t[t])
    
#     # Compute SNR for Min-SNR weighting
#     snr_t = (alphas_bar_t[t]) / (1 - alphas_bar_t[t] + 1e-20)
#     w_t = torch.minimum(torch.tensor(gamma, device=device) / snr_t, torch.tensor(1.0, device=device))
    
#     return x_t, w_t

def get_noisy_seismic(x0, t, schedule):
    """
    Apply noise to a 3D seismic cube and return the transformed noisy version.
    
    Args:
    x0 (torch.Tensor): The original seismic cube (D, H, W).
    t (int): The timestep controlling noise level.
    transform (callable): Transformation to apply after adding noise.

    Returns:
    torch.Tensor: Noisy 3D seismic cube.
    """
    x_noisy , _ = sample_q(x0, t, schedule)  # Generate noisy data
    return x_noisy

def show_noisy_seismic_cubes(noisy_cubes, slice_idx=None, spacing=2, cmap="seismic"):
    """
    Displays slices from 2D or 3D seismic cubes at different noise levels.

    Args:
        noisy_cubes (list[list[np.ndarray or torch.Tensor]]): A list of lists containing 2D or 3D arrays.
        slice_idx (int, optional): The index at which to take slices (if 3D). Defaults to the middle depth.
        spacing (int): Controls spacing between subplots.
        cmap (str): Colormap for visualization.

    Returns:
        None: Displays the slices using Matplotlib.
    """
    if not noisy_cubes or not noisy_cubes[0]:
        raise ValueError("Input noisy_cubes must be a non-empty list of lists containing 2D or 3D arrays.")

    num_rows = len(noisy_cubes)  # Number of different seismic cubes (rows)
    num_cols = len(noisy_cubes[0])  # Number of noise levels (columns)

    # Convert tensors to NumPy and check dimensions
    for row in noisy_cubes:
        for idx, cube in enumerate(row):
            if isinstance(cube, torch.Tensor):
                cube = cube.cpu().numpy()  # Convert tensor to NumPy
                row[idx] = cube  # Replace tensor with NumPy array

            if cube.ndim not in [2, 3]:
                raise ValueError(f"Expected a 2D or 3D array, but got shape {cube.shape}")

    # Determine if we're working with 2D or 3D data
    is_3D = noisy_cubes[0][0].ndim == 3

    # Set default slice index if working with 3D data
    if is_3D:
        depth, height, width = noisy_cubes[0][0].shape
        if slice_idx is None:
            slice_idx = depth // 2
    else:
        height, width = noisy_cubes[0][0].shape  # 2D case

    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    # Ensure axes is always 2D (for single row or column cases)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plot each noisy seismic cube slice
    for row_idx, cube_set in enumerate(noisy_cubes):
        for col_idx, cube in enumerate(cube_set):
            ax = axes[row_idx, col_idx]
            if is_3D:
                ax.imshow(cube[slice_idx, :, :], cmap=cmap, aspect='auto')  # 3D case (XY slice)
            else:
                ax.imshow(cube.T, cmap=cmap, aspect='auto')  # 2D case
                im = ax.imshow(cube.T, cmap=cmap, aspect='auto') 
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Amplitude")
            ax.set_title(f"Noise {col_idx + 1}")
            ax.axis("off")

    plt.tight_layout(pad=spacing / 10)  # Adjust spacing
    plt.show()
