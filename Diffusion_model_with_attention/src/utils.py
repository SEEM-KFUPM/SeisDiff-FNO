import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from numpy.fft import fft2, fftshift
import torch
from scipy.ndimage import generic_filter
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_gradient_magnitude


def plot_2D_seismic(seismic_data, cmap="seismic", title="Seismic Line from .npy"):
    """
    Plots a 2D seismic section (either raw amplitudes or attribute maps).

    Parameters:
        seismic_data (np.ndarray or torch.Tensor): 2D array of seismic data.
        cmap (str): Colormap to use for visualization.
        title (str): Title for the plot.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(seismic_data, torch.Tensor):
        seismic_data = seismic_data.detach().cpu().numpy()

    # Calculate dynamic range
    vmin = np.min(seismic_data)
    vmax = np.max(seismic_data)

    # Choose normalization based on data range
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None  # Default linear normalization

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(
        seismic_data.T,
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='nearest',
        origin='upper'
    )
    plt.colorbar(label="Amplitude" if cmap == "seismic" else "Attribute Value")
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Time Samples")
    plt.tight_layout()
    plt.show()
    
    
def normalize_to_range(seismic_data, mode='standardize', print_stats=True, return_params=True):
    # Ensure correct data type
    is_tensor = isinstance(seismic_data, torch.Tensor)
    
    if is_tensor:
        seismic_data = seismic_data.clone().detach().float()
    else:
        seismic_data = np.array(seismic_data, dtype=np.float32)
        
    # Compute initial statistics
    min_val = seismic_data.min().item() if is_tensor else seismic_data.min()
    max_val = seismic_data.max().item() if is_tensor else seismic_data.max()
    mean_val = seismic_data.mean().item() if is_tensor else seismic_data.mean()
    std_val = seismic_data.std().item() if is_tensor else seismic_data.std()

    if print_stats:
        print(f" **Before Normalization ({'Tensor' if is_tensor else 'NumPy'})**")
        print(f"   ➤ Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")

    if mode == 'minmax':
        # Compute clipping value
        clip_value = (
            torch.quantile(seismic_data.abs(), 0.999) if is_tensor 
            else np.percentile(np.abs(seismic_data), 99.9)
        )
        seismic_data = torch.clamp(seismic_data, -clip_value, clip_value) if is_tensor else np.clip(seismic_data, -clip_value, clip_value)
        min_val = seismic_data.min().item() if is_tensor else seismic_data.min()
        max_val = seismic_data.max().item() if is_tensor else seismic_data.max()
        norm_data = 2 * (seismic_data - min_val) / (max_val - min_val + 1e-8) - 1
        params = {'mode': 'minmax', 'min': min_val, 'max': max_val}

    elif mode == 'standardize':
        mean = seismic_data.mean().item() if is_tensor else seismic_data.mean()
        std = seismic_data.std().item() + 1e-8 if is_tensor else seismic_data.std() + 1e-8
        norm_data = (seismic_data - mean) / std
        params = {'mode': 'standardize', 'mean': mean, 'std': std}

    else:
        raise ValueError("Invalid mode. Choose 'standardize' or 'minmax'.")

    if print_stats:
        print(f"\n **After Normalization ({'Tensor' if is_tensor else 'NumPy'})**")
        print(f"   ➤ Min: {norm_data.min()}, Max: {norm_data.max()}, Mean: {norm_data.mean()}, Std: {norm_data.std()}\n")

    return (norm_data, params) if return_params else norm_data


def denormalize_data(norm_data, params):
    """
    Reverses the normalization applied to a seismic slice.
    
    Args:
        norm_data: Normalized tensor or array.
        params (dict): Parameters returned from normalize_to_range.
    
    Returns:
        Original-scale data.
    """
    if params['mode'] == 'standardize':
        return norm_data * params['std'] + params['mean']
    elif params['mode'] == 'minmax':
        return ((norm_data + 1) / 2) * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError("Invalid normalization parameters.")


    
def normalize_pair(x_high, x_low, mode='standardize', print_stats=True):
    """
    Normalize high- and low-resolution seismic data using shared statistics 
    to preserve amplitude ratios.

    Args:
        x_high: High-resolution data (2D or batched 3D tensor or array)
        x_low: Low-resolution data (same type as x_high)
        mode: 'standardize' or 'minmax'
        print_stats: Print stats before and after normalization

    Returns:
        norm_high, norm_low: normalized versions of x_high and x_low
    """
    is_tensor = isinstance(x_high, torch.Tensor)
    
    # Stack to compute joint statistics
    stacked = torch.cat([x_high.flatten(), x_low.flatten()]) if is_tensor else np.concatenate([x_high.flatten(), x_low.flatten()])

    if mode == 'standardize':
        mean = stacked.mean()
        std = stacked.std() + 1e-8
        norm_high = (x_high - mean) / std
        norm_low = (x_low - mean) / std

    elif mode == 'minmax':
        clip_val = (
            torch.quantile(stacked.abs(), 0.999) if is_tensor 
            else np.percentile(np.abs(stacked), 99.9)
        )
        x_high = torch.clamp(x_high, -clip_val, clip_val) if is_tensor else np.clip(x_high, -clip_val, clip_val)
        x_low  = torch.clamp(x_low,  -clip_val, clip_val) if is_tensor else np.clip(x_low, -clip_val, clip_val)

        min_val = stacked.min()
        max_val = stacked.max()
        norm_high = 2 * (x_high - min_val) / (max_val - min_val + 1e-8) - 1
        norm_low  = 2 * (x_low  - min_val) / (max_val - min_val + 1e-8) - 1

    else:
        raise ValueError("mode must be 'standardize' or 'minmax'")

    if print_stats:
        def print_summary(name, data):
            d_min = data.min().item() if is_tensor else data.min()
            d_max = data.max().item() if is_tensor else data.max()
            d_mean = data.mean().item() if is_tensor else data.mean()
            d_std = data.std().item() if is_tensor else data.std()
            print(f"?? {name}: min={d_min:.3f}, max={d_max:.3f}, mean={d_mean:.3f}, std={d_std:.3f}")

        print("?? After Normalization:")
        print_summary("High-Resolution", norm_high)
        print_summary("Low-Resolution", norm_low)

    return norm_high, norm_low

    
    

def compare_images(original, reconstructed, title_prefix=""):
    # Convert to NumPy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Convert from [1, H, W] if needed
    if original.ndim == 3 and original.shape[0] == 1:
        original = original[0]
    if reconstructed.ndim == 3 and reconstructed.shape[0] == 1:
        reconstructed = reconstructed[0]

    # Normalize to [-1, 1]
    #original = np.clip(original, -1, 1)
    #reconstructed = np.clip(reconstructed, -1, 1)

    # Metrics
    if original.shape == reconstructed.shape:
        psnr_val = psnr(original, reconstructed, data_range=2)
        ssim_val = ssim(original, reconstructed, data_range=2)
    
        print(f"?? {title_prefix} PSNR: {psnr_val:.2f} dB")
        print(f"?? {title_prefix} SSIM: {ssim_val:.4f}")

    # Compute FFTs
    original_fft = np.log(np.abs(fftshift(fft2(original))) + 1e-8)
    recon_fft = np.log(np.abs(fftshift(fft2(reconstructed))) + 1e-8)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    im0 = axes[0, 0].imshow(original.T, cmap="seismic", aspect="auto")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].set_xlabel("Trace")
    axes[0, 0].set_ylabel("Time Samples")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(reconstructed.T, cmap="seismic", aspect="auto")
    axes[0, 1].set_title("Reconstructed Image")
    axes[0, 1].set_xlabel("Trace")
    axes[0, 1].set_ylabel("Time Samples")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(original_fft.T, cmap="viridis", aspect="auto")
    axes[1, 0].set_title("Original Spectrum")
    axes[1, 0].set_xlabel("Frequency (X)")
    axes[1, 0].set_ylabel("Frequency (Y)")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(recon_fft.T, cmap="viridis", aspect="auto")
    axes[1, 1].set_title("Reconstructed Spectrum")
    axes[1, 1].set_xlabel("Frequency (X)")
    axes[1, 1].set_ylabel("Frequency (Y)")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
    
    


def compute_mean_fft_amplitude(image, dt):
    """
    Computes the mean FFT amplitude spectrum over vertical axis (samples).
    Returns frequencies (Hz) and averaged amplitude.

    Parameters:
    - image: 2D numpy array (samples x traces)
    - dt: sampling interval in seconds (e.g., 0.002 for 2 ms)

    Returns:
    - freqs: frequency values in Hz (up to Nyquist)
    - mean_amplitude: mean FFT amplitude spectrum
    """
    fft_vals = np.fft.fft(image, axis=0)
    magnitude = np.abs(fft_vals)
    mean_amplitude = np.mean(magnitude, axis=1)
    
    n_samples = image.shape[0]
    freqs = np.fft.fftfreq(n_samples, d=dt)  # now uses dt!
    
    # Only return the positive frequencies (up to Nyquist)
    pos_freqs = freqs[:n_samples // 2]
    pos_amplitude = mean_amplitude[:n_samples // 2]

    return pos_freqs, pos_amplitude


def plot_spectrum_comparison(original, dt_orig, predicted=None, dt_pred=None,
                             label1='raw', label2='output', max_freq=None):
    freqs_orig, amp_orig = compute_mean_fft_amplitude(original, dt_orig)
    plt.figure(figsize=(6,5))
    plt.plot(freqs_orig, amp_orig, label=label1, linewidth=2)

    max_freq = freqs_orig.max()
    if predicted is not None:
        if dt_pred is None:
            dt_pred = dt_orig
        freqs_pred, amp_pred = compute_mean_fft_amplitude(predicted, dt_pred)
        plt.plot(freqs_pred, amp_pred, label=label2, linewidth=2)
        if max_freq == None:
            max_freq = max(max_freq, freqs_pred.max())

    plt.xlabel("Frequency (Hz)", fontsize=14, weight='bold')
    plt.ylabel("Amplitude", fontsize=14, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xlim(0, max_freq)
    plt.tight_layout()
    plt.show()
    

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable

def compute_fault_likelihood(seismic, sigma=1.0):
    """
    Compute a basic fault likelihood map using gradient magnitude.

    Parameters:
        seismic (2D numpy array): Seismic amplitude image.
        sigma (float): Standard deviation for Gaussian smoothing (gradient scale).

    Returns:
        2D numpy array: Normalized fault likelihood map in range [0, 1].
    """
    grad = gaussian_gradient_magnitude(seismic, sigma=sigma)
    norm_grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return norm_grad


def plot_fault_likelihood_comparison(original, predicted, cmap_seismic="gray", cmap_fault="jet"):
    """
    Plot the original and predicted seismic sections with their fault likelihood maps.

    Parameters:
        original (2D numpy array): Original high-res seismic.
        predicted (2D numpy array): Predicted/reconstructed seismic.
    """
    # Compute fault likelihoods
    fault_orig = compute_fault_likelihood(original)
    fault_pred = compute_fault_likelihood(predicted)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Original seismic
    im0 = axes[0, 0].imshow(original.T, cmap=cmap_seismic, aspect="auto")
    axes[0, 0].set_title("Original Seismic")
    axes[0, 0].set_ylabel("Time")
    axes[0, 0].set_xlabel("Trace")
    plt.colorbar(im0, ax=axes[0, 0])

    # Predicted seismic
    im1 = axes[0, 1].imshow(predicted.T, cmap=cmap_seismic, aspect="auto")
    axes[0, 1].set_title("Predicted Seismic")
    axes[0, 1].set_ylabel("Time")
    axes[0, 1].set_xlabel("Trace")
    plt.colorbar(im1, ax=axes[0, 1])

    # Original Fault Likelihood
    im3 = axes[1, 0].imshow(fault_orig.T, cmap=cmap_fault, aspect="auto", vmin=0, vmax=1)
    axes[1, 0].set_title("Original Fault Likelihood")
    axes[1, 0].set_ylabel("Time")
    axes[1, 0].set_xlabel("Trace")
    plt.colorbar(im3, ax=axes[1, 0], label="Likelihood")

    # Predicted Fault Likelihood
    im4 = axes[1, 1].imshow(fault_pred.T, cmap=cmap_fault, aspect="auto", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Fault Likelihood")
    axes[1, 1].set_ylabel("Time")
    axes[1, 1].set_xlabel("Trace")
    plt.colorbar(im4, ax=axes[1, 1], label="Likelihood")

    plt.tight_layout()
    plt.show()


def compute_seismic_variance_attribute(seismic, window_size=(3, 3), print_stats=True):
    """
    Computes seismic variance attribute over a 2D seismic section.

    Parameters:
        seismic (2D numpy.ndarray): Seismic amplitude section (time/depth x traces).
        window_size (tuple): Size of the local window (vertical, horizontal).
        print_stats (bool): Whether to print basic statistics of the attribute.

    Returns:
        2D numpy.ndarray: Variance attribute map.
    """
    if not isinstance(seismic, np.ndarray):
        raise TypeError("Input seismic data must be a 2D NumPy array.")
    if seismic.ndim != 2:
        raise ValueError("Seismic input must be 2D (time x trace).")

    # Apply a sliding window to compute local variance
    variance_attr = generic_filter(seismic, np.var, size=window_size, mode='reflect')

    if print_stats:
        print(f" Variance Attribute Stats — Min: {np.min(variance_attr):.4f}, Max: {np.max(variance_attr):.4f}, Mean: {np.mean(variance_attr):.4f}")

    return variance_attr

def apply_amplitude_gain(seismic, gain_type='linear', gain_param=None, axis=1):
    """
    Apply amplitude gain to seismic data.

    Parameters:
        seismic (np.ndarray or torch.Tensor): 2D seismic data (e.g., [time, traces] or [H, W]).
        gain_type (str): Type of gain. Options:
            - 'linear': linearly increasing gain with time/depth.
            - 'exponential': exponential gain with time/depth.
            - 'constant': multiply entire data by a constant.
            - 'agc': automatic gain control (local RMS normalization).
        gain_param (float or int): Parameter depending on type.
            - For 'constant': scalar multiplier (e.g., 2.0).
            - For 'linear' or 'exponential': scaling factor (default = 1.0).
            - For 'agc': window length (e.g., 50 samples).
        axis (int): Axis along which to apply time-based gain (default: 0 for time/depth axis).

    Returns:
        Seismic data with gain applied (same type as input).
    """
    is_tensor = isinstance(seismic, torch.Tensor)
    if is_tensor:
        seismic = seismic.detach().cpu().numpy()

    data = seismic.copy()
    n_samples = data.shape[axis]

    if gain_type == 'constant':
        factor = gain_param if gain_param is not None else 1.0
        data *= factor

    elif gain_type == 'linear':
        scale = gain_param if gain_param is not None else 1.0
        gain = np.linspace(1.0, scale, n_samples).reshape(-1, 1) if axis == 0 else np.linspace(1.0, scale, n_samples)
        data = data * gain

    elif gain_type == 'exponential':
        scale = gain_param if gain_param is not None else 1.0
        gain = np.exp(np.linspace(0, scale, n_samples)).reshape(-1, 1) if axis == 0 else np.exp(np.linspace(0, scale, n_samples))
        data = data * gain

    elif gain_type == 'agc':
        window = int(gain_param) if gain_param is not None else 50
        def agc(trace):
            rms = np.sqrt(uniform_filter1d(trace**2, size=window, mode='reflect') + 1e-8)
            return trace / rms
        if axis == 0:
            data = np.apply_along_axis(agc, axis=0, arr=data)
        else:
            data = np.apply_along_axis(agc, axis=1, arr=data)

    else:
        raise ValueError("Invalid gain_type. Choose from 'constant', 'linear', 'exponential', 'agc'.")

    return torch.tensor(data, dtype=torch.float32) if is_tensor else data

def compute_seismic_variance_attribute(seismic, window_size=(3, 3), print_stats=True):
    """
    Computes seismic variance attribute over a 2D seismic section.

    Parameters:
        seismic (2D numpy.ndarray): Seismic amplitude section (time/depth x traces).
        window_size (tuple): Size of the local window (vertical, horizontal).
        print_stats (bool): Whether to print basic statistics of the attribute.

    Returns:
        2D numpy.ndarray: Variance attribute map.
    """
    if isinstance(seismic, torch.Tensor):
        seismic = seismic.detach().cpu().numpy()
    # Apply a sliding window to compute local variance
    variance_attr = generic_filter(seismic, np.var, size=window_size, mode='reflect')

    if print_stats:
        print(f" Variance Attribute Stats — Min: {np.min(variance_attr):.4f}, Max: {np.max(variance_attr):.4f}, Mean: {np.mean(variance_attr):.4f}")

    return variance_attr

def compute_variance_attribute(seismic, window_size=5):
    """
    Simple variance-based fault attribute (sliding variance window).
    """
    is_tensor = isinstance(seismic, torch.Tensor)
    if is_tensor:
        seismic = seismic.detach().cpu().numpy()

    padded = np.pad(seismic, ((window_size//2,),(0,)), mode='edge')
    attribute = np.zeros_like(seismic)

    for i in range(seismic.shape[0]):
        window = padded[i:i+window_size, :]
        attribute[i, :] = np.var(window, axis=0)

    # Normalize
    attribute = (attribute - np.min(attribute)) / (np.max(attribute) - np.min(attribute))
    return gaussian_filter(attribute, sigma=1.0)
def plot_fault_overlay(seismic, fault_map, title="Fault Likelihood Overlay", cmap='turbo'):
    """
    Overlay fault likelihood map on seismic section.
    """
    is_tensor = isinstance(seismic, torch.Tensor)
    if is_tensor:
        seismic = seismic.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show seismic
    ax.imshow(seismic.T, cmap='gray', aspect='auto', interpolation='nearest', origin='upper')

    # Overlay fault likelihood
    im = ax.imshow(fault_map.T, cmap=cmap, alpha=0.5, aspect='auto', interpolation='nearest', origin='upper')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Traces")
    ax.set_ylabel("Samples")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Fault Likelihood")

    plt.tight_layout()
    plt.show()

def compare_seismic_traces(original_trace, predicted_trace, trace_id=None):
    """
    Compare original and predicted seismic traces.

    Parameters:
    - original_trace (np.ndarray): The true seismic trace [time_samples].
    - predicted_trace (np.ndarray): The predicted seismic trace [time_samples].
    - trace_id (int or str, optional): ID or index for the trace (for labeling).
    """
    is_tensor = isinstance(original_trace, torch.Tensor)
    if is_tensor:
        original_trace = original_trace.detach().cpu().numpy()

    is_tensor = isinstance(predicted_trace, torch.Tensor)
    if is_tensor:
        predicted_trace = predicted_trace.detach().cpu().numpy()
    
    assert original_trace.shape == predicted_trace.shape, "Traces must have the same shape."

    time = np.arange(len(original_trace))

    plt.figure(figsize=(16, 6))
    plt.plot(time, original_trace, label="Ground truth", linewidth=2)
    plt.plot(time, predicted_trace, label="Output", linewidth=2)
    plt.title(f"Seismic Trace Comparison" + (f" (Trace {trace_id})" if trace_id is not None else ""))
    plt.xlabel("Time Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()
