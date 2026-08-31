import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
from numpy.fft import fft2, fftshift
import torch

from scipy.ndimage import generic_filter, gaussian_filter, gaussian_gradient_magnitude, uniform_filter1d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.signal import butter, filtfilt, sosfiltfilt, sosfreqz, iirfilter,  sosfreqz, hilbert
import matplotlib.gridspec as gridspec



# ---------------------------
# Small helpers
# ---------------------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_same_type(x_like, arr):
    # Return arr as a torch tensor if x_like was torch, else numpy
    if isinstance(x_like, torch.Tensor):
        return torch.tensor(arr, dtype=torch.float32, device=x_like.device)
    return arr

def _percentile_range(x, p_lo=1.0, p_hi=99.0):
    x = _to_numpy(x)
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if not np.isfinite(lo): lo = np.min(x)
    if not np.isfinite(hi): hi = np.max(x)
    if hi <= lo:  # degenerate
        lo, hi = float(np.min(x)), float(np.max(x))
    return lo, hi


# ---------------------------
# Visualization
# ---------------------------


def plot_2D_seismic(
    seismic_data,
    cmap="seismic",
    title="Seismic Line",
    vmin=None,
    vmax=None,
    common_range=None,
):
    """
    Plots a 2D seismic section with:
      - zero centered to black,
      - consistent color range if 'common_range' is provided.
    """
    arr = np.array(seismic_data)
    
    # compute color limits if not provided
    if common_range is not None:
        vmin, vmax = common_range
    else:
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(arr, [1, 99])
    
    # ensure symmetric range around zero for better comparison
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs

    # zero-centered norm
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # create a custom seismic colormap with black at zero
    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, 256))
    mid_idx = 128  # zero position
    black_center_cmap = LinearSegmentedColormap.from_list("seismic_black", colors)

    # plot
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        arr.T,
        cmap=black_center_cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Amplitude")
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Time Samples")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Normalization
# ---------------------------
class GlobalRobustScalerHR:
    """
    Fit on TRAIN HR only (recommended). Apply the SAME transform to LR and HR.
    Maps to ~[-1, 1] using robust percentiles, resistant to outliers.
    """
    def __init__(self, p_low=1.0, p_high=99.0, eps=1e-6):
        assert 0 <= p_low < p_high <= 100
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.fitted = False
        self.lo = None
        self.scale = None
        self.device = None

    @torch.no_grad()
    def _as_4d(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure (B,C,H,W)
        if x.ndim == 3:  # (B,H,W)
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"HR tensor must be (B,C,H,W) or (B,H,W); got {x.shape}")
        return x

    @torch.no_grad()
    def fit(self, train_loader, key_hr="hr", device="cpu", max_batches=None):
        """
        Scan HR batches, compute robust lo/hi per channel.
        - train_loader yields dicts (expects key 'hr') or tuples (takes [0] as HR).
        - max_batches: limit number of batches to speed up fitting (optional).
        """
        lows, highs = [], []
        count = 0

        for batch in train_loader:
            # 1) extract HR
            if isinstance(batch, dict):
                if key_hr not in batch:
                    raise KeyError(f"Batch dict must contain '{key_hr}' key; keys={list(batch.keys())}")
                hr = batch[key_hr]
            else:  # tuple/list
                hr = batch[0]
            hr = self._as_4d(hr.float().to(device))  # (B,C,H,W)

            B, C, H, W = hr.shape

            # 2) reshape so we can quantile over a SINGLE dim:
            #    stack (B,H,W) into one axis => (N, C)
            x = hr.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

            # 3) per-channel percentiles along N (dim=0)
            lo = torch.quantile(x, self.p_low  / 100.0, dim=0, keepdim=False).view(1, C, 1, 1)
            hi = torch.quantile(x, self.p_high / 100.0, dim=0, keepdim=False).view(1, C, 1, 1)

            lows.append(lo); highs.append(hi)
            count += 1
            if max_batches is not None and count >= max_batches:
                break

        if not lows:
            raise ValueError("No batches were seen in scaler.fit(); check your loader or max_batches.")

        lows_stack  = torch.stack(lows,  dim=0)  # (N,1,C,1,1)
        highs_stack = torch.stack(highs, dim=0)  # (N,1,C,1,1)
        lo = lows_stack.amin(dim=0)              # (1,C,1,1)
        hi = highs_stack.amax(dim=0)             # (1,C,1,1)

        self.lo = lo
        self.scale = (hi - lo).clamp(min=self.eps)
        self.device = lo.device
        self.fitted = True

    @torch.no_grad()
    def _ensure_4d(self, x: torch.Tensor):
        squeeze_B = False; squeeze_C = False
        if x.ndim == 2:            # (H,W)
            x = x.unsqueeze(0).unsqueeze(0); squeeze_B = True; squeeze_C = True
        elif x.ndim == 3:          # (B,H,W) or (C,H,W)
            if x.shape[0] > 4:     # heuristic: treat as (B,H,W)
                x = x.unsqueeze(1); squeeze_C = True
            else:                   # treat as (C,H,W)
                x = x.unsqueeze(0); squeeze_B = True
        elif x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W)/(B,H,W)/(C,H,W)/(H,W); got {x.shape}")
        return x, squeeze_B, squeeze_C

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Call fit() first."
        x = x.to(self.device).float()
        x, squeeze_B, squeeze_C = self._ensure_4d(x)

        # broadcast lo/scale over batch & spatial dims
        if self.lo.shape[1] != x.shape[1]:
            if self.lo.shape[1] == 1:
                lo, scale = self.lo, self.scale
            else:
                raise ValueError(f"Channel mismatch: scaler C={self.lo.shape[1]} vs x C={x.shape[1]}")
        else:
            lo, scale = self.lo, self.scale

        x_n = (x - lo) / scale * 2.0 - 1.0

        if squeeze_B and squeeze_C: x_n = x_n.squeeze(0).squeeze(0)
        elif squeeze_B:             x_n = x_n.squeeze(0)
        elif squeeze_C:             x_n = x_n.squeeze(1)
        return x_n

    @torch.no_grad()
    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Call fit() first."
        x = x_norm.to(self.device).float()
        x, squeeze_B, squeeze_C = self._ensure_4d(x)

        if self.lo.shape[1] != x.shape[1]:
            if self.lo.shape[1] == 1:
                lo, scale = self.lo, self.scale
            else:
                raise ValueError(f"Channel mismatch: scaler C={self.lo.shape[1]} vs x C={x.shape[1]}")
        else:
            lo, scale = self.lo, self.scale

        x_phys = (x + 1.0) * 0.5 * scale + lo

        if squeeze_B and squeeze_C: x_phys = x_phys.squeeze(0).squeeze(0)
        elif squeeze_B:             x_phys = x_phys.squeeze(0)
        elif squeeze_C:             x_phys = x_phys.squeeze(1)
        return x_phys




# ---------------------------
# Image comparison / metrics
# ---------------------------






def compare_images(original, reconstructed_one, reconstructed_two=None,
                   reconstructed_three=None, title_prefix="",
                   dt_list=None, dx_list=None, titles=None,
                   # ---- paper params ----
                   base_font=10,
                   fig_dpi=600,
                   fk_k_in_per_km=True,
                   show_metrics=False, save_fig_adr=None):   # <-- NEW (default False)
    """
    Paper-style comparison:
      Row 1: Time–Offset
      Row 2: F–K
      Row 3: Mean frequency spectrum (spans all columns)

    If show_metrics=False: PSNR/SSIM are not computed/printed/shown.
    """

    # --- Convert tensors to numpy ---
    ori  = _to_numpy(original)
    rec1 = _to_numpy(reconstructed_one)
    rec2 = _to_numpy(reconstructed_two) if reconstructed_two is not None else None
    rec3 = _to_numpy(reconstructed_three) if reconstructed_three is not None else None

    datasets = [ori, rec1]
    if rec2 is not None: datasets.append(rec2)
    if rec3 is not None: datasets.append(rec3)
    ncols = len(datasets)

    # --- Titles ---
    if titles is None:
        titles = ["Ground Truth", "Reconstruction 1"]
        if rec2 is not None: titles.append("Reconstruction 2")
        if rec3 is not None: titles.append("Reconstruction 3")

    # --- Validate dt/dx ---
    if dt_list is None or dx_list is None:
        raise ValueError("You must provide dt_list and dx_list for each dataset.")
    if len(dt_list) != ncols or len(dx_list) != ncols:
        raise ValueError("dt_list and dx_list must match number of datasets.")

    # --- Optional metrics ---
    psnr_val, ssim_val = None, None
    if show_metrics:
        def print_metrics(ref, test, label):
            if ref.shape != test.shape:
                print(f"{label} | PSNR/SSIM skipped (shape mismatch)")
                return np.nan, np.nan
            dr = float(ref.max() - ref.min()) or 1.0
            p = psnr(ref, test, data_range=dr)
            s = ssim(ref, test, data_range=dr)
            print(f"{label} | PSNR: {p:.2f} dB | SSIM: {s:.4f}")
            return p, s

        psnr_val, ssim_val = print_metrics(ori, rec1, titles[1])
        if rec2 is not None: print_metrics(ori, rec2, titles[2])
        if rec3 is not None: print_metrics(ori, rec3, titles[3])

    # --- Shared normalization for t–x ---
    all_vals = np.concatenate([d.ravel() for d in datasets])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    vmax_abs = max(abs(vmin), abs(vmax))
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    # --- Seismic colormap ---
    base_cmap = plt.get_cmap("seismic")
    colors = base_cmap(np.linspace(0, 1, 256))
    seismic_black = LinearSegmentedColormap.from_list("seismic_black", colors)

    # --- Helper: F–K spectrum (log amplitude) ---
    def fk_spectrum_log(data, dt, dx):
        data = _to_numpy(data)
        Nt, Nx = data.shape

        F = torch.fft.fft2(torch.tensor(data))
        F_shift = torch.fft.fftshift(F)
        F_abs = F_shift.abs().cpu().numpy()

        f = np.fft.fftfreq(Nt, dt)
        k = np.fft.fftfreq(Nx, dx)

        fshift = np.fft.fftshift(f)
        kshift = np.fft.fftshift(k)

        pos = fshift > 0
        F_half = F_abs[pos, :]
        f_half = fshift[pos]

        # convert k: 1/m -> 1/km
        kshift_plot = (kshift * 1000.0) if fk_k_in_per_km else kshift

        extent = [kshift_plot.min(), kshift_plot.max(), f_half.min(), f_half.max()]
        log_amp = np.log(F_half + 1e-8)
        return log_amp, extent

    # --- Fonts ---
    title_fs = base_font + 2
    label_fs = base_font
    tick_fs  = base_font - 2
    cbar_fs  = base_font - 2

    # --- Figure layout ---
    fig = plt.figure(figsize=(6 * ncols, 12), dpi=fig_dpi)
    gs = gridspec.GridSpec(
        3, ncols,
        height_ratios=[1, 1, 0.9],
        hspace=0.42, wspace=0.28
    )

    # --- Row 1: Time–Offset ---
    for i, (data, label) in enumerate(zip(datasets, titles)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(data.T, cmap=seismic_black, norm=norm,
                       aspect="auto", origin="upper")
        ax.set_title(f"{label} (Time–Offset)", fontsize=title_fs, weight="bold")
        ax.set_xlabel("Trace", fontsize=label_fs, weight="bold")
        ax.set_ylabel("Time Samples", fontsize=label_fs, weight="bold")
        ax.tick_params(labelsize=tick_fs)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Amplitude", fontsize=label_fs, weight="bold")
        cbar.ax.tick_params(labelsize=cbar_fs)

    # --- Row 2: F–K ---
    fk_data, fk_extents = [], []
    for data, dt, dx in zip(datasets, dt_list, dx_list):
        log_amp, extent = fk_spectrum_log(data, dt, dx)
        fk_data.append(log_amp)
        fk_extents.append(extent)

    all_fk_vals = np.concatenate([d.ravel() for d in fk_data])
    vmin_fk, vmax_fk = np.percentile(all_fk_vals, [1, 99])

    for i, (log_amp, extent, label) in enumerate(zip(fk_data, fk_extents, titles)):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(log_amp, extent=extent, aspect="auto",
                       origin="lower", cmap="jet",
                       vmin=vmin_fk, vmax=vmax_fk)

        ax.set_title(f"{label} (F–K Spectrum)", fontsize=title_fs, weight="bold")
        ax.set_ylabel("Frequency f (Hz)", fontsize=label_fs, weight="bold")

        ax.set_xlabel(
            "Wavenumber k (1/km)" if fk_k_in_per_km else "Wavenumber k (1/m)",
            fontsize=label_fs, weight="bold"
        )

        ax.tick_params(labelsize=tick_fs)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log |F(f,k)|", fontsize=label_fs, weight="bold")
        cbar.ax.tick_params(labelsize=cbar_fs)

    # --- Row 3: Spectrum comparison ---
    ax_spec = fig.add_subplot(gs[2, :])
    plot_spectrum_comparison(
        original.T, dt_list[0],
        predicted_one=reconstructed_one.T, dt_pred_one=dt_list[1],
        predicted_two=reconstructed_two if rec2 is not None else None,
        dt_pred_two=dt_list[2] if rec2 is not None else None,
        predicted_three=reconstructed_three if rec3 is not None else None,
        dt_pred_three=dt_list[3] if rec3 is not None else None,
        label1=titles[0],
        label2=titles[1],
        label3=titles[2] if rec2 is not None else "Model 2",
        label4=titles[3] if rec3 is not None else "Model 3",
        title="Mean Frequency Spectrum",
        ax=ax_spec
    )
    ax_spec.title.set_fontsize(title_fs)
    ax_spec.title.set_weight("bold")
    ax_spec.xaxis.label.set_fontsize(label_fs)
    ax_spec.yaxis.label.set_fontsize(label_fs)
    ax_spec.xaxis.label.set_weight("bold")
    ax_spec.yaxis.label.set_weight("bold")
    ax_spec.tick_params(labelsize=tick_fs)
    leg = ax_spec.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(tick_fs)

    # --- Suptitle (no metrics) ---
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=title_fs + 2, weight="bold", y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    if save_fig_adr is not None:
        plt.savefig(save_fig_adr,
            dpi=600, bbox_inches="tight")
    plt.show()



# ---------------------------
# Spectrum utilities
# ---------------------------


def fk_spectrum_log(data, dt, dx):
        Nt, Nx = data.shape
        F = torch.fft.fft2(torch.tensor(data))
        F_shift = torch.fft.fftshift(F)
        F_abs = F_shift.abs().numpy()
        f = np.fft.fftfreq(Nt, dt)
        k = np.fft.fftfreq(Nx, dx)
        fshift = np.fft.fftshift(f)
        kshift = np.fft.fftshift(k)
        pos = fshift > 0
        F_half = F_abs[pos, :]
        f_half = fshift[pos]
        extent = [kshift.min(), kshift.max(), f_half.min(), f_half.max()]
        log_amp = np.log(F_half + 1e-8)
        return log_amp, extent

def compute_mean_fft_amplitude(image, dt):
    """
    Computes the mean FFT amplitude spectrum over traces (axis=1),
    treating axis 0 as time/depth samples. Returns positive freqs and mean amplitude.
    """
    img = _to_numpy(image)
    fft_vals = np.fft.fft(img, axis=0)
    magnitude = np.abs(fft_vals)
    mean_amplitude = magnitude.mean(axis=1)

    n_samples = img.shape[0]
    freqs = np.fft.fftfreq(n_samples, d=dt)
    pos = slice(0, n_samples // 2)
    return freqs[pos], mean_amplitude[pos]


def plot_spectrum_comparison(original, dt_orig,
                             predicted_one=None, dt_pred_one=None,
                             predicted_two=None, dt_pred_two=None,
                             predicted_three=None, dt_pred_three=None,
                             label1='Ground Truth', label2='Model 1',
                             label3='Model 2', label4='Model 3',
                             title='Mean Frequency Spectrum Comparison',
                             max_freq=None,
                             ax=None,
                             show=True):
    """
    Compare up to four seismic sections in the frequency domain.
    If ax is provided, plots into that axis (no new figure).
    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    # --- Compute base spectrum (reference) ---
    f1, a1 = compute_mean_fft_amplitude(original, dt_orig)
    max_ampli = a1.max() if a1.max() != 0 else 1.0
    maxf = f1.max() if max_freq is None else max_freq
    ax.plot(f1, a1 / max_ampli, label=label1, linewidth=2)

    # --- Predicted one ---
    if predicted_one is not None:
        dt_pred_one = dt_orig if dt_pred_one is None else dt_pred_one
        f2, a2 = compute_mean_fft_amplitude(predicted_one, dt_pred_one)
        ax.plot(f2, a2 / max_ampli, label=label2, linewidth=2)
        if max_freq is None:
            maxf = max(maxf, f2.max())

    # --- Predicted two ---
    if predicted_two is not None:
        dt_pred_two = dt_orig if dt_pred_two is None else dt_pred_two
        f3, a3 = compute_mean_fft_amplitude(predicted_two, dt_pred_two)
        ax.plot(f3, a3 / max_ampli, label=label3, linewidth=2)
        if max_freq is None:
            maxf = max(maxf, f3.max())

    # --- Predicted three ---
    if predicted_three is not None:
        dt_pred_three = dt_orig if dt_pred_three is None else dt_pred_three
        f4, a4 = compute_mean_fft_amplitude(predicted_three, dt_pred_three)
        ax.plot(f4, a4 / max_ampli, label=label4, linewidth=2)
        if max_freq is None:
            maxf = max(maxf, f4.max())

    # --- Formatting ---
    ax.set_xlabel("Frequency (Hz)", fontsize=12, weight='bold')
    ax.set_ylabel("Normalized Amplitude", fontsize=12, weight='bold')
    ax.set_title(title, fontsize=13, weight='bold')
    ax.grid(True, linewidth=0.6, linestyle=':')
    ax.set_xlim(0, maxf)
    ax.legend(loc='upper right', fontsize=11)

    # Only show if we created the figure (standalone use)
    if created_fig and show:
        plt.tight_layout()
        plt.show()


# ---------------------------
# Params / counting
# ---------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


# ---------------------------
# Fault / attributes
# ---------------------------
def compute_fault_likelihood(seismic, sigma=1.0):
    """
    Basic fault likelihood via gradient magnitude + [0,1] normalization.
    Input can be torch or numpy (2D).
    """
    arr = _to_numpy(seismic)
    grad = gaussian_gradient_magnitude(arr, sigma=sigma)
    norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return _to_same_type(seismic, norm)

def plot_fault_likelihood_comparison(original, predicted, cmap_seismic="gray", cmap_fault="jet"):
    ori = _to_numpy(original)
    pred = _to_numpy(predicted)
    fault_orig = _to_numpy(compute_fault_likelihood(ori))
    fault_pred = _to_numpy(compute_fault_likelihood(pred))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im0 = axes[0, 0].imshow(ori.T, cmap=cmap_seismic, aspect="auto")
    axes[0, 0].set_title("Original Seismic")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pred.T, cmap=cmap_seismic, aspect="auto")
    axes[0, 1].set_title("Predicted Seismic")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(fault_orig.T, cmap=cmap_fault, aspect="auto", vmin=0, vmax=1)
    axes[1, 0].set_title("Original Fault Likelihood")
    plt.colorbar(im2, ax=axes[1, 0], label="Likelihood")

    im3 = axes[1, 1].imshow(fault_pred.T, cmap=cmap_fault, aspect="auto", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted Fault Likelihood")
    plt.colorbar(im3, ax=axes[1, 1], label="Likelihood")

    plt.tight_layout()
    plt.show()


def compute_seismic_variance_attribute(seismic, window_size=(3, 3), print_stats=True):
    """
    Local variance attribute using a sliding window (reflect padding).
    Accepts torch or numpy; returns same type as input.
    """
    arr = _to_numpy(seismic)
    var_attr = generic_filter(arr, np.var, size=window_size, mode='reflect')
    if print_stats:
        print(f"[Variance] min={var_attr.min():.4f} max={var_attr.max():.4f} mean={var_attr.mean():.4f}")
    return _to_same_type(seismic, var_attr)


def compute_variance_attribute(seismic, window_size=5):
    """
    Simple variance attribute over a vertical sliding window (manual implementation).
    """
    arr = _to_numpy(seismic)
    pad = window_size // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode='edge')
    out = np.zeros_like(arr)
    # vectorized-ish rolling variance
    for i in range(arr.shape[0]):
        w = padded[i:i + window_size, :]
        out[i, :] = np.var(w, axis=0)
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    out = gaussian_filter(out, sigma=1.0)
    return _to_same_type(seismic, out)


def plot_fault_overlay(seismic, fault_map, title="Fault Likelihood Overlay", cmap='turbo'):
    arr = _to_numpy(seismic)
    flt = _to_numpy(fault_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(arr.T, cmap='gray', aspect='auto', interpolation='nearest', origin='upper')
    im = ax.imshow(flt.T, cmap=cmap, alpha=0.5, aspect='auto', interpolation='nearest', origin='upper')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Traces"); ax.set_ylabel("Samples")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Fault Likelihood")
    plt.tight_layout()
    plt.show()


def compare_seismic_traces(original_trace,
                           one_trace,
                           second_trace=None,
                           third_trace=None,
                           label="Ground Truth",
                           label_one="Model 1",
                           label_two="Model 2",
                           label_three="Model 3",
                           trace_id=None,
                           dt_list=None):
    """
    Compare up to four seismic traces with different sampling intervals (no interpolation).
    Each trace is plotted over the same total time window according to its dt value.
    
    Parameters
    ----------
    original_trace, one_trace, second_trace, third_trace : np.ndarray or torch.Tensor
        1D seismic traces to compare.
    label, label_one, label_two, label_three : str
        Labels for each trace in the legend.
    trace_id : int or None
        Optional trace number for the title.
    dt_list : list of float
        Sampling intervals for each trace, e.g. [0.002, 0.004, 0.008, 0.002].
    """

    # Convert tensors to numpy arrays
    o = _to_numpy(original_trace).flatten()
    p = _to_numpy(one_trace).flatten()
    u = _to_numpy(second_trace).flatten() if second_trace is not None else None
    v = _to_numpy(third_trace).flatten() if third_trace is not None else None

    # --- Default dt_list handling ---
    if dt_list is None:
        dt_list = [1.0, 1.0] + ([1.0] if u is not None else []) + ([1.0] if v is not None else [])
    if len(dt_list) < 2:
        raise ValueError("dt_list must have at least two elements (for original and first trace).")

    # --- Build time axes based on sampling interval ---
    time_o = np.arange(len(o)) * dt_list[0]
    time_p = np.arange(len(p)) * dt_list[1]
    time_u = np.arange(len(u)) * dt_list[2] if (u is not None and len(dt_list) > 2) else None
    time_v = np.arange(len(v)) * dt_list[3] if (v is not None and len(dt_list) > 3) else None

    # --- Plotting ---
    plt.figure(figsize=(18, 6))
    plt.plot(time_o, o, label=label, linewidth=2)
    plt.plot(time_p, p, label=label_one, linewidth=2)
    if u is not None:
        plt.plot(time_u, u, label=label_two, linewidth=2)
    if v is not None:
        plt.plot(time_v, v, label=label_three, linewidth=2)

    plt.title(f"Seismic Trace Comparison" + (f" (Trace {trace_id})" if trace_id is not None else ""),
              fontsize=13, weight='bold')
    plt.xlabel("Time (s)", fontsize=12, weight='bold')
    plt.ylabel("Amplitude", fontsize=12, weight='bold')
    plt.grid(True, linestyle=':')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Normalization process
# ---------------------------


def normalize_to_range(seismic_data, mode='minmax', print_stats=True, return_params=True):
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
        norm_data = 2 * (seismic_data - min_val) / (max_val - min_val + 1e-20) - 1
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

# ---------------------------
# Filters
# ---------------------------


def butter_bandpass(lowcut, highcut, fs,band, order=5):
    """ Bandpass filter

    Parameters
    ----------
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        b : np.array
            The numerator coefficient vector of the filter
        a : np.array
            The denominator coefficient vector of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if band=="band":
        b, a = butter(order, [low, high], btype=band)
    elif band=="low":
        b, a = butter(order, high, btype=band)
    elif band=="high":
        b, a = butter(order, low, btype=band)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs,band="low",order=5):
    """ Apply bandpass filter to trace

    Parameters
    ----------
    data: np.array [1D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order
    band: s

    Returns
    -------
        y : np.array
            Bandpassed data
    """
    
    b, a = butter_bandpass(lowcut, highcut, fs,band, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_filter_response(dt, lowcut, highcut, band='low', order=4,fmax=None):
    """
    Utility: plot the magnitude response of the designed Butterworth low-pass.
    """
    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if band=="band":
        b, a = butter(order, [low, high], btype=band)
    elif band=="low":
        b, a = butter(order, high, btype=band)
    elif band=="high":
        b, a = butter(order, low, btype=band)
    
    w, h = b, a          # frequency response in Hz
    if fmax is None:
        fmax = fs * 0.5
    plt.figure(figsize=(6,4))
    plt.plot(w, np.abs(h), linewidth=2)
    plt.xlim(0, fmax)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (linear)")
    if band=="band":
        plt.title(f"Butterworth BPF | Type={band}, order={order}, low={low} Hz, high={high} Hz")
    elif band=="low":
        plt.title(f"Butterworth LPF | Type={band}, order={order}, high={high} Hz")
    elif band=="high":
        plt.title(f"Butterworth HPF | Type={band}, order={order}, high={low} Hz")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def apply_notch_butter(data, dt, f0=60.0, Q=30.0, order=4, axis=0):
    """
    Zero-phase Butterworth notch (band-stop) along the time axis.

    data : 1D or 2D array
        (samples,) or (samples, traces) if axis=0.
    dt   : float
        Sampling interval [s].
    f0   : float
        Center frequency of the notch [Hz].
    Q    : float
        Quality factor (higher Q = narrower notch). Bandwidth = f0 / Q.
    order: int
        IIR order of the band-stop prototype (4 is a good default).
    axis : int
        Time axis (usually 0).

    Returns: filtered array (same shape).
    """
    x = np.asarray(data)
    fs = 1.0 / dt
    nyq = 0.5 * fs

    # Convert (f0, Q) to stopband edges
    bw = f0 / Q                       # total bandwidth in Hz
    f1 = (f0 - 0.5 * bw)/nyq
    f2 = (f0 + 0.5 * bw)/nyq


    # Design stable Butterworth band-stop in SOS form (more robust than (b,a))
    b, a = butter(order, [f1, f2], btype="bandstop")

    # Zero-phase: forward+backward filtering (no phase distortion)
    y = filtfilt(b, a, data)
    return np.ascontiguousarray(y)    # avoid negative strides later
    
    
    
    
    
# ======================================================
# ������ Geophysically Realistic Noise Functions (Poststack)
# ======================================================

def coherent_lowfreq_noise(x, fs=500, fmin=5, fmax=15, amp=0.1):
    """Low-frequency coherent background noise (residual ground roll)."""
    n_traces, n_samples = x.shape
    t = np.arange(n_samples) / fs
    freq = np.random.uniform(fmin, fmax)
    phase = np.random.uniform(0, 2*np.pi)
    base = np.sin(2*np.pi*freq*t + phase)
    roll = np.tile(base, (n_traces, 1))
    roll *= (1 + 0.05*np.random.randn(n_traces, 1))
    return x + amp * roll


def smooth_random_noise(x, std=0.05, corr_len=5):
    """Smooth random noise — simulates mild random background."""
    noise = np.random.normal(0, std, size=x.shape)
    kernel = np.ones((corr_len, 1)) / corr_len
    smooth = np.apply_along_axis(
        lambda m: np.convolve(m, kernel.ravel(), mode='same'),
        axis=0,
        arr=noise
    )
    return x + smooth


def trace_gain_loss(x, prob=0.05, gain_min=0.5, gain_max=0.9):
    """Random trace amplitude dropouts (weak traces)."""
    mask = np.ones(x.shape[0])
    drop_idx = np.random.choice(
        np.arange(x.shape[0]),
        int(prob * x.shape[0]),
        replace=False
    )
    mask[drop_idx] = np.random.uniform(gain_min, gain_max, len(drop_idx))
    return x * mask[:, None]


def localized_spike(x, n_spikes=3, amp=0.4):
    """Random high-amplitude spikes (instrument or processing noise)."""
    n_traces, n_samples = x.shape
    x = x.copy()
    for _ in range(n_spikes):
        i = np.random.randint(0, n_traces)
        j = np.random.randint(0, n_samples)
        x[i, j] += amp * np.random.choice([-1, 1])
    return x


def mild_phase_shift(x, max_phase_deg=15):
    """Small phase rotation using Hilbert transform (residual migration/NMO)."""
    analytic = hilbert(x, axis=1)
    shift_deg = np.random.uniform(-max_phase_deg, max_phase_deg)
    shift_rad = np.deg2rad(shift_deg)
    rotated = np.real(analytic * np.exp(1j * shift_rad))
    return rotated
    
def apply_noise(x, params):
    x = coherent_lowfreq_noise(x, amp=params["coherent_lowfreq_noise"]["amp"])
    x = smooth_random_noise(x, std=params["smooth_random_noise"]["std"])
    x = trace_gain_loss(
        x,
        prob=params["trace_gain_loss"]["prob"],
        gain_min=params["trace_gain_loss"]["gain_min"],
        gain_max=params["trace_gain_loss"]["gain_max"]
    )
    x = localized_spike(
        x,
        n_spikes=params["localized_spike"]["n_spikes"],
        amp=params["localized_spike"]["amp"]
    )
    x = mild_phase_shift(
        x,
        max_phase_deg=params["mild_phase_shift"]["max_phase_deg"]
    )
    return x