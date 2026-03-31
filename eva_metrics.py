from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


def snr(g, f):
    """
    Compute SNR (Signal-to-Noise Ratio) between two 2D images.
    
    Args:
        g (np.ndarray): Ground truth image, shape (H, W)
        f (np.ndarray): Reconstructed/denoised image, shape (H, W)

    Returns:
        float: SNR value in dB
    """

    # Convert to float in case images are uint8, uint16, etc.
    g = np.double(g)
    f = np.double(f)

    # Check same size
    if g.shape != f.shape:
        raise ValueError("Dimension of two images don't match!")

    # Frobenius norm (Euclidean norm for matrices)
    numerator = np.linalg.norm(g, 'fro')
    denominator = np.linalg.norm(g - f, 'fro')

    # Avoid division by zero
    if denominator == 0:
        return np.inf

    snr_value = 20. * np.log10(numerator / denominator)
    return snr_value

def mse_cal(g, f):
    if g.shape != f.shape:
        raise ValueError("The two cubes must have the same shape.")
    mse = np.mean((g - f)**2)
    return mse

def calculate_snr_psnr(img1, img2):
    """
    Compute SNR and PSNR between two 2D images.

    Parameters:
        img1 (numpy.ndarray): Original 2D image (signal).
        img2 (numpy.ndarray): Reconstructed 2D image.

    Returns:
        tuple: (SNR, PSNR) in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError("The two images must have the same shape.")
    
    # Ensure floating-point
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    # Signal and noise
    signal_power = np.mean(img1 ** 2)
    noise_power = np.mean((img1 - img2) ** 2)

    if noise_power == 0:
        return np.inf, np.inf  # perfect reconstruction

    # SNR
    snr = 10 * np.log10(signal_power / noise_power)

    # PSNR
    dr = float(img1.max() - img2.min()) or 1.0
    psnr_val = psnr(img1, img2, data_range=dr)

    return snr, psnr_val

def compute_2d_ssim(img1, img2):
    """
    Compute SSIM between two 2D images.

    Parameters:
        img1 (numpy.ndarray): Original 2D image (reference).
        img2 (numpy.ndarray): Reconstructed 2D image.

    Returns:
        float: SSIM value.
    """
    if img1.shape != img2.shape:
        raise ValueError("The two images must have the same shape.")

    return ssim(img1, img2, data_range=img1.max() - img1.min())

def FrechetDistances(x,y):
    #Calculate mean and standard deviation on x
    mean_x , std_dev_x = np.mean(x), np.std(x)

    #Calculate mean and standard deviation on y
    mean_y , std_dev_y = np.mean(y), np.std(y)

    FD = np.sqrt((mean_x - mean_y) ** 2 + (std_dev_x - std_dev_y) ** 2)


    return FD