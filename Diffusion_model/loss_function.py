import torch
import torch.nn as nn
from Diffusion_model.foward_process import sample_q ,sample_by_t, cosine_schedule
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
from dataclasses import dataclass

@dataclass
class NoiseSchedule:
    betas:                    torch.Tensor
    alphas:                   torch.Tensor
    alphas_bar:               torch.Tensor
    alphas_bar_minus_1:       torch.Tensor
    sqrt_alphas:              torch.Tensor
    sqrt_alphas_bar:          torch.Tensor
    sqrt_1_minus_alphas_bar:  torch.Tensor
    posterior_variance:       torch.Tensor
    one_over_sqrt_alphas:     torch.Tensor

    def to(self, device):
        return NoiseSchedule(**{k: v.to(device) for k, v in self.__dict__.items()})


def build_noise_schedule(num_timesteps, device='cpu'):
    betas                   = cosine_schedule(num_timesteps).to(device)
    alphas                  = 1. - betas
    alphas_bar              = torch.cumprod(alphas, dim=0)
    alphas_bar_minus_1      = torch.cat((torch.tensor([0.], device=device), alphas_bar[:-1]))
    sqrt_alphas             = torch.sqrt(alphas)
    sqrt_alphas_bar         = torch.sqrt(alphas_bar)
    sqrt_1_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
    posterior_variance      = ((1. - alphas_bar_minus_1) / (1. - alphas_bar)) * betas
    one_over_sqrt_alphas    = 1. / sqrt_alphas

    return NoiseSchedule(
        betas                   = betas,
        alphas                  = alphas,
        alphas_bar              = alphas_bar,
        alphas_bar_minus_1      = alphas_bar_minus_1,
        sqrt_alphas             = sqrt_alphas,
        sqrt_alphas_bar         = sqrt_alphas_bar,
        sqrt_1_minus_alphas_bar = sqrt_1_minus_alphas_bar,
        posterior_variance      = posterior_variance,
        one_over_sqrt_alphas    = one_over_sqrt_alphas,
    )

def get_posterior_mean_variance(x0, x_t, t, schedule, device):
    """
    Compute q(x_{t-1} | x_t, x_0): the true posterior in DDPM.

    Args:
        x0:       clean signal                (B, C, L)
        x_t:      noisy signal at timestep t  (B, C, L)
        t:        timestep tensor             (B,)
        schedule: NoiseSchedule — precomputed once before training

    Returns:
        posterior_mean:               (B, C, L)
        posterior_log_variance_clipped: (B, 1, 1) broadcastable
    """
    # Sample scalars for each t and reshape for (B, C, L) broadcasting
    beta_t          = schedule.betas[t].view(-1, 1, 1)
    sqrt_alpha_t    = schedule.sqrt_alphas[t].view(-1, 1, 1)
    alpha_bar_t     = schedule.alphas_bar[t].view(-1, 1, 1)
    alpha_bar_tm1   = schedule.alphas_bar_minus_1[t].view(-1, 1, 1)
    posterior_var_t = schedule.posterior_variance[t].view(-1, 1, 1)

    # Posterior mean — DDPM Eq. 11
    coef_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1. - alpha_bar_t)
    coef_xt = (sqrt_alpha_t * (1. - alpha_bar_tm1)) / (1. - alpha_bar_t)
    posterior_mean = coef_x0 * x0 + coef_xt * x_t

    # Posterior log variance — clamp to avoid log(0)
    posterior_log_variance_clipped = torch.log(posterior_var_t.clamp(min=1e-20))

    return posterior_mean, posterior_log_variance_clipped

# def get_posterior_mean_variance(x0, x_t, t, num_timesteps,device):
#     """
#     Compute q(x_{t-1} | x_t, x_0): the true posterior in DDPM

#     Inputs:
#     - x0: clean data [B, C, H, W]
#     - x_t: noisy data at timestep t [B, C, H, W]
#     - t: timestep tensor [B]

#     Returns:
#     - posterior_mean: [B, C, H, W]
#     - posterior_log_variance_clipped: [B, 1, 1, 1] broadcastable
#     """

#     betas_t = cosine_schedule(num_timesteps)
    
#     alphas_t = 1. - betas_t
#     alphas_bar_t = torch.cumprod(alphas_t, dim=0)
#     alphas_bar_t_minus_1 = torch.cat((torch.tensor([0]), alphas_bar_t[:-1]))
#     one_over_sqrt_alphas_t = 1. / torch.sqrt(alphas_t)
#     sqrt_alphas_bar_t = torch.sqrt(alphas_bar_t)
#     sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - alphas_bar_t)
#     # the variance of q(xₜ₋₁ | xₜ, x₀) as in part 3
#     posterior_variance = (((1. - alphas_bar_t_minus_1) / (1. - alphas_bar_t)) * betas_t)
#     sqrt_alphas_t = torch.sqrt(alphas_t)
    
#     # Sample scalars for each t
#     beta_t = sample_by_t(betas_t, t, x_t.shape)
#     sqrt_alpha_t = sample_by_t(sqrt_alphas_t, t, x_t.shape)
#     sqrt_alpha_bar_t = sample_by_t(sqrt_alphas_bar_t, t, x_t.shape)
#     sqrt_one_minus_alpha_bar_t = sample_by_t(sqrt_1_minus_alphas_bar_t, t, x_t.shape)
#     alpha_bar_t = sample_by_t(alphas_bar_t, t, x_t.shape)
#     alpha_bar_tm1 = sample_by_t(alphas_bar_t_minus_1, t, x_t.shape)
#     posterior_var_t = sample_by_t(posterior_variance, t, x_t.shape)

#     # Posterior mean (Eq. 11 from DDPM paper)
#     coef_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / (1. - alpha_bar_t)
#     coef_xt = (sqrt_alpha_t * (1. - alpha_bar_tm1)) / (1. - alpha_bar_t)

#     posterior_mean = coef_x0 * x0 + coef_xt * x_t


#     # Posterior log variance (used in KL)
#     posterior_log_variance_clipped = torch.log(posterior_var_t)

#     return posterior_mean, posterior_log_variance_clipped

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def compute_loss(model, x0, condition, t, schedule, noise=None, device='cpu', lambda_vlb=0.001):
    """
    Args:
        model:     DiffusionUnet1d
        x0:        clean signal        (B, C, L)
        condition: conditioning signal (B, C, L)
        t:         timesteps           (B,)
        schedule:  NoiseSchedule — precomputed once before training
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # Forward diffusion
    x_t, w_t = sample_q(x0, t, schedule, noise)

    # Model prediction
    pred_noise, log_var = model(x_t, condition, t)

    # L_simple: weighted MSE on noise
    loss_fn = nn.MSELoss()
    L_simple = w_t * loss_fn(pred_noise, noise)

    # Predicted posterior mean (mu_theta)
    sqrt_alpha_t            = schedule.sqrt_alphas[t].view(-1, 1, 1)
    sqrt_one_minus_ab_t     = schedule.sqrt_1_minus_alphas_bar[t].view(-1, 1, 1)
    beta_t                  = schedule.betas[t].view(-1, 1, 1)

    mu_pred = (1.0 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_ab_t) * pred_noise)

    # True posterior (mu_q, logvar_q)
    mu_true, logvar_true = get_posterior_mean_variance(x0, x_t, t, schedule, device)

    # L_vlb: KL divergence
    L_vlb = normal_kl(mu_true, logvar_true, mu_pred, log_var).mean()
    
    # Final combined loss
    total_loss = L_simple + lambda_vlb * L_vlb 
    # print(f'L_simple = {L_simple}')
    # print(f'lambda_vlb * L_vlb = {lambda_vlb}*{L_vlb} = {lambda_vlb * L_vlb}')
    # print(f'lambda_ssim * L_ssim = {lambda_ssim} * {L_ssim} = {lambda_ssim * L_ssim }')
    # print()
    # print()
    return total_loss