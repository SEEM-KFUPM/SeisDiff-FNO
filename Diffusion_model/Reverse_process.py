import torch
from Diffusion_model.foward_process import sample_by_t
import numpy as np

@torch.no_grad()
def ddpm_sample_conditional_xct(
    model,
    image_size,
    num_timesteps,
    betas,
    cond,
    device='cpu',
    batch_size=1,
    x_t=None,
    Plot=False
):
    """
    Conditional sampling from an Improved DDPM model that predicts both ε and log(σ²).
    """
    C, H, W = image_size
    T = num_timesteps

    # Precompute schedules
    betas_t = betas.to(device)
    alphas_t = 1. - betas_t
    alphas_bar_t = torch.cumprod(alphas_t, dim=0).to(device)
    alphas_bar_t_minus_1 = torch.cat([torch.tensor([1.], device=device), alphas_bar_t[:-1]])
    one_over_sqrt_alphas_t = 1. / torch.sqrt(alphas_t)
    sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - alphas_bar_t)
    posterior_variance = (((1. - alphas_bar_t_minus_1) / (1. - alphas_bar_t)) * betas_t)

    # Start from Gaussian noise
    if x_t is None:
        x_t = torch.randn(batch_size,C, H, W).to(device)  
        # x_t = torch.randn(batch_size, H, W, device=device) 

    for t in reversed(range(T)):
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
        # Model predicts noise and log-variance
        if Plot==True:
            eps_theta, logvar_theta = model(x_t, cond, t_tensor, Plot)
            Plot=False
        
        eps_theta, v = model(x_t, cond, t_tensor)
        

        # Sample scalars for timestep t
        beta_t             = sample_by_t(betas_t,                  t_tensor, x_t.shape)
        one_over_sqrt_alpha = sample_by_t(one_over_sqrt_alphas_t,  t_tensor, x_t.shape)
        sqrt_1m_alpha_bar  = sample_by_t(sqrt_1_minus_alphas_bar_t, t_tensor, x_t.shape)
        posterior_var_t    = sample_by_t(posterior_variance,        t_tensor, x_t.shape)

        # Learned variance: interpolate between β̃_t (posterior) and β_t (upper bound) in log-space
        # as in Nichol & Dhariwal (2021), Eq. 15
        log_beta_t       = torch.log(beta_t.clamp(min=1e-20))
        log_post_var_t   = torch.log(posterior_var_t.clamp(min=1e-20))
        logvar_theta     = v * log_beta_t + (1. - v) * log_post_var_t
        sigma            = torch.exp(0.5 * logvar_theta)

        # Posterior mean μ_θ(x_t, t)
        mean = one_over_sqrt_alpha * (x_t - (beta_t / sqrt_1m_alpha_bar) * eps_theta)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + sigma * noise
        else:
            x_t = mean

    return x_t

def ddpm_sample_conditional_xtc(
    model,
    image_size,
    num_timesteps,
    betas,
    cond,
    device='cpu',
    batch_size=1,
    x_t=None,
    Plot=False
):
    """
    Conditional sampling from an Improved DDPM model that predicts both ε and v (variance interpolation weight).
    """
    C, H, W = image_size
    T = num_timesteps

    # Precompute schedules
    betas_t = betas.to(device)
    alphas_t = 1. - betas_t
    alphas_bar_t = torch.cumprod(alphas_t, dim=0).to(device)
    alphas_bar_t_minus_1 = torch.cat([torch.tensor([1.], device=device), alphas_bar_t[:-1]])
    one_over_sqrt_alphas_t = 1. / torch.sqrt(alphas_t)
    sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - alphas_bar_t)
    posterior_variance = ((1. - alphas_bar_t_minus_1) / (1. - alphas_bar_t)) * betas_t

    # Start from Gaussian noise
    if x_t is None:
        x_t = torch.randn(batch_size, C, H, W).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)

        # Forward pass (single)
        if Plot:
            output = model(x_t, t_tensor, cond, Plot)
            Plot = False
        else:
            output = model(x_t, t_tensor, cond)

        # eps_theta: predicted noise, v: variance interpolation weight in [0, 1]
        eps_theta = output[:, 0, :, :]
        v         = output[:, 1, :, :]

        # Sample scalars for timestep t
        beta_t             = sample_by_t(betas_t,                  t_tensor, x_t.shape)
        one_over_sqrt_alpha = sample_by_t(one_over_sqrt_alphas_t,  t_tensor, x_t.shape)
        sqrt_1m_alpha_bar  = sample_by_t(sqrt_1_minus_alphas_bar_t, t_tensor, x_t.shape)
        posterior_var_t    = sample_by_t(posterior_variance,        t_tensor, x_t.shape)

        # Learned variance: interpolate between β̃_t (posterior) and β_t (upper bound) in log-space
        # as in Nichol & Dhariwal (2021), Eq. 15
        log_beta_t       = torch.log(beta_t.clamp(min=1e-20))
        log_post_var_t   = torch.log(posterior_var_t.clamp(min=1e-20))
        logvar_theta     = v * log_beta_t + (1. - v) * log_post_var_t
        sigma            = torch.exp(0.5 * logvar_theta)

        # Posterior mean μ_θ(x_t, t)
        mean = one_over_sqrt_alpha * (x_t - (beta_t / sqrt_1m_alpha_bar) * eps_theta)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + sigma * noise
        else:
            x_t = mean

    return x_t

@torch.no_grad()
def ddim_sample_conditional(
    model, image_size, num_timesteps, betas, cond,
    device="cuda", batch_size=1, ddim_steps=50, eta=0.0, x_t=None):
    
    C, H, W = image_size
    betas = betas.to(device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)

    # choose subsampled time grid
    ts = torch.linspace(0, num_timesteps - 1, steps=ddim_steps, device=device).long()  # [0,...,T-1]
    # integrate from large t down to small t
    if x_t is None:
        x = torch.randn(batch_size,C, H, W, device=device)
        # x = torch.randn(batch_size, H, W, device=device)
    else:
        x = x_t.to(device)

    for i in range(ddim_steps - 1, -1, -1):
        t = ts[i]
        t_batch = torch.full((batch_size,), int(t), dtype=torch.long, device=device)

        eps_theta, _logvar, = model(x, cond, t_batch)[:2]  # ignore x0 head if present

        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[ts[i-1]] if i > 0 else torch.tensor(1.0, device=device)

        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_m_ab_t = torch.sqrt(1 - ab_t)

        # x0 from eps
        x0_pred = (x - sqrt_one_m_ab_t * eps_theta) / sqrt_ab_t

        # DDIM update
        sigma = eta * torch.sqrt(
            (1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)
        ) if i > 0 else torch.tensor(0.0, device=device)

        noise = torch.randn_like(x) if (i > 0 and eta > 0) else 0.0
        x = (
            torch.sqrt(ab_prev) * x0_pred
            + torch.sqrt(1 - ab_prev - sigma**2) * eps_theta
            + sigma * noise
        )

    return x0_pred