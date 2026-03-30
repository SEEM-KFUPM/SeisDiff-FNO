import torch
from torch import nn
import torchvision.transforms.functional as FV
import torch.nn.functional as F

def sinc_interpolation(data: torch.Tensor, 
                              target_size: tuple = None,
                              scale_factor: float = None) -> torch.Tensor:
    """
    Sinc interpolation via zero-padding in Fourier domain.
    Input:  torch.Tensor (B, C, H, W) or (C, H, W) or (H, W)
    
    Args:
        target_size:  (H_out, W_out) — tamaño objetivo explícito
        scale_factor: float — factor de escala (e.g. 2.0, 0.5, 1.0)
        
    One of target_size or scale_factor must be provided.
    Output: same rank, spatial dims as specified
    """
    assert target_size is not None or scale_factor is not None, \
        "Must provide either target_size or scale_factor"

    original_ndim = data.ndim
    device = data.device
    dtype  = data.dtype

    # ── Extract 2D slice ──
    if data.ndim == 4:
        B, C, H, W = data.shape
        data_2d = data[0, 0]
    elif data.ndim == 3:
        C, H, W = data.shape
        data_2d = data[0]
    elif data.ndim == 2:
        H, W = data.shape
        data_2d = data
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    # ── Compute target dimensions ──
    if target_size is not None:
        H_out, W_out = target_size
    else:
        H_out = int(H * scale_factor)
        W_out = int(W * scale_factor)

    # ── Save statistics ──
    original_mean = data_2d.mean()
    original_std  = data_2d.std()

    # ── Forward FFT ──
    F = torch.fft.fft2(data_2d.to(torch.complex64))
    F_shifted = torch.fft.fftshift(F)

    # ── Pad or crop in frequency domain ──
    F_out = torch.zeros((H_out, W_out), dtype=torch.complex64, device=device)

    # How much of the original spectrum fits in the output
    H_copy = min(H, H_out)
    W_copy = min(W, W_out)

    # Source region (center of original spectrum)
    h_src_start = H // 2 - H_copy // 2
    w_src_start = W // 2 - W_copy // 2

    # Destination region (center of output spectrum)
    h_dst_start = H_out // 2 - H_copy // 2
    w_dst_start = W_out // 2 - W_copy // 2

    F_out[h_dst_start:h_dst_start + H_copy,
          w_dst_start:w_dst_start + W_copy] = \
    F_shifted[h_src_start:h_src_start + H_copy,
              w_src_start:w_src_start + W_copy]

    # ── Inverse FFT ──
    F_unshifted = torch.fft.ifftshift(F_out)
    data_interp = torch.fft.ifft2(F_unshifted).real

    # ── Rescale to preserve original statistics ──
    data_interp = data_interp - data_interp.mean()
    data_interp = data_interp / (data_interp.std() + 1e-8)
    data_interp = data_interp * original_std + original_mean

    # ── Restore original rank ──
    if original_ndim == 4:
        data_interp = data_interp.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        data_interp = data_interp.unsqueeze(0)

    return data_interp.to(dtype)

def space_to_depth(x, size=2):
    """
    Downsacle method that use the depth dimension to
    downscale the spatial dimensions
    Args:
        x (torch.Tensor): a tensor to downscale
        size (float): the scaling factor

    Returns:
        (torch.Tensor): new spatial downscale tensor
    """
    b, c, h, w = x.shape
    out_h = h // size
    out_w = w // size
    out_c = c * (size * size)
    x = x.reshape((-1, c, out_h, size, out_w, size))
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape((-1, out_c, out_h, out_w))
    return x


class SpaceToDepth(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, x):
    return space_to_depth(x, self.size)


class Residual(nn.Module):
  """
  Apply residual connection using an input function
  Args:
    func (function): a function to apply over the input
  """
  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x, *args, **kwargs):
    return x + self.func(x, *args, **kwargs)

def upsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.Conv2d(in_channels, out_channels, 3, padding=1)
  )
  return seq

def downsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      SpaceToDepth(2),
      nn.Conv2d(4 * in_channels, out_channels, 1)
  )
  return seq


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in x-direction
        self.modes2 = modes2  # Number of Fourier modes to keep in y-direction

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to a factor of e^(- something)
        x_ft = torch.fft.rfft2(x, dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Implementation of Weight Standardized Convolution for 2D.
    Reference: https://arxiv.org/abs/1903.10520
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight  # Shape: (out_channels, in_channels, kernel_h, kernel_w)

        # ✅ Compute mean & variance across (input channels, kernel height, kernel width)
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)  # (out_channels, 1, 1, 1)
        variance = weight.var(dim=[1, 2, 3], keepdim=True, correction=0)  # (out_channels, 1, 1, 1)

        # ✅ Normalize weight
        normalized_weight = (weight - mean) / torch.sqrt(variance + eps)

        # ✅ Use standard convolution with the normalized weight
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, modes=8):
        super().__init__()
        self.fno = SpectralConv2d(in_channels, out_channels, modes, modes)  # Fourier layers for 3D
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        self.linear = nn.Conv2d(in_channels, out_channels, 1)  # Local transformation

    def forward(self, x):
        x_fno = self.fno(x) + self.linear(x)  # Mix spectral and local features
        x_fno = self.norm(x_fno)
        return self.act(x_fno)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, time_emb_dim=None, groups=4, modes=8):
        super().__init__()
        
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * out_channels)
            )
        else:
            self.mlp = None
        
        self.extra_in_conv = Block(cond_channels, out_channels, groups)
        self.extra_in_FNO = FNOBlock(out_channels, out_channels, groups, modes)
        
        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = FNOBlock(out_channels, out_channels, groups, modes) 
        
        self.block3 = Block(out_channels, out_channels, groups)
        self.block4 = FNOBlock(out_channels, out_channels, groups, modes) 
        
        # self.dropout = nn.Dropout(p=0.1)
        self.dropout = nn.Identity()
        
        if in_channels == out_channels:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x , cond, time_emb=None):
        scale_shift = None
        
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(*time_emb.shape, 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)


        if cond is not None:
            # cond = FV.resize(
            #     cond,
            #     (x.shape[2], x.shape[3]),
            #     interpolation=FV.InterpolationMode.BILINEAR,
            # )
            cond = self.extra_in_conv(cond)
            cond = self.extra_in_FNO(cond)

        
        h = self.block1(x, scale_shift=scale_shift)  # First conv-based block
        h = self.block2(h)
        if cond is not None:
            h = h + cond
        
        h = self.block3(h, scale_shift=scale_shift)
        h = self.block4(h)  # Second block now uses FNO

        h = self.dropout(h)

        
        return h + self.res_conv(x)

class SinusodialPositionEmbedding(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(self, time_steps):
    positions = torch.unsqueeze(time_steps, 1)
    half_dim = self.embedding_dim // 2
    embeddings = torch.zeros((time_steps.shape[0], self.embedding_dim), device=time_steps.device)
    denominators = 10_000 ** (2 * torch.arange(self.embedding_dim // 2, device=time_steps.device) / self.embedding_dim)
    embeddings[:, 0::2] = torch.sin(positions/denominators)
    embeddings[:, 1::2] = torch.cos(positions/denominators)
    return embeddings



class DiffusionFNOUnet(nn.Module):
    def __init__(self, dim, cond_cha, init_dim=None, output_dim=None, dim_mults=(1, 2, 4, 8),modes=8, channels=3, resnet_block_groups=4):
        super().__init__()
        self.channels = channels
        self.cond_cha = cond_cha  # Conditional input channels
        
        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(self.channels, init_dim, 1)
        self.cond_proj = nn.Conv2d(self.cond_cha, init_dim, 1)  # Project condition to same space

        dims = [init_dim] + [m * dim for m in dim_mults]
        input_output_dims = list(zip(dims[:-1], dims[1:]))
        time_dim = 4 * dim  # time embedding

        
        self.time_mlp = nn.Sequential(
            SinusodialPositionEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Down layers
        self.down_layers = nn.ModuleList([])
        for ii, (dim_in, dim_out) in enumerate(input_output_dims, 1):
            is_last = ii == len(input_output_dims)
            self.down_layers.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    ResnetBlock(dim_in, dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    ResnetBlock(dim_in, dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ])
            )

        # Middle layers
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim,mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim,mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes)
        self.mid_block3 = ResnetBlock(mid_dim, mid_dim,mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes)

        # Up layers
        self.up_layers = nn.ModuleList([])
        for ii, (dim_in, dim_out) in enumerate(reversed(input_output_dims), 1):
            is_last = ii == len(input_output_dims)
            self.up_layers.append(
                nn.ModuleList([
                    ResnetBlock(dim_out + dim_in, dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    ResnetBlock(dim_out + dim_in, dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    ResnetBlock(dim_out + dim_in, dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes),
                    upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ])
            )
        self.output_dim = output_dim if output_dim is not None else channels
        self.final_res_block = ResnetBlock(2 * dim, dim, dim, time_emb_dim=time_dim, groups=resnet_block_groups, modes=modes)
        self.final_conv = nn.Conv2d(dim, self.output_dim, 1)

    def forward(self, x, cond, time):
        x = self.init_conv(x)
        cond = sinc_interpolation(cond,
                                      target_size=(x.shape[2],
                                                   x.shape[3])
                                     )
        cond = self.cond_proj(cond)  # Project conditional input
        init_result = x.clone()
        t = self.time_mlp(time)
        h = []
        # Downsample
        for block1, block2, block3, downsample_block in self.down_layers:
            x = block1(x,cond, t )
            h.append(x)
            x = block2(x, cond, t )
            h.append(x)
            x = block3(x, cond, t )
            h.append(x)
            x = downsample_block(x)
            cond = downsample_block(cond)

        # Bottleneck
        x = self.mid_block1(x, cond, t)
        x = self.mid_block2(x, cond, t)
        x = self.mid_block3(x, cond, t)
        
        # Upsample
        for block1, block2,block3, upsample_block in self.up_layers:
            skip1 = h.pop()
            x = torch.cat((x, skip1), dim=1)
            x = block1(x, cond, t)
            skip2 = h.pop()
            x = torch.cat((x, skip2), dim=1)
            x = block2(x, cond, t)
            skip3 = h.pop()
            x = torch.cat((x, skip3), dim=1)
            x = block3(x, cond, t)
            x = upsample_block(x)
            cond = upsample_block(cond)

        # Final output
        x = torch.cat((x, init_result), dim=1)
        x = self.final_res_block(x, cond, t)
        x = self.final_conv(x)
        pred_noise, pred_logvar = torch.chunk(x, 2, dim=1)
        return pred_noise, pred_logvar
