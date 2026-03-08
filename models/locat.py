from functools import lru_cache
from typing import Type, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn
from timm.utils import AverageMeter
from torch.jit import Final


# EPS is added to sigma in the denominator. Assuming sigma is close to zero, find the value of EPS to avoid overflows in
# the division in float16 (for both forward and backward path). 65000 (near max value of float16) is divided by 2 since
# sigma is 2D.
def get_eps(grid_size):
    eps = 1e-1
    nominator_abs_max = 0.5 * (max(grid_size) - 1) ** 2
    while nominator_abs_max / eps ** 2 > 65000 / 2:
        eps *= 2
    return eps


class GaussianAugment(nn.Module):
    fused_attn: Final[bool]

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.disable_clamp = False

        self.log_var = nn.Linear(self.head_dim, 2, bias=True)
        self.log_alpha = nn.Linear(self.head_dim, 1, bias=True)

        # The following variables are initialized using the command function
        self.initial_grid_size, self.current_grid_size, self.EPS, self.sigmoid_fn = [None] * 4
        self.fast_gaug = False
        self.num_prefix_tokens = 0
        self.metrics_store, self.metrics_keys = None, []

    def command(self, initial_grid_size: tuple, num_prefix_tokens: int, store_metrics: bool = True):
        self.initial_grid_size = self.current_grid_size = initial_grid_size
        self.EPS = get_eps(initial_grid_size)
        self.sigmoid_fn = self.get_sigmoid_fn(max(initial_grid_size) ** 2)
        if initial_grid_size[0] >= 32:
            self.fast_gaug = True

        self.num_prefix_tokens = num_prefix_tokens
        if store_metrics:
            self.metrics_keys = ['var', 'alpha']
            self.metrics_store = {k: AverageMeter() for k in self.metrics_keys}

    def set_grid_size(self, grid_size):
        self.current_grid_size = grid_size

    def capture_metrics(self, **kwargs):
        if self.metrics_store is None:
            return
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.detach().mean().item()
            self.metrics_store[k].update(v)

    def reset_metrics_store(self):
        if not self.metrics_store:
            return
        for _, metric in self.metrics_store.items():
            metric.reset()

    def pad_beginning(self, addition: torch.Tensor, value: float = 0.) -> torch.Tensor:
        if self.num_prefix_tokens == 0:
            return addition
        return F.pad(addition, pad=(self.num_prefix_tokens, 0, self.num_prefix_tokens, 0), mode='constant', value=value)

    @staticmethod
    def get_sigmoid_fn(coef: float):
        b = math.log(coef - 1) if coef > 2 else 0  # shift so that f(0) = 1
        def fn(x: torch.Tensor) -> torch.Tensor:
            return coef * torch.special.expit(x - b)
        return fn

    def get_var_and_alpha(self, q: torch.Tensor):
        var = self.sigmoid_fn(self.log_var(q))
        if self.current_grid_size != self.initial_grid_size:
            var = var * (min(self.current_grid_size) / min(self.initial_grid_size))
        self.capture_metrics(var=var)

        alpha = self.log_alpha(q)
        with torch.autocast(device_type='cuda', enabled=False):
            alpha = F.softplus(alpha)
        self.capture_metrics(alpha=alpha)

        return var, alpha

    def addition_2d(self, q: torch.Tensor, pad_value: float = 0.):
        var, alpha = self.get_var_and_alpha(q[..., self.num_prefix_tokens:, :])
        var = var.unsqueeze(3)
        d = self.gaussian_2d_numerator(self.current_grid_size, var.device, var.dtype)
        with torch.autocast(device_type='cuda', enabled=False):
            gaussian = (d / (var + self.EPS)).sum(dim=-1)
            gaussian = torch.exp(gaussian)
        addition = alpha * gaussian
        addition = self.pad_beginning(addition, value=pad_value)
        return addition

    def fast_addition_2d(self, q: torch.Tensor, pad_value: float = 0., rate: int = 2):
        if not self.training:
            rate = 1
        q = q[..., self.num_prefix_tokens:, :]
        b, n_head, n, c = q.shape
        q = q.view(b, n_head, self.current_grid_size[0] // rate, rate, self.current_grid_size[1] // rate, rate, c)
        q = q.mean(dim=(3, 5))

        var, alpha = self.get_var_and_alpha(q)
        var, alpha = var.unsqueeze(4).unsqueeze(5), alpha.unsqueeze(5)

        d = self.gaussian_2d_numerator(self.current_grid_size, var.device, var.dtype)
        d = d.view(1, 1, *self.current_grid_size, *self.current_grid_size, -1)[..., ::rate, ::rate, ::rate, ::rate, :]
        with torch.autocast(device_type='cuda', enabled=False):
            gaussian = (d / (var + self.EPS)).sum(dim=-1)
            gaussian = torch.exp(gaussian)
        addition = alpha * gaussian

        for dim in range(-4, 0):
            addition = addition.repeat_interleave(rate, dim=dim)
        addition = addition.flatten(4, 5).flatten(2, 3)

        addition = self.pad_beginning(addition, value=pad_value)
        return addition

    @lru_cache
    def gaussian_2d_numerator(self, grid_size: tuple, device: torch.device, dtype: torch.dtype):
        """
        Returns the Gaussian numerator based on `current_grid_size`.
        `initial_grid_size` is only used to clamp large values and ensure numerical stability of the Gaussian formula.
        """
        # Although grid_size is the same as self.current_grid_size, it should be given as input and not read from self
        # since lru_cache caches based on the argument. If read from self, lru_cache should be disabled.
        n0, n1 = grid_size
        i = (torch.stack((torch.meshgrid(torch.arange(n0), torch.arange(n1), indexing='ij')), dim=2)).to(device)
        d = (i.unsqueeze(0).unsqueeze(1) - i.unsqueeze(2).unsqueeze(3)) ** 2
        d = (-0.5 * d.to(dtype)).view(1, 1, n0 * n1, n0 * n1, 2)  # (b, h, N-1, N-1, 2)

        if not self.disable_clamp:  # AttentionExtract cannot handle clamp, so disable it for visualizations
            clamp_value = -0.5 * (max(self.initial_grid_size) - 1) ** 2
            clamp_value = max(clamp_value, -6550)  # Based on float16 and min EPS
            d[d < clamp_value] = clamp_value
        return d

    def forward(self, q: torch.Tensor) -> Union[torch.Tensor, int]:
        if self.fast_gaug:
            return self.fast_addition_2d(q)
        return self.addition_2d(q)


class PRR(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self, dim: int, num_heads: int, nchw: bool = False,
            pre_norm: bool = False, post_norm: bool = False, norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.fused_attn = use_fused_attn()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.nchw = nchw
        self.pre_norm = norm_layer(dim) if pre_norm else nn.Identity()
        self.post_norm = norm_layer(dim) if post_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        shape = x.shape
        if self.nchw:
            x = x.movedim(1, -1)
        x = self.pre_norm(x)
        x = x.flatten(1, -2)
        x = x.view(x.shape[0], x.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(x, x, x)
        else:
            attn = (x * self.scale) @ x.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)
            x = attn @ x
        x = x.permute(0, 2, 1, 3).flatten(2, 3)
        x = self.post_norm(x)
        if self.nchw:
            x = x.movedim(-1, 1)
        return x.reshape(shape)
