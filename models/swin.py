from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import PatchEmbed, Mlp, DropPath, ClassifierHead, to_2tuple, to_ntuple
from timm.models import build_model_with_cfg
from timm.models import generate_default_cfgs, register_model
from timm.models.swin_transformer import SwinTransformerBlock, PatchMerging, SwinTransformerStage, SwinTransformer, \
    WindowAttention, checkpoint_filter_fn, _cfg

from models.locat import GaussianAugment, PRR

__all__ = ['LocAtSwin']

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


class LocAtWindowAttention(WindowAttention):
    def __init__(self, dim: int, num_heads: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.gaug = GaussianAugment(dim, num_heads)
        self.gaug.command(initial_grid_size=self.window_size, num_prefix_tokens=0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        addition = self.gaug(q)
        if self.fused_attn:
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            attn_mask = attn_mask + addition
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = attn + addition
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocAtSwinBlock(SwinTransformerBlock):
    def __init__(self,
                dim: int,
                input_resolution: _int_or_tuple_2_t,
                num_heads: int = 4,
                head_dim: Optional[int] = None,
                window_size: _int_or_tuple_2_t = 7,
                shift_size: int = 0,
                always_partition: bool = False,
                dynamic_mask: bool = False,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                proj_drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = nn.LayerNorm,
    ):
        nn.Module.__init__(self)
        self.dim = dim
        self.input_resolution = input_resolution
        self.target_shift_size = to_2tuple(shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = LocAtWindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )


class LocAtSwinStage(SwinTransformerStage):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm,
    ):
        nn.Module.__init__(self)
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            LocAtSwinBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])


class LocAtSwin(SwinTransformer):
    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            always_partition: bool = False,
            strict_img_size: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Union[str, Callable] = nn.LayerNorm,
            weight_init: str = '',
    ):
        nn.Module.__init__(self)
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
            output_fmt='NHWC',
        )
        patch_grid = self.patch_embed.grid_size

        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        if not isinstance(window_size, (list, tuple)):
            window_size = to_ntuple(self.num_layers)(window_size)
        elif len(window_size) == 2:
            window_size = (window_size,) * self.num_layers
        assert len(window_size) == self.num_layers
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [LocAtSwinStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    patch_grid[0] // scale,
                    patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=patch_size * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)

        self.prr = PRR(embed_dim[-1], num_heads[-1])

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )
        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.prr(x)
        x = self.forward_head(x)
        return x


def _create_locatswin(variant: str, pretrained: bool = False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 3, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)
    kwargs.pop('store_metrics', None)

    model = build_model_with_cfg(
        LocAtSwin, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)

    return model


default_cfgs = generate_default_cfgs({
    'locatswin_tiny': _cfg(),
    'locatswin_base': _cfg(),
})

@register_model
def locatswin_tiny(pretrained=False, **kwargs) -> LocAtSwin:
    model_args = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    return _create_locatswin('locatswin_tiny', pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def locatswin_base(pretrained=False, **kwargs) -> LocAtSwin:
    model_args = dict(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_locatswin('locatswin_base', pretrained=pretrained, **dict(model_args, **kwargs))
