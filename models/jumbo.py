"""
The code is based on https://github.com/antofuller/jumbo (eb4470d),
Modified based on descriptions in https://arxiv.org/abs/2502.15021v2.
"""
from typing import Optional, Type

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models import register_model, generate_default_cfgs, build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer, Attention, LayerNorm, LayerScale, DropPath, Mlp, Block, \
    _cfg

from models.locat import PRR
from models.vit import GaussianAugmentedAttention


class JumboBlock(Block):
    def __init__(
            self,
            J: int,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            neighbor_aware: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        # From ViT
        self.norm1 = norm_layer(dim)
        attn_class = GaussianAugmentedAttention if neighbor_aware else Attention
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Jumbo
        self.J = J
        self.jumbo_dim = int(J * dim)

        self.norm3 = nn.LayerNorm(self.jumbo_dim)
        self.jumbo_mlp = None  # Will be created and assigned by Jumbo, to accommodate layer sharing
        self.ls3 = LayerScale(self.jumbo_dim, init_values=init_values)
        self.drop_path3 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        x_cls = x[:, :self.J, :]  # (bsz, J, dim)
        x_cls = rearrange(x_cls, "b l d -> b (l d)")  # (bsz, J * dim)
        x_cls = x_cls + self.drop_path3(self.ls3(self.jumbo_mlp(self.norm3(x_cls))))

        x_patches = x[:, self.J:, :]  # (bsz, num_patches, dim)
        x_patches = x_patches + self.drop_path2(self.ls2(self.mlp(self.norm2(x_patches))))

        x_cls = rearrange(x_cls, "b (l d) -> b l d", d=x_patches.shape[-1])
        x = torch.cat([x_cls, x_patches], dim=1)
        return x


class Jumbo(VisionTransformer):
    """
    This class is used for both Jumbo (locat=False) and LocAtJumbo (locat=True).
    """
    def __init__(self, J: int = 6, jumbo_mlp_ratio: int = 4, locat: bool = False, **kwargs):
        def custom_block_fn(**block_kwargs):
            return JumboBlock(J, neighbor_aware=locat, **block_kwargs)

        kwargs["block_fn"] = custom_block_fn
        super().__init__(**kwargs)
        self.locat = locat
        self.J = J  # number of tokens to combine
        self.num_prefix_tokens = J
        self.cls_token = nn.Parameter(torch.zeros(1, self.J, self.embed_dim))
        self.head = nn.Linear(int(J * kwargs['embed_dim']), kwargs['num_classes'])
        self.norm = nn.LayerNorm(int(J * kwargs['embed_dim']))  # Norm of CLS
        self.norm_patches = nn.LayerNorm(kwargs['embed_dim']) if locat else nn.Identity()

        jumbo_dim = int(J * kwargs['embed_dim'])
        self.jumbo_mlp = Mlp(
            in_features=jumbo_dim,
            hidden_features=int(jumbo_dim * jumbo_mlp_ratio),
            act_layer=nn.GELU,
            drop=kwargs.get('proj_drop_rate', 0),
        )
        for block in self.blocks:
            block.jumbo_mlp = self.jumbo_mlp

        self.prr = PRR(kwargs['embed_dim'], kwargs['num_heads']) if locat else nn.Identity()
        self.current_grid_size = self.initial_grid_size = self.patch_embed.grid_size
        if locat:
            for blk in self.blocks:
                blk.attn.gaug.command(initial_grid_size=self.initial_grid_size, num_prefix_tokens=self.num_prefix_tokens)

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        torch.nn.init.constant_(self.head.weight, 0)  # Set weights to 0

        if kwargs['num_classes'] == 100:
            torch.nn.init.constant_(self.head.bias, -4.6)  # init at 1/100
        elif kwargs['num_classes'] == 1_000:
            torch.nn.init.constant_(self.head.bias, -6.9)  # init at 1/1_000
        elif kwargs['num_classes'] == 10_450:
            torch.nn.init.constant_(self.head.bias, -9.25)  # init at 1/10_450
        else:
            raise ValueError("num_classes should be 100, 1000 or 10450")

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.locat and self.dynamic_img_size:
            self.propagate_grid_size(tuple(x.size()[1:-1]))
        return super()._pos_embed(x)

    def propagate_grid_size(self, grid_size):
        if self.current_grid_size == grid_size or not self.locat:
            return
        self.current_grid_size = grid_size
        for blk in self.blocks:
            blk.attn.gaug.set_grid_size(grid_size)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)

        x_cls, x_patches = x[:, :self.J].flatten(1), x[:, self.J:]
        x_cls = self.norm(x_cls)
        x_cls = rearrange(x_cls, "b (l d) -> b l d", d=x_patches.shape[-1])
        x_patches = self.norm_patches(x_patches)
        x = torch.cat([x_cls, x_patches], dim=1)
        return x

    def forward_head(self, x):
        x_cls = x[:, :self.J].flatten(1)
        return self.head(x_cls)  # (bsz, num_classes)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.prr(x)
        x = self.forward_head(x)
        return x


def _create_jumbo(variant: str, pretrained: bool, model_args: dict, **kwargs):
    kwargs.pop('store_metrics', None)
    return build_model_with_cfg(
        Jumbo,
        variant,
        pretrained=pretrained,
        feature_cfg=dict(out_indices=kwargs.pop('out_indices', 3), feature_cls='getter'),
        **dict(model_args, **kwargs),
    )

default_cfgs = generate_default_cfgs({
    'jumbo_tiny': _cfg(),
    'jumbo_base': _cfg(),
    'locatjumbo_tiny': _cfg(),
    'locatjumbo_base': _cfg(),
})

@register_model
def jumbo_tiny(pretrained: bool = False, **kwargs) -> Jumbo:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                      init_values=1e-4, no_embed_class=True)
    return _create_jumbo('jumbo_tiny', pretrained, model_args, **kwargs)

@register_model
def jumbo_base(pretrained: bool = False, **kwargs) -> Jumbo:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                      init_values=1e-4, no_embed_class=True)
    return _create_jumbo('jumbo_base', pretrained, model_args, **kwargs)

@register_model
def locatjumbo_tiny(pretrained: bool = False, **kwargs) -> Jumbo:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                      init_values=1e-4, no_embed_class=True, locat=True)
    return _create_jumbo('locatjumbo_tiny', pretrained, model_args, **kwargs)

@register_model
def locatjumbo_base(pretrained: bool = False, **kwargs) -> Jumbo:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                      init_values=1e-4, no_embed_class=True, locat=True)
    return _create_jumbo('locatjumbo_base', pretrained, model_args, **kwargs)
