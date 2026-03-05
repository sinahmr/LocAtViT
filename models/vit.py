from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, LayerNorm
from timm.models import register_model, generate_default_cfgs, build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer, Attention, Block, LayerScale, maybe_add_mask, \
    checkpoint_filter_fn, _cfg

from models.locat import PRR, GaussianAugment

__all__ = ['LocAtViT']


class GaussianAugmentedAttention(Attention):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(dim, **kwargs)
        self.gaug = GaussianAugment(dim, self.num_heads)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        addition = maybe_add_mask(self.gaug(q), attn_mask)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=addition,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = torch.softmax(attn + addition, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GaussianAugmentedBlock(Block):
    def __init__(
            self,
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
    ) -> None:
        nn.Module.__init__(self)
        self.norm1 = norm_layer(dim)
        self.attn = GaussianAugmentedAttention(
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


class LocAtViT(VisionTransformer):
    def __init__(
            self, *args,
            gaug: bool = True, prr: bool = True, no_pos_emb: bool = False, store_metrics: bool = True,
            **kwargs
    ):
        self.pos_embed_type = 'none' if no_pos_emb else 'learn'
        block_fn = GaussianAugmentedBlock if gaug else Block
        super().__init__(*args, block_fn=block_fn, pos_embed=self.pos_embed_type, **kwargs)
        self.current_grid_size = self.initial_grid_size = self.patch_embed.grid_size

        if gaug:
            for blk in self.blocks:
                blk.attn.gaug.command(initial_grid_size=self.initial_grid_size,
                                      num_prefix_tokens=self.num_prefix_tokens, store_metrics=store_metrics)

        self.prr = PRR(kwargs.get('embed_dim'), kwargs.get('num_heads')) if prr else nn.Identity()

    def disable_gaussian_clamp(self):
        for blk in self.blocks:
            blk.attn.gaug.disable_clamp = True

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            self.propagate_grid_size(tuple(x.size()[1:-1]))
        if self.pos_embed_type == 'none':
            if self.dynamic_img_size:
                x = x.flatten(1, 2)
            to_cat = list()
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            return x
        else:
            return super()._pos_embed(x)

    def propagate_grid_size(self, grid_size):
        if self.current_grid_size == grid_size:
            return
        self.current_grid_size = grid_size
        for blk in self.blocks:
            if hasattr(blk.attn, 'gaug'):
                blk.attn.gaug.set_grid_size(grid_size)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.prr(x)
        x = self.forward_head(x)
        return x

def _create_locatvit(variant: str, pretrained: bool, model_args: dict, **kwargs):
    return build_model_with_cfg(
        LocAtViT,
        variant,
        pretrained=pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=kwargs.pop('out_indices', 3), feature_cls='getter'),
        **dict(model_args, **kwargs),
    )

default_cfgs = generate_default_cfgs({
    'locatvit_tiny': _cfg(),
    'locatvit_small': _cfg(),
    'locatvit_base': _cfg(),
})

@register_model
def locatvit_tiny(pretrained: bool = False, **kwargs) -> LocAtViT:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return _create_locatvit('locatvit_tiny', pretrained, model_args, **kwargs)

@register_model
def locatvit_small(pretrained: bool = False, **kwargs) -> LocAtViT:
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    return _create_locatvit('locatvit_small', pretrained, model_args, **kwargs)

@register_model
def locatvit_base(pretrained: bool = False, **kwargs) -> LocAtViT:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return _create_locatvit('locatvit_base', pretrained, model_args, **kwargs)

