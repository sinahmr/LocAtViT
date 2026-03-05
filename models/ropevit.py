from typing import Optional, Union, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Mlp, DropPath, trunc_normal_, AttentionPoolLatent, RotaryEmbeddingCat, RotaryEmbeddingMixed, \
    to_2tuple, PatchDropoutWithIndices, PatchEmbed, LayerNorm, SwiGLU, GluMlp, AttentionRope, maybe_add_mask, \
    apply_rot_embed_cat
from timm.models import register_model, generate_default_cfgs, build_model_with_cfg
from timm.models.eva import Eva, EvaBlock, checkpoint_filter_fn, _cfg, _create_eva

from models.locat import PRR, GaussianAugment

__all__ = ['LocAtRoPEViT']


class GaussianAugmentedAttentionRope(AttentionRope):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(dim, **kwargs)
        self.gaug = GaussianAugment(dim, self.num_heads)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        addition = maybe_add_mask(self.gaug(q), attn_mask)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=addition,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = torch.softmax(attn + addition, dim=-1)

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GaussianAugmentedEvaBlock(EvaBlock):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            attn_type: str = 'eva',
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.norm1 = norm_layer(dim)
        assert attn_type == 'rope', 'Not implemented'
        self.attn = GaussianAugmentedAttentionRope(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


class LocAtRoPEViT(Eva):
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            attn_type: str = 'eva',
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            no_embed_class: bool = False,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            rope_mixed_mode: bool = False,
            rope_grid_offset: float = 0.,
            rope_grid_indexing: str = 'ij',
            rope_temperature: float = 10000.,
            use_post_norm: bool = False,
            use_pre_transformer_norm: bool = False,
            use_post_transformer_norm: Optional[bool] = None,
            use_fc_norm: Optional[bool] = None,
            attn_pool_num_heads: Optional[int] = None,
            attn_pool_mlp_ratio: Optional[float] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
    ):
        nn.Module.__init__(self)
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        # resolve norm / pool usage
        activate_pre_norm = use_pre_transformer_norm
        if use_fc_norm is not None:
            activate_fc_norm = use_fc_norm  # pass through if explicit
        else:
            activate_fc_norm = global_pool == 'avg'  # default on if avg pool used
        if use_post_transformer_norm is not None:
            activate_post_norm = use_post_transformer_norm  # pass through if explicit
        else:
            activate_post_norm = not activate_fc_norm  # default on if fc_norm isn't active

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            bias=not use_pre_transformer_norm,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        num_pos_tokens = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropoutWithIndices(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            if rope_mixed_mode:
                self.rope_mixed = True
                # Mixed mode to supports depth-dependent frequencies
                self.rope = RotaryEmbeddingMixed(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    temperature=rope_temperature,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    grid_indexing=rope_grid_indexing,
                )
            else:
                self.rope_mixed = False
                self.rope = RotaryEmbeddingCat(
                    dim=embed_dim // num_heads,
                    temperature=rope_temperature,
                    in_pixels=False,
                    feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                    ref_feat_shape=ref_feat_shape,
                    grid_offset=rope_grid_offset,
                    grid_indexing=rope_grid_indexing,
                )
        else:
            self.rope_mixed = False
            self.rope = None

        self.norm_pre = norm_layer(embed_dim) if activate_pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert not use_post_norm, 'Custom EvaBlockPostNorm not implemented'
        block_fn = GaussianAugmentedEvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                attn_type=attn_type,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        self.norm = norm_layer(embed_dim) if activate_post_norm else nn.Identity()

        ### Added ###
        self.current_grid_size = self.initial_grid_size = self.patch_embed.grid_size

        for blk in self.blocks:
            blk.attn.gaug.command(initial_grid_size=self.initial_grid_size, num_prefix_tokens=self.num_prefix_tokens)

        self.prr = PRR(embed_dim, num_heads)
        #############

        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=attn_pool_num_heads or num_heads,
                mlp_ratio=attn_pool_mlp_ratio or mlp_ratio,
                norm_layer=norm_layer,
                act_layer=nn.GELU,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if activate_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            self.propagate_grid_size(tuple(x.size()[1:-1]))
        return super()._pos_embed(x)

    def propagate_grid_size(self, grid_size):
        if self.current_grid_size == grid_size:
            return
        self.current_grid_size = grid_size
        for blk in self.blocks:
            blk.attn.gaug.set_grid_size(grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.prr(x)
        x = self.forward_head(x)
        return x

def _create_locatropevit(variant: str, pretrained: bool, model_args: dict, **kwargs):
    kwargs.pop('store_metrics', None)
    return build_model_with_cfg(
        LocAtRoPEViT,
        variant,
        pretrained=pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=kwargs.pop('out_indices', 3), feature_cls='getter'),
        **dict(model_args, **kwargs),
    )

default_cfgs = generate_default_cfgs({
    'locatvit_tiny_rope_mixed': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'locatvit_base_rope_mixed': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'vit_tiny_patch16_rope_mixed_224': _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
})

common_args = dict(
    mlp_ratio=4,
    qkv_bias=True,
    attn_type='rope',
    init_values=1e-5,
    class_token=True,
    global_pool='token',
    use_abs_pos_emb=False,
    use_rot_pos_emb=True,
    rope_grid_indexing='xy',
    rope_temperature=10.0,
    rope_mixed_mode=True,
)

@register_model
def locatvit_tiny_rope_mixed(pretrained: bool = False, **kwargs) -> LocAtRoPEViT:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **common_args)
    return _create_locatropevit('locatvit_tiny_rope_mixed', pretrained, model_args, **kwargs)

@register_model
def locatvit_base_rope_mixed(pretrained: bool = False, **kwargs) -> LocAtRoPEViT:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **common_args)
    return _create_locatropevit('locatvit_base_rope_mixed', pretrained, model_args, **kwargs)

@register_model
def vit_tiny_patch16_rope_mixed_224(pretrained: bool = False, **kwargs) -> Eva:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **common_args)
    model = _create_eva('vit_tiny_patch16_rope_mixed_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
