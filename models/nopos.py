import torch
from timm.models import build_model_with_cfg, generate_default_cfgs, register_model
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn, _cfg

__all__ = ['NoPosVisionTransformer']


def build_model(model_cls, variant, pretrained, model_args, **kwargs):
    return build_model_with_cfg(
        model_cls,
        variant,
        pretrained=pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=kwargs.pop('out_indices', 3), feature_cls='getter'),
        **dict(model_args, **kwargs),
    )


# Not using timm's with pos_embed='none' since it has a bug in the _pos_embed function, it doesn't use the CLS token
class NoPosVisionTransformer(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, pos_embed='none', **kwargs)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
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


default_cfgs = generate_default_cfgs({
    'nopos_vit_tiny_patch16_224': _cfg(),
    'nopos_vit_base_patch16_224': _cfg(),
})

@register_model
def nopos_vit_base_patch16_224(pretrained: bool = False, **kwargs) -> NoPosVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return build_model(NoPosVisionTransformer, 'nopos_vit_base_patch16_224', pretrained, model_args, **kwargs)

@register_model
def nopos_vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> NoPosVisionTransformer:
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return build_model(NoPosVisionTransformer, 'nopos_vit_tiny_patch16_224', pretrained, model_args, **kwargs)

