from mmseg.models.builder import BACKBONES
from timm.models import create_model
from timm.models import load_checkpoint

from evaluations.segmentation.utils import get_model_type_and_size

POTENTIAL_FROZEN_MODULES = ['cls_token', 'stem', 'patch_embed', 'pos_embed', 'fc_norm', 'head', 'head_drop', 'norm_pre', 'patch_drop', 'pos_drop', 'reg_token']


def get_layers(model):
    layers = None
    if hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'stages'):
        layers = model.stages
    elif hasattr(model, 'blocks'):
        layers = model.blocks
    else:
        if hasattr(model, 'layers_0'):
            layers = [model.layers_0, model.layers_1, model.layers_2, model.layers_3]
        elif hasattr(model, 'stages_0'):
            layers = [model.stages_0, model.stages_1, model.stages_2, model.stages_3]
        else:
            print('Layers could not be found')
    return layers

@BACKBONES.register_module()
def TimmModels(model_name, indices=-1, frozen_stages=None, pretrained_path=None, **kwargs):
    pretrained_from_hub = False
    model_type, _ = get_model_type_and_size(model_name)
    if model_type not in ['vit', 'swin', 'jumbo']:
        pretrained_from_hub = True
    model = create_model(model_name, pretrained=pretrained_from_hub, **kwargs)
    if model_type == 'swin':
        model.patch_embed.dynamic_img_pad = not model.patch_embed.strict_img_size  # Doesn't take this argument in init

    model.indices = indices
    model.frozen_stages = frozen_stages

    def new_freeze_stages():
        layers = get_layers(model)
        assert layers is not None or model.frozen_stages in [None, 'full']
        if model.frozen_stages is None:
            pass
        elif model.frozen_stages == 'full' or model.frozen_stages == len(layers):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            for module_type in POTENTIAL_FROZEN_MODULES:
                if model.frozen_stages >= 0 and hasattr(model, module_type) and getattr(model, module_type) is not None:
                    try:
                        getattr(model, module_type).eval()
                        for param in getattr(model, module_type).parameters():
                            param.requires_grad = False
                    except:
                        getattr(model, module_type).requires_grad = False

            if model.frozen_stages >= 1:
                for i in range(model.frozen_stages):
                    layers[i].eval()
                    for param in layers[i].parameters():
                        param.requires_grad = False

    model._freeze_stages = new_freeze_stages
    model._freeze_stages()

    def new_init_weights(pretrained=None):
        if pretrained:
            def filter_heads(state_dict, _):
                state_dict.pop('head.weight', None)
                state_dict.pop('head.bias', None)
                state_dict.pop('head.fc.weight', None)
                state_dict.pop('head.fc.bias', None)
                return state_dict
            print('Loading checkpoint...')
            incompatible = load_checkpoint(model, pretrained, strict=False, filter_fn=filter_heads)
            print('Incompatible keys: ', incompatible)
            print('Encoder weights loaded successfully!')

    if not pretrained_from_hub:
        model.init_weights = new_init_weights
        if pretrained_path is not None:
            model.init_weights(pretrained_path)

    def vit_forward(x):
        features = model.forward_intermediates(
            x,
            indices=model.indices,
            return_prefix_tokens=False,
            norm=False,
            output_fmt='NCHW',
            intermediates_only=True,
        )
        return features

    def swin_forward(x):
        features = model.forward_intermediates(
            x,
            indices=model.indices,
            output_fmt='NCHW',
            norm=False,
            intermediates_only=True,
        )
        return features

    if model_type in ['vit', 'jumbo']:
        model.forward = vit_forward
        model.forward_features = vit_forward
    elif model_type == 'swin':
        model.forward = swin_forward
        model.forward_features = swin_forward
    else:
        pass
    return model
