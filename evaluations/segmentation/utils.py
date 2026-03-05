import os

model_kwargs = {
    'vit': dict(dynamic_img_pad=True, dynamic_img_size=True),
    'swin': dict(strict_img_size=False), # Sets dynamic_img_pad=True for PatchEmbed later in TimmModels
    'jumbo': dict(dynamic_img_pad=True, dynamic_img_size=True),
}
decode_head_in_channels = {
    'vit':   {'tiny': [192, 192, 192, 192], 'base': [768, 768, 768, 768]},
    'swin':  {'tiny': [96, 192, 384, 768], 'base': [128, 256, 512, 1024]},
    'jumbo':   {'tiny': [192, 192, 192, 192], 'base': [768, 768, 768, 768]},
}
indices = {  # Should be iterable, int is wrong for our use case
    'vit':   (2, 5, 8, 11),
    'swin':  (0, 1, 2, 3),
    'jumbo': (2, 5, 8, 11),
}
featmap_strides = {
    'vit': (16, 16, 16, 16),
    'swin': (56, 28, 14, 7),
}
drop_path_rate = {'tiny': 0.1, 'small': 0.2, 'base': 0.4, 'large': 0.4}

def get_model_info():
    path = os.getenv('CHECKPOINT', '')
    if 'pth' not in path:
        model_name = path
        path, wandb_id = None, None
    else:
        model_name = path.split('/')[-2].split('-')[-2]
        wandb_id = path.split('/')[-2].split('-')[-3]
    return model_name, path, wandb_id

def get_model_type_and_size(model_name):
    model_type, size = None, None
    if 'jumbo' in model_name:
        model_type = 'jumbo'
    elif 'swin' in model_name:
        model_type = 'swin'
    else:
        model_type = 'vit'
    for s in ['tiny', 'small', 'base', 'large']:
        if s in model_name:
            size = s
    return model_type, size

def get_params(model_name, full=False, num_scales=4):
    model_type, size = get_model_type_and_size(model_name)
    frozen_stages = None if full else 'full'

    kwargs = model_kwargs.get(model_type, dict())
    if model_name.startswith('locat'):
        kwargs['store_metrics'] = False
    if frozen_stages != 'full':
        kwargs['drop_path_rate'] = drop_path_rate[size]

    ind = indices[model_type][(-num_scales if full else -1):]
    in_ch = decode_head_in_channels[model_type][size]
    in_ch = in_ch[-num_scales:] if full else in_ch[-1]

    return ind, frozen_stages, kwargs, in_ch
