import os
from evaluations.segmentation.utils import get_model_info, get_params

DEBUG = os.getenv('DEBUG', False)
dataset = os.getenv('DATASET', '')
if dataset == 'ade':
    dataset_path = 'ade20k.py'
    num_classes = 150
elif dataset == 'pc':
    dataset_path = 'pascal_context_59.py'
    num_classes = 59
else:
    dataset_path = 'coco_stuff164k.py'
    num_classes = 171

_base_ = [dataset_path, 'default_runtime.py']

model_name, path, backbone_wandb_id = get_model_info()
indices, frozen_stages, kwargs, in_channels = get_params(model_name)

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TimmModels',
        model_name=model_name,
        indices=indices,
        frozen_stages=frozen_stages,
        pretrained_path=path,
        **kwargs
    ),
    decode_head=dict(
        type='LinearSegHead',
        in_channels=in_channels,
        in_index=-1,
        channels=512,
        num_classes=num_classes,
        dropout_ratio=0.,
        norm_cfg=dict(type='GN', num_groups=1),  # Equivalent to LayerNorm
        act_cfg=dict(type='GELU'),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

LR = 0.001
total_steps = 20000
optimizer = dict(type='AdamW', lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()

lr_config = dict(
    policy='OneCycle',
    max_lr=LR,
    total_steps=total_steps,
    pct_start=0.25,
    anneal_strategy='linear',
    div_factor=1e4,
    final_div_factor=1e4,
    by_epoch=False,
)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=total_steps)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=4000, metric='mIoU', by_epoch=False)

data = dict(samples_per_gpu=32, workers_per_gpu=8)

cfg_dict = {'backbone_wandb_id': backbone_wandb_id}
cfg_dict.update(lr_config)
cfg_dict.update(optimizer)
cfg_dict.update(model)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='MMSegWandbHook',
            by_epoch=False,
            interval=50,
            with_step=False,
            init_kwargs={
                'project': 'seg',
                'entity': 'locatvit',
                'name': f'{dataset}_{model_name}',
                'config': cfg_dict,
            },
            log_checkpoint=False,
            log_artifact=False,
            log_checkpoint_metadata=False,
            num_eval_images=0,
        )
    ] if not DEBUG else [dict(type='TextLoggerHook', by_epoch=False)]
)