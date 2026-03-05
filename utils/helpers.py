import csv
from collections import OrderedDict

import math
import wandb
from timm.layers import resample_abs_pos_embed


def extract_gaug_metrics_and_reset(model, model_name, distributed):
    m = model.module if distributed else model
    stores, keys = list(), list()
    if model_name == 'locatswin':
        stages = m.layers
    else:
        stages = [m]
    try:
        l = 0
        for stage in stages:
            for block in stage.blocks:
                gaug = block.attn.gaug
                store = getattr(gaug, 'metrics_store', None)
                if store:
                    keys = getattr(gaug, 'metrics_keys', ['alpha', 'sigma'])
                    stores.append(([store[k].avg for k in keys], l))
                    gaug.reset_metrics_store()
                l += 1
    except Exception:
        pass
    return stores, keys

def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd, step=epoch)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

# Assuming a square grid
def maybe_resize_pos_embed(state_dict, model):
    if hasattr(model, 'pos_embed') and model.pos_embed.shape != state_dict['pos_embed'].shape:
        old_side, new_side = [int(math.sqrt(pe.shape[1] - model.num_prefix_tokens)) for pe in [state_dict['pos_embed'], model.pos_embed]]
        state_dict['pos_embed'] = resample_abs_pos_embed(
            state_dict['pos_embed'],
            new_size=[new_side] * 2,
            old_size=[old_side] * 2,
            num_prefix_tokens=model.num_prefix_tokens,
        )
    return state_dict
