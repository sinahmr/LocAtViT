import argparse
from pathlib import Path
import sys

import torch
import yaml
from hbird.hbird_eval import hbird_evaluation
from timm.models import create_model, load_checkpoint

# Ensure models is importable (project root is 2 levels up from this file)
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models import *

DEVICE = f'cuda:0'
INPUT_SIZE = 224  # Images are resized to this size, so no need to set strict_img_size and dynamic_img_pad
BATCH_SIZE = 128
PATCH_SIZE = 16


def evaluate(model_path, datasets):
    for dataset_name in datasets.keys():
        run_args = yaml.safe_load((model_path.parent / 'args.yaml').read_text())
        model_name = run_args['model']
        output_index = 2 if 'swin' in model_name else 11

        model = create_model(
            model_name,
            pretrained=False,
            num_classes=run_args['num_classes'],
            drop_rate=0,
            features_only=False,
            **run_args['model_kwargs']
        ).to(DEVICE)

        incompatible_keys = load_checkpoint(model, model_path.as_posix(),
                                            use_ema=False, strict=True, filter_fn=None)
        print('Incompatible_keys: ', incompatible_keys)
        print('Encoder weights loaded successfully!')
        model.eval()

        def extract_model_features(model, imgs):
            features = model.forward_intermediates(
                imgs,
                indices=[output_index],
                norm=False,
                output_fmt='NCHW',
                intermediates_only=True,
            )[0]
            return features.flatten(2, 3).permute(0, 2, 1), None

        f = extract_model_features(model, torch.randn(2, 3, 224, 224).cuda())
        embed_dim = f[0].shape[-1]

        hbird_miou = hbird_evaluation(
            model.to(DEVICE),
            d_model=embed_dim,  # size of the embedding feature vectors of patches
            patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE,
            input_size=INPUT_SIZE,
            augmentation_epoch=1, # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
            device=DEVICE,
            return_knn_details=False,  # whether to return additional NNs details
            n_neighbours=30,  # the number of neighbors to fetch per image patch
            nn_method='faiss', # options: faiss or scann as the k-nn library to be used, scann uses cpu, faiss gpu
            nn_params=None,  # Other parameters to be used for the k-NN operator
            ftr_extr_fn=extract_model_features, # function that extracts image patch features with a vision encoder
            dataset_name=dataset_name, # the name of the dataset to use
            data_dir=datasets[dataset_name],  # path to the dataset to use for evaluation
            memory_size=None, # How much you want to limit your datasetNone if to be left unbounded
            train_fs_path=None, # The path to the file with the subset of filenames for training
            val_fs_path=None, # The path to the file with the subset of filenames for validation
        )

        print('==' * 20)
        print(f'mIoU score: {hbird_miou}\t|\tDataset: {dataset_name}\t|\tModel: {model_path}')
        print('==' * 20 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path, help='Path to the model checkpoint directory')
    parser.add_argument('--voc-path', type=str, default=None, help='Path to VOC segmentation dataset')
    parser.add_argument('--ade-path', type=str, default=None, help='Path to ADE20k dataset')
    args = parser.parse_args()

    datasets = dict()
    if args.voc_path:
        datasets['voc'] = args.voc_path
    if args.ade_path:
        datasets['ade20k'] = args.ade_path

    evaluate(args.model_path, datasets)
