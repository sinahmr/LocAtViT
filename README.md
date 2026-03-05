# LocAtViT: Locality-Attending Vision Transformer

<div align="center">
<img src="./.assets/locatvit-illustration.png" width="25%">
<p></p>
</div>

Official implementation of [**Locality-Attending Vision Transformer**](https://openreview.net/forum?id=KvEjv5klWi) (ICLR 2026).

> **TL;DR:** Pretrain vision transformers so their patch representations transfer better to dense prediction (e.g., segmentation), without changing the pretraining objective.

<details>
<summary><b>Abstract</b></summary>

> Vision transformers have demonstrated remarkable success in classification by leveraging global self-attention to capture long-range dependencies. However, this same mechanism can obscure fine-grained spatial details crucial for tasks such as segmentation. In this work, we seek to enhance segmentation performance of vision transformers after standard image-level classification training. More specifically, we present a simple yet effective add-on that improves performance on segmentation tasks while retaining vision transformers' image-level recognition capabilities. In our approach, we modulate the self-attention with a learnable Gaussian kernel that biases the attention toward neighboring patches. We further refine the patch representations to learn better embeddings at patch positions. These modifications encourage tokens to focus on local surroundings and ensure meaningful representations at spatial positions, while still preserving the model's ability to incorporate global information. Experiments demonstrate the effectiveness of our modifications, evidenced by substantial segmentation gains on three benchmarks (e.g., over 6% and 4% on ADE20K for ViT Tiny and Base), without changing the training regime or sacrificing classification performance.
</details>

## &#x1F3AC; Getting Started

### &#x1F4E6; Environment Setup

We use **Python 3.10** and **PyTorch 2.4**. Install the core dependencies:

```bash
pip install -r requirements.txt
```

Evaluation tasks have separate environments:

| Task | Requirements |
|------|-------------|
| Segmentation | Follow [`evaluations/segmentation/env.sh`](evaluations/segmentation/env.sh) |
| DINO | Covered by the main `requirements.txt` |
| Hummingbird | `pip install -r evaluations/hummingbird/hbird_requirements.txt` |

### &#x1F5C2; Datasets

**ImageNet / Mini-ImageNet:**
Download the ImageNet dataset and extract it using `./data/extract_imagenet.sh`.
Mini-ImageNet can be downloaded from [Hugging Face](https://huggingface.co/datasets/timm/mini-imagenet).

**Segmentation datasets:**
Follow the [MMSeg data preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md), then update the `data_root` paths in the corresponding files in `evaluations/segmentation/configs/`.


## &#x1F3CB; Classification Training

Train on ImageNet and obtain the checkpoint used for downstream evaluations:

```bash
./train.sh {num_gpus} \
  --data-dir {data_path} \
  --num-classes {num_classes} \
  --model {model_name} \
  --batch-size {batch_size} \
  --lr {lr} \
  --experiment {exp_name} \
  --log-wandb
```

<details>
<summary>Examples</summary>

```bash
# LocAtViT-Base on ImageNet
./train.sh 4 --data-dir /path/to/imagenet --num-classes 1000 --model locatvit_base \
  --batch-size 256 --lr 1e-3 --experiment myexp --log-wandb

# LocAtViT-Tiny on mini-ImageNet
./train.sh 2 --data-dir /path/to/mini-imagenet --num-classes 100 --model locatvit_tiny \
  --batch-size 512 --lr 5e-4 --epochs 600 --warmup-epochs 120 --experiment myexp --log-wandb
```
</details>


## &#x1F9EA; Evaluations

### &#128444; Segmentation

```bash
./evaluations/segmentation/train.sh {checkpoint_path} {ade|pc|stf}
```

Dataset keys: `ade` (ADE20K), `pc` (PASCAL Context 59), `stf` (COCO-Stuff 164K).

### &#129429; DINO

Pretrain with DINO, then evaluate with linear probing or k-NN:

```bash
cd evaluations/dino

# Pretraining
./train.sh {num_gpus} --data_path {data_path} --output_dir {output_dir} \
  --epochs {epochs} --batch_size_per_gpu {batch_size} --arch {model_name}

# Linear evaluation
./eval_linear.sh {num_gpus} --data_path {data_path} --output_dir {output_dir} \
  --epochs {epochs} --batch_size_per_gpu {batch_size} --arch {model_name} \
  --pretrained_weights {checkpoint_path}

# k-NN evaluation
./eval_knn.sh --data_path {data_path} --arch {model_name} \
  --pretrained_weights {checkpoint_path}
```

### &#x1FAB6; Hummingbird

```bash
cd evaluations/hummingbird
python eval.py {checkpoint_path} --voc-path {voc_path} --ade-path {ade_path}
```


## &#x1F64F; Acknowledgments

This codebase builds upon several open-source projects. We thank the authors of
[timm](https://github.com/huggingface/pytorch-image-models),
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation),
[DINO](https://github.com/facebookresearch/dino),
[Open Hummingbird Evaluation](https://github.com/vpariza/open-hummingbird-eval),
[DaViT](https://github.com/dingmyu/davit), and
[GCViT](https://github.com/NVlabs/GCVit).
The illustration is generated using ChatGPT.


## &#x1F4DA; Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hajimiri2026locatvit,
  author    = {Hajimiri, Sina and Beizaee, Farzad and Shakeri, Fereshteh and Desrosiers, Christian and Ben Ayed, Ismail and Dolz, Jose},
  title     = {Locality-Attending Vision Transformer},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
