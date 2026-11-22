# Polyp classification: supervised vs. semi-supervised vs. self-supervised (k-fold)

Code accompanying the manuscript **Exploring Annotation Efficiency in Endoscopic Polyp Classification Across Multiple Learning Paradigms**.

Three training paradigms are benchmarked with the same 5-fold protocol and the same metric
definitions (overall and per-class precision / recall / F1 / specificity with 95% bootstrap CIs,
confusion matrices, inference time).

| Paradigm | Methods | Folder |
|---|---|---|
| Supervised | ResNet-50, EfficientNet-B0, ViT-B/32 | `supervised/` |
| Semi-supervised | Pseudo-Label, Pi-Model, Mean Teacher (via [USB](https://github.com/microsoft/Semi-supervised-learning)) | `semisupervised/` |
| Self-supervised | SimCLR, MAE, BYOL (via [Lightly](https://github.com/lightly-ai/lightly)) with k-NN / linear / fine-tune evaluation | `selfsupervised/` |

`common/metrics.py` holds the shared metric code (used by the supervised and semi-supervised pipelines).

## Data layout

```
<data_root>/
  fold1/  train/<class>/*.png   val/<class>/*.png   test/<class>/*.png
  fold2/  ...
  ...
  fold5/
```

Class folders are read alphabetically (`Normal` -> 0, `Polyp` -> 1). 

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Semi-supervised runs additionally need a USB checkout:

```bash
git clone https://github.com/microsoft/Semi-supervised-learning
pip install -e Semi-supervised-learning
```

## Usage

All commands are run from the repository root as modules (`python -m ...`). Every script accepts
`--help`. See `run_all_example.sh` for a complete example.

### Supervised
```bash
python -m supervised.train    --data_root DATA --ckpt_root checkpoints/supervised --archs resnet50 efficientb0 ViT
python -m supervised.evaluate --data_root DATA --ckpt_root checkpoints/supervised --out_dir results/supervised --benchmark
```
Writes `best_<arch>.pth` / `last_<arch>.pth` per fold, per-checkpoint metric reports, and
`results/supervised/supervised_summary.csv` (one row per arch x fold x checkpoint x split).

### Self-supervised
```bash
python -m selfsupervised.train    --model_type simclr --data_path DATA --checkpoint_dir checkpoints/selfsupervised
python -m selfsupervised.evaluate --model_type simclr --data_path DATA --checkpoint_dir checkpoints/selfsupervised \
                                  --output_dir results/selfsupervised --eval_all --use_best --save_plots
```
Model classes are expected in `selfsupervised/models.py`.

### Semi-supervised (USB)
```bash
python -m semisupervised.run_usb_kfold --usb_root PATH/TO/Semi-supervised-learning \
    --base_config PATH/TO/Semi-supervised-learning/config/base_config.yaml \
    --data_root DATA --save_root checkpoints/usb \
    --algorithms pseudolabel pimodel meanteacher --models vit_small_patch16_224
python -m semisupervised.evaluate_usb --base_model_dir checkpoints/usb --base_data_dir DATA \
    --num_labels 490 --ulb_num_labels 49600
```
`run_usb_kfold.py` only generates one YAML per run and calls USB's own `train.py`. 

## Metric definitions

* Overall precision / recall / F1 / specificity: **support-weighted** averages over classes.
* Specificity of a class: TN / (TN + FP), one-vs-rest.
* 95% CIs: percentile bootstrap, 1000 resamples, seed 42.
* `--specificity_average macro` switches overall specificity to the unweighted class mean.
* Inference time is forward-pass only on in-memory tensors (no image decoding or augmentation).

## Reproducibility notes

* Supervised training seeds Python, NumPy and PyTorch (`--seed`, default 42); GPU kernels may still be non-deterministic.
* Checkpoints are loaded with `weights_only=False`; only load checkpoints you trained yourself.
* Pretrained weights are downloaded on first use (torchvision / `efficientnet_pytorch`).

## License and citation

