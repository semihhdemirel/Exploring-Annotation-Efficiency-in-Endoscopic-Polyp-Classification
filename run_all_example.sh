#!/usr/bin/env bash
# End-to-end example. Edit the three paths, then run:  bash run_all_example.sh
set -euo pipefail

DATA=/path/to/k_fold_dataset          # fold1..fold5, each with train/ val/ test/
CKPT=./checkpoints
RES=./results

# 1) Supervised
python -m supervised.train    --data_root "$DATA" --ckpt_root "$CKPT/supervised" --archs resnet50 efficientb0 ViT
python -m supervised.evaluate --data_root "$DATA" --ckpt_root "$CKPT/supervised" --out_dir "$RES/supervised" \
                              --archs resnet50 efficientb0 ViT --benchmark

# 2) Self-supervised (one method per call)
for M in simclr moco simsiam mae nnclr simmim byol dinov2; do
  python -m selfsupervised.train    --model_type $M --data_path "$DATA" --checkpoint_dir "$CKPT/selfsupervised"
  python -m selfsupervised.evaluate --model_type $M --data_path "$DATA" --checkpoint_dir "$CKPT/selfsupervised" \
                                    --output_dir "$RES/selfsupervised" --eval_all --use_best --save_plots
done

# 3) Semi-supervised (needs a USB checkout; see README)
# python -m semisupervised.run_usb_kfold --usb_root /path/to/Semi-supervised-learning \
#     --base_config /path/to/Semi-supervised-learning/config/base_config.yaml \
#     --data_root "$DATA" --save_root "$CKPT/usb"
# python -m semisupervised.evaluate_usb --base_model_dir "$CKPT/usb" --base_data_dir "$DATA" \
#     --num_labels 490 --ulb_num_labels 49600
