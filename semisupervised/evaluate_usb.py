"""Evaluate semi-supervised (USB / semilearn) k-fold checkpoints.

Run this from an environment where the USB `semilearn` package is importable
(see README). For every (algorithm, model, fold) it evaluates the EMA weights
of model_best.pth (or latest_model.pth) on the labelled-train, val and test
splits and writes, next to the checkpoint:
    <split>_metrics.txt / <split>_metrics.png      metrics + CIs, ROC and confusion matrix
    <split>_inference_time.txt                     only with --inference true

NOTE: the "train_lb" split is the *labelled subset* used for SSL training
(--num_labels images), not the full training folder.

Example
-------
    python -m semisupervised.evaluate_usb --base_model_dir ./saved_models/usb_cv \
        --base_data_dir /data/k_fold --algorithms pseudolabel pimodel meanteacher \
        --models vit_small_patch16_224 --folds 1 2 3 4 5 --num_labels 490 --ulb_num_labels 49600
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from semilearn.core.utils import get_dataset, get_net_builder
from sklearn.metrics import (ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from common.metrics import compute_metrics, format_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y")


# --------------------------------------------------------------------------
# Inference benchmark: repeated forward passes of one fixed batch
# --------------------------------------------------------------------------
def benchmark_inference_time(net, dataset, save_dir, split_name, batch_sizes=(1, 2, 4, 8, 16, 32),
                             num_warmup=5, num_runs=20):
    lines = [f"Inference Time Benchmark - {split_name}", "=" * 60,
             f"  Warmup runs : {num_warmup}", f"  Timed runs  : {num_runs}",
             f"  Device      : {DEVICE}", ""]
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)

    for bs in batch_sizes:
        loader = DataLoader(dataset, batch_size=bs, drop_last=False, shuffle=False, num_workers=0)
        try:
            batch = next(iter(loader))
        except StopIteration:
            lines.append(f"  Batch size {bs:>2}: SKIPPED (dataset too small)")
            continue
        image = batch["x_lb"].float().to(DEVICE)
        with torch.no_grad():
            for _ in range(num_warmup):
                net(net(image, only_feat=True), only_fc=True)
            sync()
            lat = []
            for _ in range(num_runs):
                sync()
                t0 = time.perf_counter()
                net(net(image, only_feat=True), only_fc=True)
                sync()
                lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        actual = image.shape[0]
        lines += [f"  Batch size {bs:>2} (actual={actual})",
                  f"    Mean latency : {lat.mean():.3f} ms   Std: {lat.std():.3f} ms",
                  f"    Min / Max    : {lat.min():.3f} / {lat.max():.3f} ms",
                  f"    Per-sample   : {lat.mean() / actual:.3f} ms",
                  f"    Throughput   : {actual * 1000 / lat.mean():.1f} img/s", ""]
        print(lines[-6])
    with open(os.path.join(save_dir, f"{split_name}_inference_time.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Split evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(net, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
    labels, preds, probs = [], [], []
    for data in loader:
        image = data["x_lb"].float().to(DEVICE)
        prob = net(net(image, only_feat=True), only_fc=True).softmax(dim=-1)
        preds.append(prob.argmax(1).cpu().numpy())
        probs.append(prob.cpu().numpy())
        labels.append(data["y_lb"].cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)


def evaluate_split(net, dataset, batch_size, num_classes, save_dir, split_name, class_names,
                   n_bootstrap=1000, specificity_average="weighted"):
    y_true, y_pred, y_prob = predict(net, dataset, batch_size)
    is_binary = num_classes == 2
    res = compute_metrics(y_true, y_pred, class_names, n_bootstrap, specificity_average)

    pos = y_prob[:, 1] if is_binary else y_prob
    auc = roc_auc_score(y_true, pos, multi_class="ovr") if not is_binary else roc_auc_score(y_true, pos)

    block = format_metrics(res, class_names, f"Split: {split_name}") + f"\n  AUC: {auc:.4f}\n"
    print(block)

    with open(os.path.join(save_dir, f"{split_name}_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(block)
        f.write("\nROC curve data\n" + "=" * 60 + "\n")
        if is_binary:
            fpr, tpr, thr = roc_curve(y_true, pos)
            for a, b, c in zip(fpr, tpr, thr):
                f.write(f"  FPR={a:.4f}  TPR={b:.4f}  Threshold={c:.4f}\n")
        else:
            yb = label_binarize(y_true, classes=list(range(num_classes)))
            for i, cname in enumerate(class_names):
                fpr, tpr, thr = roc_curve(yb[:, i], y_prob[:, i])
                f.write(f"\n  {cname} (AUC={roc_auc_score(yb[:, i], y_prob[:, i]):.4f}):\n")
                for a, b, c in zip(fpr, tpr, thr):
                    f.write(f"    FPR={a:.4f}  TPR={b:.4f}  Threshold={c:.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, pos)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    else:
        yb = label_binarize(y_true, classes=list(range(num_classes)))
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"{cname} (AUC={roc_auc_score(yb[:, i], y_prob[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC - {split_name}")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ConfusionMatrixDisplay(res["confusion_matrix"], display_labels=class_names).plot(
        ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"Confusion matrix - {split_name}")
    o = res["overall"]
    plt.suptitle(f"[{split_name}] Acc={o['accuracy']['value']:.3f} Prec={o['precision']['value']:.3f} "
                 f"Rec={o['recall']['value']:.3f} F1={o['f1']['value']:.3f} AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{split_name}_metrics.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model_dir", required=True, help="Root with <algorithm>_<model>/fold<k>/ checkpoints")
    p.add_argument("--base_data_dir", required=True, help="Root containing fold1..foldK")
    p.add_argument("--algorithms", nargs="+", default=["pseudolabel", "pimodel", "meanteacher"])
    p.add_argument("--models", nargs="+", default=["vit_small_patch16_224"])
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--class_names", nargs="+", default=["Normal", "Polyp"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dataset", default="semi_inat", help="USB dataset name used to build the loaders")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--crop_ratio", type=float, default=0.875)
    p.add_argument("--num_labels", type=int, required=True, help="Labelled images used for SSL training")
    p.add_argument("--ulb_num_labels", type=int, required=True, help="Unlabelled images used for SSL training")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--specificity_average", choices=["weighted", "macro"], default="weighted")
    p.add_argument("--use_best_model", type=str2bool, default=True, help="model_best.pth vs latest_model.pth")
    p.add_argument("--inference", type=str2bool, default=False, help="Benchmark inference time")
    p.add_argument("--net_from_name", action="store_true")
    a = p.parse_args()
    a.num_classes = len(a.class_names)
    # Values semilearn's get_dataset expects on the args namespace
    a.max_length, a.max_length_seconds, a.sample_rate = 512, 4.0, 16000
    a.lb_imb_ratio = a.ulb_imb_ratio = 1
    a.seed, a.epoch, a.num_train_iter, a.save_name = 0, 1, 1024, ""
    return a


def main():
    args = parse_args()
    model_file = "model_best.pth" if args.use_best_model else "latest_model.pth"
    skipped = []

    for algorithm in args.algorithms:
        for model in args.models:
            for fold in args.folds:
                fold_name = f"fold{fold}"
                ckpt = os.path.join(args.base_model_dir, f"{algorithm}_{model}", fold_name, model_file)
                data_dir = os.path.join(args.base_data_dir, fold_name)
                print(f"\n{'#' * 70}\n  {algorithm} | {model} | {fold_name}\n  {ckpt}\n{'#' * 70}")
                if not os.path.isfile(ckpt):
                    skipped.append(f"checkpoint not found: {ckpt}"); print(f"[SKIP] {skipped[-1]}"); continue
                if not os.path.isdir(data_dir):
                    skipped.append(f"data dir not found: {data_dir}"); print(f"[SKIP] {skipped[-1]}"); continue

                state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["ema_model"]
                state = {(".".join(k.split(".")[1:]) if k.startswith("module") else k): v for k, v in state.items()}
                save_dir = os.path.dirname(ckpt)
                args.save_dir, args.data_dir = save_dir, data_dir

                net = get_net_builder(model, args.net_from_name)(num_classes=args.num_classes)
                net.load_state_dict(state)
                net.to(DEVICE).eval()

                ds = get_dataset(args, algorithm, args.dataset, args.num_labels, args.num_classes, data_dir, False)
                splits = {"train_lb": ds["train_lb"], "eval": ds["eval"], "test": ds["test"]}
                alias = {"train_lb": "train", "eval": "val", "test": "test"}

                if args.inference:
                    for key, d in splits.items():
                        benchmark_inference_time(net, d, save_dir, alias[key])
                for key, d in splits.items():
                    evaluate_split(net, d, args.batch_size, args.num_classes, save_dir, key,
                                   args.class_names, args.n_bootstrap, args.specificity_average)

                del net
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    if skipped:
        print(f"\nSkipped {len(skipped)} combination(s):")
        for s in skipped:
            print("  ", s)
    else:
        print("\nAll combinations evaluated.")


if __name__ == "__main__":
    main()
