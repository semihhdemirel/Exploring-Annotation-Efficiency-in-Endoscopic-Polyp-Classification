"""Evaluate supervised checkpoints on train / val / test splits.

For every (arch, fold, checkpoint, split) this writes
    <ckpt_root>/<arch>/fold<k>/<arch>_fold<k>_<ckpt>_<split>_metrics.txt
and appends one row per combination to <out_dir>/supervised_summary.csv.

Example
-------
    python -m supervised.evaluate --data_root /data/k_fold --ckpt_root ./checkpoints/supervised \
        --archs resnet50 efficientb0 ViT --folds 1 2 3 4 5 --splits test --benchmark
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from common.metrics import benchmark_forward, compute_metrics, format_metrics
from supervised.models import ARCHITECTURES, build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt_root", default="./checkpoints/supervised")
    p.add_argument("--out_dir", default="./results/supervised")
    p.add_argument("--archs", nargs="+", default=["resnet50", "efficientb0", "ViT"], choices=ARCHITECTURES)
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--ckpts", nargs="+", default=["best", "last"], choices=["best", "last"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--specificity_average", choices=["weighted", "macro"], default="weighted",
                   help="Overall specificity averaging (weighted by class support, or unweighted mean)")
    p.add_argument("--benchmark", action="store_true", help="Also time the forward pass (test split)")
    p.add_argument("--benchmark_batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--benchmark_max_images", type=int, default=512)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        y_pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
        y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.Resize((args.image_size,) * 2), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "supervised_summary.csv")
    rows = []

    for arch in args.archs:
        for fold in args.folds:
            fold_name = f"fold{fold}"
            ckpt_dir = os.path.join(args.ckpt_root, arch, fold_name)
            for ckpt_kind in args.ckpts:
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_kind}_{arch}.pth")
                if not os.path.isfile(ckpt_path):
                    print(f"[skip] missing checkpoint: {ckpt_path}")
                    continue
                ck = torch.load(ckpt_path, map_location=device, weights_only=False)
                state = ck["model_state_dict"] if "model_state_dict" in ck else ck
                class_names = ck.get("classes") if isinstance(ck, dict) else None

                for split in args.splits:
                    split_dir = os.path.join(args.data_root, fold_name, split)
                    if not os.path.isdir(split_dir):
                        print(f"[skip] missing split: {split_dir}")
                        continue
                    ds = datasets.ImageFolder(split_dir, transform=tf)
                    class_names = class_names or ds.classes
                    model = build_model(arch, len(class_names), pretrained=False).to(device)
                    model.load_state_dict(state)
                    model.eval()

                    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                        pin_memory=device.type == "cuda")
                    y_true, y_pred = predict(model, loader, device)
                    res = compute_metrics(y_true, y_pred, class_names, args.n_bootstrap, args.specificity_average)
                    title = f"{arch} | {fold_name} | {ckpt_kind} | {split}"
                    block = format_metrics(res, class_names, title)
                    print(block)

                    stem = f"{arch}_{fold_name}_{ckpt_kind}_{split}"
                    with open(os.path.join(ckpt_dir, f"{stem}_metrics.txt"), "w", encoding="utf-8") as f:
                        f.write(block + "\n")

                    row = {"arch": arch, "fold": fold, "ckpt": ckpt_kind, "split": split, "n": res["n_samples"]}
                    for k, v in res["overall"].items():
                        row[k] = v["value"]; row[f"{k}_ci_lo"] = v["ci"][0]; row[f"{k}_ci_hi"] = v["ci"][1]
                    rows.append(row)

                    if args.benchmark and split == "test":
                        n = min(args.benchmark_max_images, len(ds))
                        imgs = torch.stack([ds[i][0] for i in range(n)])
                        timing = benchmark_forward(model, imgs, args.benchmark_batch_sizes, device)
                        params = sum(p.numel() for p in model.parameters())
                        with open(os.path.join(ckpt_dir, f"{stem}_inference_time.json"), "w") as f:
                            json.dump({"device": str(device), "params": params,
                                       "n_images": n, "timing": timing}, f, indent=2)
                        for bs, t in timing.items():
                            print(f"  bs={bs:<3} {t['ms_per_image']:.3f} ms/img  {t['throughput_img_s']:.1f} img/s")

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
