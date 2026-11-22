"""Fully supervised k-fold training (ImageNet-pretrained CNN / ViT baselines).

Expected data layout (torchvision ImageFolder, one directory per fold):

    <data_root>/fold1/{train,val,test}/<class_name>/*.png
    <data_root>/fold2/...

Example
-------
    python -m supervised.train --data_root /data/k_fold --ckpt_root ./checkpoints/supervised \
        --archs resnet50 efficientb0 ViT --folds 1 2 3 4 5

Outputs per (arch, fold) in <ckpt_root>/<arch>/fold<k>/ :
    best_<arch>.pth   lowest validation loss
    last_<arch>.pth   most recent epoch (used by --resume)
    <arch>_fold<k>_training_acc_loss_values.txt   per-epoch curves
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from common.metrics import specificity_overall
from supervised.models import ARCHITECTURES, build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Folder containing fold1..foldK")
    p.add_argument("--ckpt_root", default="./checkpoints/supervised")
    p.add_argument("--archs", nargs="+", default=["resnet50"], choices=ARCHITECTURES)
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_pretrained", action="store_true", help="Random init instead of ImageNet weights")
    p.add_argument("--resume", action="store_true", help="Resume each fold from last_<arch>.pth if present")
    p.add_argument("--early_stopping_patience", type=int, default=0,
                   help="Stop after N epochs without val-loss improvement (0 = disabled)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(size):
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ToTensor(), norm,
    ])
    val = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return train, val


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None, desc=""):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, preds_all = 0.0, [], []
    with torch.set_grad_enabled(training), tqdm(loader, unit="batch", desc=desc, leave=False) as bar:
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            labels_all.extend(y.cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())
            bar.set_postfix(loss=loss.item())
    return {
        "loss": total_loss / len(loader),
        "acc": accuracy_score(labels_all, preds_all),
        "f1": f1_score(labels_all, preds_all, average="weighted", zero_division=0),
        "recall": recall_score(labels_all, preds_all, average="weighted", zero_division=0),
        "precision": precision_score(labels_all, preds_all, average="weighted", zero_division=0),
        "specificity": specificity_overall(labels_all, preds_all, num_classes, "weighted"),
    }


def fmt(m):
    return " | ".join(f"{k}: {v:.4f}" for k, v in m.items())


def train_one(arch, fold, args, device):
    fold_name = f"fold{fold}"
    fold_dir = os.path.join(args.data_root, fold_name)
    train_tf, val_tf = get_transforms(args.image_size)
    train_set = datasets.ImageFolder(os.path.join(fold_dir, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(fold_dir, "val"), transform=val_tf)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    print(f"Classes: {train_set.classes} | train: {len(train_set)} | val: {len(val_set)}")

    n_cls = len(train_set.classes)
    model = build_model(arch, n_cls, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.ckpt_root, arch, fold_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, f"last_{arch}.pth")
    best_path = os.path.join(ckpt_dir, f"best_{arch}.pth")

    start_epoch, best_loss, best_acc = 0, float("inf"), 0.0
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_loss, best_acc = ck.get("best_loss", best_loss), ck.get("best_accuracy", best_acc)
        hist = ck.get("history", hist)
        print(f"Resumed {arch}/{fold_name} from epoch {start_epoch} (best val loss {best_loss:.4f})")

    stale = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[{arch} | {fold_name}] epoch {epoch + 1}/{args.epochs}")
        tr = run_epoch(model, train_loader, criterion, device, n_cls, optimizer, "train")
        va = run_epoch(model, val_loader, criterion, device, n_cls, None, "val")
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"  train -> {fmt(tr)}\n  val   -> {fmt(va)}\n  lr    -> {lr:.6f}")

        hist["train_loss"].append(round(tr["loss"], 5)); hist["train_acc"].append(round(tr["acc"], 5))
        hist["val_loss"].append(round(va["loss"], 5)); hist["val_acc"].append(round(va["acc"], 5))

        improved = va["loss"] < best_loss
        if improved:
            best_loss, best_acc, stale = va["loss"], va["acc"], 0
        else:
            stale += 1

        ck = {"epoch": epoch, "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "scheduler_state_dict": scheduler.state_dict(),
              "train_loss": tr["loss"], "train_accuracy": tr["acc"],
              "val_loss": va["loss"], "val_accuracy": va["acc"],
              "best_loss": best_loss, "best_accuracy": best_acc,
              "classes": train_set.classes, "arch": arch, "history": hist}
        torch.save(ck, last_path)
        if improved:
            torch.save(ck, best_path)
            print(f"  new best model (val loss {best_loss:.4f}, val acc {best_acc:.4f})")

        with open(os.path.join(ckpt_dir, f"{arch}_{fold_name}_training_acc_loss_values.txt"), "w") as f:
            f.write(f"Architecture: {arch} | Fold: {fold_name}\n"
                    f"Last epoch: {epoch + 1}/{args.epochs} | LR: {lr:.6f}\n"
                    f"Best val loss: {best_loss:.4f} | val acc at best: {best_acc:.4f}\n\n")
            for k, v in hist.items():
                f.write(f"{k} = {v}\n\n")

        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print(f"Early stopping: no improvement for {stale} epochs")
            break

    print(f"\nDone {arch}/{fold_name}: best val loss {best_loss:.4f} | val acc {best_acc:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    for arch in args.archs:
        for fold in args.folds:
            print("\n" + "=" * 70 + f"\nTRAIN {arch} / fold{fold}\n" + "=" * 70)
            train_one(arch, fold, args, device)
    print("\nAll training finished.")


if __name__ == "__main__":
    main()
