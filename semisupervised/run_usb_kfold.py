"""Launch USB (Unified Semi-supervised Benchmark) training for every
(algorithm, backbone, fold) combination.

This script does not implement any SSL algorithm itself: it writes one YAML
config per run (derived from your base USB config) and calls USB's own
`train.py` as a subprocess. Run it from anywhere; `--usb_root` must point to a
clone of https://github.com/microsoft/Semi-supervised-learning with the
dataset code for your k-fold data registered (see README).

Example
-------
    python -m semisupervised.run_usb_kfold \
        --usb_root ~/Semi-supervised-learning \
        --base_config ~/Semi-supervised-learning/config/base_config.yaml \
        --data_root /data/k_fold --save_root ~/Semi-supervised-learning/saved_models/usb_cv \
        --algorithms pseudolabel pimodel meanteacher --models vit_small_patch16_224 --folds 1 2 3 4 5
"""
import argparse
import copy
import os
import subprocess
import sys
import tempfile

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usb_root", default=".", help="Root of the USB repository (working dir for train.py)")
    p.add_argument("--base_config", required=True, help="Base USB YAML config shared by all runs")
    p.add_argument("--data_root", required=True, help="Folder containing fold1..foldK")
    p.add_argument("--save_root", required=True, help="Output root; runs go to <save_root>/<algo>_<model>/fold<k>")
    p.add_argument("--train_script", default="train.py", help="USB training entry point, relative to --usb_root")
    p.add_argument("--algorithms", nargs="+", default=["pseudolabel", "pimodel", "meanteacher"],
                   help="USB algorithm names (e.g. pseudolabel pimodel meanteacher fixmatch)")
    p.add_argument("--models", nargs="+", default=["vit_small_patch16_224"],
                   help="USB/timm backbone names (e.g. resnet50 vit_small_patch16_224)")
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--dry_run", action="store_true", help="Print the planned runs and exit")
    return p.parse_args()


def make_run_config(base, args, algorithm, model, fold):
    cfg = copy.deepcopy(base)
    cfg["algorithm"] = algorithm
    cfg["net"] = model
    cfg["save_dir"] = os.path.join(args.save_root, f"{algorithm}_{model}").replace("\\", "/")
    cfg["save_name"] = f"fold{fold}"
    cfg["data_dir"] = os.path.join(args.data_root, f"fold{fold}").replace("\\", "/")
    return cfg


def main():
    args = parse_args()
    if not os.path.isfile(args.base_config):
        sys.exit(f"Base config not found: {args.base_config}")
    with open(args.base_config) as f:
        base = yaml.safe_load(f)

    runs = [(a, m, k) for a in args.algorithms for m in args.models for k in args.folds]
    failed = []
    for i, (algorithm, model, fold) in enumerate(runs, 1):
        tag = f"[{algorithm} | {model} | fold{fold}]"
        print(f"\n{'=' * 60}\n  Run {i}/{len(runs)}: {tag}\n{'=' * 60}")
        if args.dry_run:
            continue
        cfg = make_run_config(base, args, algorithm, model, fold)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="usb_run_", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
        try:
            subprocess.run([sys.executable, args.train_script, "--c", tmp.name], check=True, cwd=args.usb_root)
            print(f"  finished: {tag}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {tag} (return code {e.returncode}); continuing")
            failed.append(tag)
        finally:
            os.unlink(tmp.name)

    print(f"\n{'=' * 60}\n  {len(runs) - len(failed)}/{len(runs)} runs succeeded")
    for t in failed:
        print(f"  failed: {t}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
