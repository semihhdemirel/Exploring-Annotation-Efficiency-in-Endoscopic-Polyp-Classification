"""
Evaluate self-supervised checkpoints (k-fold) with three protocols:

  --eval_knn       frozen embeddings + k-NN (k chosen on the val split)
  --eval_linear    frozen encoder + trained linear classifier
  --eval_finetune  full fine-tuning of encoder + classifier
  --eval_all       all of the above

Reports overall and per-class metrics with 95% bootstrap CIs and inference time.

Example
-------
    python -m selfsupervised.evaluate --model_type simclr --data_path /data/k_fold \
        --checkpoint_dir ./checkpoints/selfsupervised --output_dir ./results/selfsupervised \
        --eval_all --use_best --save_plots

Model definitions live in selfsupervised/models.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from lightly.data import LightlyDataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              multilabel_confusion_matrix)
from sklearn.preprocessing import normalize
from scipy import stats
from tqdm import tqdm
import argparse
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Model classes (see selfsupervised/models.py)
from selfsupervised.models import (
    SimCLRModel,
    MoCoModel,
    SimSiamModel,
    MAEModel,
    NNCLRModel,
    SimMIMModel,
    BYOLModel,
    DINOv2Model,
    DINOModel  # Alias for compatibility
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Per-Class Metrics, CI, and Inference Timing Helpers
# ============================================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, **kwargs):
    """Compute bootstrapped confidence interval for a scalar metric."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx], **kwargs))
        except Exception:
            pass
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return lo, hi


def compute_per_class_metrics_with_ci(y_true, y_pred, class_names, n_bootstrap=1000):
    """
    Compute per-class precision, recall, f1, specificity with 95% CI.
    Returns a list of dicts, one per class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)

    # Multi-label confusion matrix gives TP, FP, FN, TN per class
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    results = []
    for c in range(n_classes):
        tn, fp, fn, tp = mcm[c].ravel()

        # Point estimates
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Bootstrap CI helpers
        def _prec(yt, yp, _c=c):
            return precision_score(yt, yp, labels=[_c], average='macro', zero_division=0)
        def _rec(yt, yp, _c=c):
            return recall_score(yt, yp, labels=[_c], average='macro', zero_division=0)
        def _f1(yt, yp, _c=c):
            return f1_score(yt, yp, labels=[_c], average='macro', zero_division=0)
        def _spec(yt, yp, _c=c):
            _mcm = multilabel_confusion_matrix(yt, yp)
            if _c >= len(_mcm):
                return 0.0
            _tn, _fp, _fn, _tp = _mcm[_c].ravel()
            return _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0

        prec_lo,  prec_hi  = bootstrap_ci(y_true, y_pred, _prec,  n_bootstrap)
        rec_lo,   rec_hi   = bootstrap_ci(y_true, y_pred, _rec,   n_bootstrap)
        f1_lo,    f1_hi    = bootstrap_ci(y_true, y_pred, _f1,    n_bootstrap)
        spec_lo,  spec_hi  = bootstrap_ci(y_true, y_pred, _spec,  n_bootstrap)

        results.append({
            'class': class_names[c],
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'precision': prec, 'precision_ci': (prec_lo, prec_hi),
            'recall':    rec,  'recall_ci':    (rec_lo,  rec_hi),
            'f1':        f1,   'f1_ci':        (f1_lo,   f1_hi),
            'specificity': spec, 'specificity_ci': (spec_lo, spec_hi),
        })
    return results


def compute_overall_metrics_with_ci(y_true, y_pred, n_bootstrap=1000):
    """Compute weighted overall metrics with 95% CI."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Weighted specificity
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    support = np.bincount(y_true, minlength=len(mcm))
    total = support.sum()
    specs = []
    for c in range(len(mcm)):
        tn, fp, fn, tp = mcm[c].ravel()
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    spec = float(np.average(specs, weights=support / total))

    def _acc(yt, yp):  return accuracy_score(yt, yp)
    def _prec(yt, yp): return precision_score(yt, yp, average='weighted', zero_division=0)
    def _rec(yt, yp):  return recall_score(yt, yp, average='weighted', zero_division=0)
    def _f1(yt, yp):   return f1_score(yt, yp, average='weighted', zero_division=0)
    def _spec(yt, yp):
        _mcm = multilabel_confusion_matrix(yt, yp)
        _sup = np.bincount(yt, minlength=len(_mcm))
        _tot = _sup.sum()
        _sp = [(_tn / (_tn + _fp)) if (_tn + _fp) > 0 else 0.0
               for _c in range(len(_mcm))
               for _tn, _fp, _fn, _tp in [_mcm[_c].ravel()]]
        return float(np.average(_sp, weights=_sup / _tot))

    return {
        'accuracy':    acc,  'accuracy_ci':    bootstrap_ci(y_true, y_pred, _acc,  n_bootstrap),
        'precision':   prec, 'precision_ci':   bootstrap_ci(y_true, y_pred, _prec, n_bootstrap),
        'recall':      rec,  'recall_ci':      bootstrap_ci(y_true, y_pred, _rec,  n_bootstrap),
        'f1':          f1,   'f1_ci':          bootstrap_ci(y_true, y_pred, _f1,   n_bootstrap),
        'specificity': spec, 'specificity_ci': bootstrap_ci(y_true, y_pred, _spec, n_bootstrap),
    }


def print_detailed_metrics(y_true, y_pred, class_names, split_name, eval_method,
                           fold_num, model_type, checkpoint_name=None, n_bootstrap=1000):
    """
    Print full metrics report matching the supervised model example format,
    including per-class metrics with CI, confusion matrix, and classification report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall = compute_overall_metrics_with_ci(y_true, y_pred, n_bootstrap)
    per_class = compute_per_class_metrics_with_ci(y_true, y_pred, class_names, n_bootstrap)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred,
                               target_names=class_names, digits=4, zero_division=0)

    header = f"fold{fold_num} - {model_type} - {eval_method.upper()} - {split_name.upper()}"
    if checkpoint_name:
        header = f"{checkpoint_name}  [{eval_method.upper()} | {split_name.upper()} | fold{fold_num}]"

    print("\n" + "=" * 100)
    print(f"Metrics for {header}")
    print("=" * 100)

    if checkpoint_name:
        print("\n" + "=" * 100)
        print(f"Model: {checkpoint_name}")
        print("=" * 100)

    def _fmt(val, lo, hi):
        return f"{val:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}])"

    print("\n── Overall (Weighted) Metrics ──")
    print(f"  Accuracy:    {_fmt(overall['accuracy'],    *overall['accuracy_ci'])}")
    print(f"  Precision:   {_fmt(overall['precision'],   *overall['precision_ci'])}")
    print(f"  Recall:      {_fmt(overall['recall'],      *overall['recall_ci'])}")
    print(f"  F1 Score:    {_fmt(overall['f1'],          *overall['f1_ci'])}")
    print(f"  Specificity: {_fmt(overall['specificity'], *overall['specificity_ci'])}")

    print("\n── Per-Class Metrics ──")
    col_w = max(len(n) for n in class_names) + 2
    hdr = f"  {'Class':<{col_w}}{'Precision':^28}{'Recall':^28}{'F1':^24}{'Specificity':^28}"
    print(hdr)
    print("  " + "-" * (col_w + 28 + 28 + 24 + 28))
    for pc in per_class:
        name = pc['class']
        p_str   = f"{pc['precision']:.4f} [{pc['precision_ci'][0]:.4f},{pc['precision_ci'][1]:.4f}]"
        r_str   = f"{pc['recall']:.4f} [{pc['recall_ci'][0]:.4f},{pc['recall_ci'][1]:.4f}]"
        f_str   = f"{pc['f1']:.4f} [{pc['f1_ci'][0]:.4f},{pc['f1_ci'][1]:.4f}]"
        s_str   = f"{pc['specificity']:.4f} [{pc['specificity_ci'][0]:.4f},{pc['specificity_ci'][1]:.4f}]"
        print(f"  {name:<{col_w}}{p_str:^28}  {r_str:^28}  {f_str:^24}  {s_str:^28}")

    # Confusion matrix
    pred_labels = [f"Pred: {n}" for n in class_names]
    true_labels = [f"True: {n}" for n in class_names]
    col_width = 14
    print("\n── Confusion Matrix ──")
    header_row = " " * 14 + "  ".join(f"{lbl:<{col_width}}" for lbl in pred_labels)
    print(f"  {header_row}")
    print("  " + "-" * (14 + (col_width + 2) * len(class_names)))
    for i, row_lbl in enumerate(true_labels):
        row_vals = "  ".join(f"{cm[i, j]:<{col_width}}" for j in range(len(class_names)))
        print(f"  {row_lbl:<14}{row_vals}")

    print("\n── Per-Class Confusion Matrix Elements (TP / FP / FN / TN) ──")
    for pc in per_class:
        print(f"  {pc['class']:<22}TP={pc['tp']:>5}  FP={pc['fp']:>5}  FN={pc['fn']:>5}  TN={pc['tn']:>5}")

    print("\n── Classification Report ──")
    print(cr)
    print("=" * 100)


def save_detailed_metrics_to_file(y_true, y_pred, class_names, split_name, eval_method,
                                   fold_num, model_type, output_dir,
                                   checkpoint_name=None, n_bootstrap=1000):
    """Save the detailed metrics report to a text file."""
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_detailed_metrics(y_true, y_pred, class_names, split_name, eval_method,
                           fold_num, model_type, checkpoint_name, n_bootstrap)
    sys.stdout = old_stdout
    content = buf.getvalue()

    fname = f"{model_type}_fold{fold_num}_{eval_method}_{split_name}_detailed_metrics.txt"
    if checkpoint_name:
        safe_name = checkpoint_name.replace('.pth', '').replace(' ', '_')
        fname = f"{model_type}_fold{fold_num}_{eval_method}_{split_name}_{safe_name}_metrics.txt"
    out_path = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(content)
    return out_path


# ============================================================================
# Inference Timing
# ============================================================================

def measure_inference_time(model_or_fn, data_loader_or_dataset, batch_sizes,
                            device, transform=None, n_warmup=10):
    """
    Measure inference time for a callable (encoder + classifier) over multiple batch sizes.
    `model_or_fn` should be a callable that takes a batch tensor and returns logits/features.
    `data_loader_or_dataset` should expose raw images as a list/array (via ImageFolder).
    Returns a dict: {batch_size: {total_time, total_images, time_per_image, throughput}}
    """
    results = {}

    # Collect all images once (as tensors via the existing test_loader transform)
    all_images = []
    for batch in data_loader_or_dataset:
        imgs = batch[0]  # already transformed tensors
        all_images.append(imgs)
    all_images = torch.cat(all_images, dim=0)  # (N, C, H, W)
    total_images = len(all_images)

    model_or_fn.eval()

    for bs in batch_sizes:
        # Warmup
        with torch.no_grad():
            warmup_batch = all_images[:min(bs, total_images)].to(device)
            for _ in range(n_warmup):
                _ = model_or_fn(warmup_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Timed run
        start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, total_images, bs):
                batch = all_images[i:i + bs].to(device)
                _ = model_or_fn(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        time_per_image = elapsed / total_images
        throughput = total_images / elapsed

        results[bs] = {
            'total_time': elapsed,
            'total_images': total_images,
            'time_per_image': time_per_image,
            'throughput': throughput,
        }

    return results


def print_inference_time_report(timing_results, fold_num, model_type, eval_method,
                                 split_name, checkpoint_name=None):
    """Print inference timing report matching the supervised model example format."""
    header = f"fold{fold_num} - {model_type} - {eval_method.upper()} - {split_name.upper()}"
    print("\n" + "=" * 100)
    print(f"Inference Time Analysis for {header}")
    print("=" * 100)

    if checkpoint_name:
        print("\n" + "=" * 100)
        print(f"Model: {checkpoint_name}")
        print("=" * 100)

    for bs, r in sorted(timing_results.items()):
        print(f"\nBatch Size: {bs}")
        print(f"  Total Time:     {r['total_time']:.6f} seconds")
        print(f"  Total Images:   {r['total_images']}")
        print(f"  Time per Image: {r['time_per_image']:.6f} s ({r['time_per_image']*1000:.4f} ms)")
        print(f"  Throughput:     {r['throughput']:.2f} images/second")

    print()


def save_inference_time_to_file(timing_results, fold_num, model_type, eval_method,
                                 split_name, output_dir, checkpoint_name=None):
    """Save inference timing report to file."""
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_inference_time_report(timing_results, fold_num, model_type,
                                eval_method, split_name, checkpoint_name)
    sys.stdout = old_stdout
    content = buf.getvalue()

    safe = (checkpoint_name or f"{eval_method}_{split_name}").replace('.pth', '').replace(' ', '_')
    fname = f"{model_type}_fold{fold_num}_{safe}_inference_time.txt"
    out_path = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(content)
    return out_path


# ============================================================================
# Checkpoint and Metrics Management
# ============================================================================

def get_fold_checkpoint_dir(checkpoint_base_dir, model_type, fold_num):
    """Get checkpoint directory for a specific fold"""
    fold_dir = os.path.join(checkpoint_base_dir, model_type, f'fold{fold_num}')
    os.makedirs(fold_dir, exist_ok=True)
    return fold_dir


def save_linear_checkpoint(classifier, encoder, model_type, optimizer, epoch, 
                           train_loss, val_loss, train_acc, val_acc, fold_dir, checkpoint_type='best'):
    """Save linear classifier checkpoint (best or last)"""
    checkpoint_path = os.path.join(fold_dir, f'{model_type}_{checkpoint_type}_linear_checkpoint.pth')
    
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'model_type': model_type
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_finetune_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                            train_acc, val_acc, fold_dir, model_type, checkpoint_type='best'):
    """Save fine-tuned model checkpoint (best or last)"""
    checkpoint_path = os.path.join(fold_dir, f'{model_type}_{checkpoint_type}_finetune_checkpoint.pth')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'model_type': model_type
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def append_linear_metrics(fold_dir, model_type, epoch, train_loss, val_loss, train_acc, val_acc):
    """Append linear training metrics to file"""
    metrics_path = os.path.join(fold_dir, f'{model_type}_linear_metrics.txt')
    
    with open(metrics_path, 'a') as f:
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%\n")


def append_finetune_metrics(fold_dir, model_type, epoch, train_loss, val_loss, train_acc, val_acc):
    """Append fine-tuning metrics to file"""
    metrics_path = os.path.join(fold_dir, f'{model_type}_finetune_metrics.txt')
    
    with open(metrics_path, 'a') as f:
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%\n")


def save_final_metrics(fold_dir, model_type, knn_results, linear_results, finetune_results, 
                      linear_best_epoch=None, finetune_best_epoch=None):
    """Save all evaluation metrics to file"""
    metrics_path = os.path.join(fold_dir, f'{model_type}_metrics_results.txt')
    
    with open(metrics_path, 'w') as f:  # Changed to 'w' to overwrite
        f.write("="*80 + "\n")
        f.write(f"COMPREHENSIVE EVALUATION RESULTS - {model_type.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # KNN Results (Train, Val, Test)
        if knn_results:
            f.write("KNN EVALUATION\n")
            f.write("-"*80 + "\n")
            if 'best_k' in knn_results:
                f.write(f"Best K: {knn_results.get('best_k', 'N/A')}\n")
            elif 'k' in knn_results:
                f.write(f"K Value: {knn_results.get('k', 'N/A')}\n")
            f.write("\n")
            
            # Train metrics
            if 'train' in knn_results:
                f.write("TRAIN SET:\n")
                f.write(f"  Top-1 Accuracy: {knn_results['train'].get('top1_accuracy', 0):.4f} "
                       f"({knn_results['train'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {knn_results['train'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {knn_results['train'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {knn_results['train'].get('f1_score', 0):.4f}\n\n")
            
            # Val metrics
            if 'val' in knn_results:
                f.write("VALIDATION SET:\n")
                f.write(f"  Top-1 Accuracy: {knn_results['val'].get('top1_accuracy', 0):.4f} "
                       f"({knn_results['val'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {knn_results['val'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {knn_results['val'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {knn_results['val'].get('f1_score', 0):.4f}\n\n")
            
            # Test metrics
            if 'test' in knn_results:
                f.write("TEST SET:\n")
                f.write(f"  Top-1 Accuracy: {knn_results['test'].get('top1_accuracy', 0):.4f} "
                       f"({knn_results['test'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {knn_results['test'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {knn_results['test'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {knn_results['test'].get('f1_score', 0):.4f}\n\n")
        
        # Linear Results (Train, Val, Test)
        if linear_results:
            f.write("LINEAR EVALUATION\n")
            f.write("-"*80 + "\n")
            if linear_best_epoch is not None:
                f.write(f"Best Epoch: {linear_best_epoch}\n\n")
            
            # Train metrics
            if 'train' in linear_results:
                f.write("TRAIN SET:\n")
                f.write(f"  Top-1 Accuracy: {linear_results['train'].get('top1_accuracy', 0):.4f} "
                       f"({linear_results['train'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {linear_results['train'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {linear_results['train'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {linear_results['train'].get('f1_score', 0):.4f}\n\n")
            
            # Val metrics
            if 'val' in linear_results:
                f.write("VALIDATION SET:\n")
                f.write(f"  Top-1 Accuracy: {linear_results['val'].get('top1_accuracy', 0):.4f} "
                       f"({linear_results['val'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {linear_results['val'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {linear_results['val'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {linear_results['val'].get('f1_score', 0):.4f}\n\n")
            
            # Test metrics
            if 'test' in linear_results:
                f.write("TEST SET:\n")
                f.write(f"  Top-1 Accuracy: {linear_results['test'].get('top1_accuracy', 0):.4f} "
                       f"({linear_results['test'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {linear_results['test'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {linear_results['test'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {linear_results['test'].get('f1_score', 0):.4f}\n\n")
        
        # Fine-tuning Results (Train, Val, Test)
        if finetune_results:
            f.write("FINE-TUNING EVALUATION\n")
            f.write("-"*80 + "\n")
            if finetune_best_epoch is not None:
                f.write(f"Best Epoch: {finetune_best_epoch}\n\n")
            
            # Train metrics
            if 'train' in finetune_results:
                f.write("TRAIN SET:\n")
                f.write(f"  Top-1 Accuracy: {finetune_results['train'].get('top1_accuracy', 0):.4f} "
                       f"({finetune_results['train'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {finetune_results['train'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {finetune_results['train'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {finetune_results['train'].get('f1_score', 0):.4f}\n\n")
            
            # Val metrics
            if 'val' in finetune_results:
                f.write("VALIDATION SET:\n")
                f.write(f"  Top-1 Accuracy: {finetune_results['val'].get('top1_accuracy', 0):.4f} "
                       f"({finetune_results['val'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {finetune_results['val'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {finetune_results['val'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {finetune_results['val'].get('f1_score', 0):.4f}\n\n")
            
            # Test metrics
            if 'test' in finetune_results:
                f.write("TEST SET:\n")
                f.write(f"  Top-1 Accuracy: {finetune_results['test'].get('top1_accuracy', 0):.4f} "
                       f"({finetune_results['test'].get('top1_accuracy', 0)*100:.2f}%)\n")
                f.write(f"  Precision:      {finetune_results['test'].get('precision', 0):.4f}\n")
                f.write(f"  Recall:         {finetune_results['test'].get('recall', 0):.4f}\n")
                f.write(f"  F1-Score:       {finetune_results['test'].get('f1_score', 0):.4f}\n\n")
        
        f.write("="*80 + "\n")


def save_knn_detailed_results(fold_dir, model_type, k_results, best_k_result):
    """Save detailed KNN results for all k values tested"""
    knn_path = os.path.join(fold_dir, f'{model_type}_knn_all_k_results.txt')
    
    with open(knn_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"KNN EVALUATION - ALL K VALUES - {model_type.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write("TESTED K VALUES:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'K':>4}  {'Train Acc':>10}  {'Val Acc':>10}  {'Test Acc':>10}  {'Val F1':>10}\n")
        f.write("-"*80 + "\n")
        
        for result in k_results:
            k = result['k']
            train_acc = result['train']['top1_accuracy']
            val_acc = result['val']['top1_accuracy']
            test_acc = result['test']['top1_accuracy']
            val_f1 = result['val']['f1_score']
            
            marker = " *" if k == best_k_result['k'] else "  "
            f.write(f"{k:>4}{marker}  {train_acc:>10.4f}  {val_acc:>10.4f}  {test_acc:>10.4f}  {val_f1:>10.4f}\n")
        
        f.write("-"*80 + "\n")
        f.write(f"* = Best K (highest validation F1-Score)\n\n")
        
        f.write("BEST K VALUE:\n")
        f.write("-"*80 + "\n")
        f.write(f"K: {best_k_result['k']}\n\n")
        
        f.write("TRAIN SET:\n")
        f.write(f"  Top-1 Accuracy: {best_k_result['train']['top1_accuracy']:.4f} ({best_k_result['train']['top1_accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:      {best_k_result['train']['precision']:.4f}\n")
        f.write(f"  Recall:         {best_k_result['train']['recall']:.4f}\n")
        f.write(f"  F1-Score:       {best_k_result['train']['f1_score']:.4f}\n\n")
        
        f.write("VALIDATION SET:\n")
        f.write(f"  Top-1 Accuracy: {best_k_result['val']['top1_accuracy']:.4f} ({best_k_result['val']['top1_accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:      {best_k_result['val']['precision']:.4f}\n")
        f.write(f"  Recall:         {best_k_result['val']['recall']:.4f}\n")
        f.write(f"  F1-Score:       {best_k_result['val']['f1_score']:.4f}\n\n")
        
        f.write("TEST SET:\n")
        f.write(f"  Top-1 Accuracy: {best_k_result['test']['top1_accuracy']:.4f} ({best_k_result['test']['top1_accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:      {best_k_result['test']['precision']:.4f}\n")
        f.write(f"  Recall:         {best_k_result['test']['recall']:.4f}\n")
        f.write(f"  F1-Score:       {best_k_result['test']['f1_score']:.4f}\n")
        f.write("="*80 + "\n")


def load_linear_checkpoint(checkpoint_path, classifier, encoder):
    """Load linear classifier checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Linear checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    print(f"Loaded linear checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}, Train Acc: {checkpoint['train_acc']:.2f}%")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}, Val Acc: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint


def load_finetune_checkpoint(checkpoint_path, model):
    """Load fine-tuned model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded fine-tune checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}, Train Acc: {checkpoint['train_acc']:.2f}%")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}, Val Acc: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint


# ============================================================================
# Data Transformations
# ============================================================================

def get_train_transform(image_size=224, dataset_mean=None, dataset_std=None):
    """Get training transformation with augmentation"""
    if dataset_mean is None:
        dataset_mean = [0.485, 0.456, 0.406]
    if dataset_std is None:
        dataset_std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])


def get_test_transform(image_size=224, dataset_mean=None, dataset_std=None):
    """Get test transformation (no augmentation)"""
    if dataset_mean is None:
        dataset_mean = [0.485, 0.456, 0.406]
    if dataset_std is None:
        dataset_std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])


# ============================================================================
# Embedding Extraction Functions (Lightly Models Only)
# ============================================================================

def extract_embeddings_simclr(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly SimCLR model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            h, _ = model(images)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_moco(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly MoCo model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            h = model.get_features(images)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_simsiam(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly SimSiam model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            h = model.encoder(images).flatten(start_dim=1)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_mae(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly MAE model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            latent = model.forward_encoder(images, idx_keep=None)
            cls_token = latent[:, 0, :]
            embeddings.append(cls_token.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_nnclr(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly NNCLR model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            h = model.encoder(images).flatten(start_dim=1)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_simmim(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly SimMIM model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            # Extract features from backbone (ViT)
            # For SimMIM, use the backbone encoder
            features = model.backbone.vit.conv_proj(images)
            features = features.flatten(2).transpose(1, 2)  # B, N, C
            cls_token = model.backbone.vit.class_token.expand(features.shape[0], -1, -1)
            features = torch.cat([cls_token, features], dim=1)
            features = model.backbone.vit.encoder(features)
            cls_features = features[:, 0]  # CLS token
            embeddings.append(cls_features.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_byol(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly BYOL model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            # Use get_features method if available, otherwise use backbone
            if hasattr(model, 'get_features'):
                h = model.get_features(images)
            else:
                h = model.backbone(images).flatten(start_dim=1)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_dinov2(model, data_loader, split_name='test'):
    """Extract embeddings from Lightly DINOv2 model"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Extracting {split_name} embeddings'):
            images = batch[0].to(device)
            targets = batch[1]
            # Use get_features method which returns CLS token from teacher
            h = model.get_features(images)
            embeddings.append(h.cpu().numpy())
            labels.append(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    embeddings = normalize(embeddings)
    
    return embeddings, labels


def extract_embeddings_dispatch(model_type, model, data_loader, split_name='test'):
    """Dispatch to appropriate embedding extraction function"""
    extraction_functions = {
        'simclr': extract_embeddings_simclr,
        'moco': extract_embeddings_moco,
        'simsiam': extract_embeddings_simsiam,
        'mae': extract_embeddings_mae,
        'nnclr': extract_embeddings_nnclr,
        'simmim': extract_embeddings_simmim,
        'byol': extract_embeddings_byol,
        'dinov2': extract_embeddings_dinov2
    }
    
    if model_type not in extraction_functions:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(extraction_functions.keys())}")
    
    return extraction_functions[model_type](model, data_loader, split_name)


# ============================================================================
# Model Loading and Encoder Extraction (Lightly Models Only)
# ============================================================================

def load_model(model_type, args):
    """Load Lightly-based model"""
    if model_type == 'simclr':
        return SimCLRModel(base_encoder=args.encoder, projection_dim=args.projection_dim)
    elif model_type == 'moco':
        return MoCoModel(base_encoder=args.encoder, projection_dim=args.projection_dim,
                        queue_size=args.queue_size, momentum=args.moco_momentum,
                        temperature=args.temperature)
    elif model_type == 'simsiam':
        return SimSiamModel(base_encoder=args.encoder, projection_dim=args.simsiam_proj_dim,
                           pred_dim=args.simsiam_pred_dim)
    elif model_type == 'mae':
        return MAEModel(img_size=args.image_size, patch_size=args.mae_patch_size,
                       mask_ratio=args.mae_mask_ratio, decoder_embed_dim=args.mae_decoder_embed_dim,
                       decoder_depth=args.mae_decoder_depth, decoder_num_heads=args.mae_decoder_num_heads)
    elif model_type == 'nnclr':
        return NNCLRModel(base_encoder=args.encoder, projection_dim=args.projection_dim,
                         queue_size=args.queue_size, temperature=args.temperature)
    elif model_type == 'simmim':
        return SimMIMModel(img_size=args.image_size, patch_size=args.simmim_patch_size,
                          mask_ratio=args.simmim_mask_ratio)
    elif model_type == 'byol':
        return BYOLModel(base_encoder=args.encoder, projection_dim=args.byol_proj_dim,
                        hidden_dim=args.byol_hidden_dim)
    elif model_type == 'dinov2':
        return DINOv2Model(
            img_size=args.image_size,
            patch_size=args.dinov2_patch_size,
            embed_dim=args.dinov2_embed_dim,
            drop_path_rate=args.dinov2_drop_path_rate,
            ibot_separate_head=args.dinov2_separate_head
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_encoder(model, model_type):
    """Extract encoder from Lightly SSL model"""
    if model_type == 'simclr':
        return model.encoder
    elif model_type == 'moco':
        return model.backbone
    elif model_type == 'simsiam':
        return model.encoder
    elif model_type == 'mae':
        return model
    elif model_type == 'nnclr':
        return model.encoder
    elif model_type == 'simmim':
        return model.backbone
    elif model_type == 'byol':
        return model.backbone
    elif model_type == 'dinov2':
        return model.teacher_backbone
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_feature_dim(model, model_type):
    """Get feature dimension from encoder"""
    if model_type in ['mae', 'simmim']:
        return model.feature_dim
    else:
        return model.feature_dim


# ============================================================================
# 1. KNN Evaluation (Multiple K Values)
# ============================================================================

def evaluate_knn_on_dataset(knn_classifier, embeddings, labels, dataset_name='dataset'):
    """Evaluate KNN on a single dataset"""
    predictions = knn_classifier.predict(embeddings)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'top1_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'labels': labels,
    }


def evaluate_knn_multiple_k(train_embeddings, train_labels, val_embeddings, val_labels,
                           test_embeddings, test_labels, k_values=[1, 3, 5, 10, 20, 50]):
    """Evaluate using KNN with multiple k values on train, val, and test sets"""
    print(f"\n{'='*70}")
    print("KNN EVALUATION - MULTIPLE K VALUES (TRAIN, VAL, TEST)")
    print(f"{'='*70}")
    
    results = []
    
    for k in k_values:
        print(f"\nTraining KNN classifier with k={k}...")
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
            weights="distance",
            n_jobs=-1
        )
        knn.fit(train_embeddings, train_labels)
        
        # Evaluate on all three datasets
        train_results = evaluate_knn_on_dataset(knn, train_embeddings, train_labels, 'train')
        val_results = evaluate_knn_on_dataset(knn, val_embeddings, val_labels, 'val')
        test_results = evaluate_knn_on_dataset(knn, test_embeddings, test_labels, 'test')
        
        results.append({
            'k': k,
            'train': train_results,
            'val': val_results,
            'test': test_results
        })
        
        print(f"k={k:2d}:")
        print(f"  Train - Acc={train_results['top1_accuracy']:.4f} ({train_results['top1_accuracy']*100:.2f}%), "
              f"F1={train_results['f1_score']:.4f}")
        print(f"  Val   - Acc={val_results['top1_accuracy']:.4f} ({val_results['top1_accuracy']*100:.2f}%), "
              f"F1={val_results['f1_score']:.4f}")
        print(f"  Test  - Acc={test_results['top1_accuracy']:.4f} ({test_results['top1_accuracy']*100:.2f}%), "
              f"F1={test_results['f1_score']:.4f}")
    
    # Find best k based on validation F1-score
    best_result = max(results, key=lambda x: x['val']['f1_score'])
    print(f"\n{'='*70}")
    print(f"BEST K: {best_result['k']} (based on validation F1-Score={best_result['val']['f1_score']:.4f})")
    print(f"  Train Acc: {best_result['train']['top1_accuracy']:.4f}")
    print(f"  Val Acc:   {best_result['val']['top1_accuracy']:.4f}")
    print(f"  Test Acc:  {best_result['test']['top1_accuracy']:.4f}")
    print(f"{'='*70}")
    
    return results, best_result


# ============================================================================
# 2. Linear Evaluation (Top-1 Accuracy)
# ============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen features"""
    def __init__(self, feature_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_linear_classifier(encoder, train_loader, val_loader, num_classes, feature_dim, 
                           model_type, fold_dir, epochs=100, lr=0.001):
    """Train linear classifier on frozen features"""
    print(f"\n{'='*70}")
    print("LINEAR EVALUATION")
    print(f"{'='*70}")
    
    classifier = LinearClassifier(feature_dim, num_classes).to(device)
    
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_classifier_state = None
    best_encoder_state = None
    best_epoch = 0
    
    # Clear previous metrics file
    metrics_path = os.path.join(fold_dir, f'{model_type}_linear_metrics.txt')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    
    print(f"\nTraining linear classifier for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Linear Epoch {epoch}/{epochs}', leave=False)
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Extract features
            with torch.no_grad():
                if model_type == 'mae':
                    latent = encoder.forward_encoder(images, idx_keep=None)
                    features = latent[:, 0, :]
                elif model_type == 'simmim':
                    # For SimMIM, extract features from backbone
                    feats = encoder.vit.conv_proj(images)
                    feats = feats.flatten(2).transpose(1, 2)
                    cls_token = encoder.vit.class_token.expand(feats.shape[0], -1, -1)
                    feats = torch.cat([cls_token, feats], dim=1)
                    feats = encoder.vit.encoder(feats)
                    features = feats[:, 0]
                else:
                    features = encoder(images).flatten(start_dim=1)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                if model_type == 'mae':
                    latent = encoder.forward_encoder(images, idx_keep=None)
                    features = latent[:, 0, :]
                elif model_type == 'simmim':
                    feats = encoder.vit.conv_proj(images)
                    feats = feats.flatten(2).transpose(1, 2)
                    cls_token = encoder.vit.class_token.expand(feats.shape[0], -1, -1)
                    feats = torch.cat([cls_token, feats], dim=1)
                    feats = encoder.vit.encoder(feats)
                    features = feats[:, 0]
                else:
                    features = encoder(images).flatten(start_dim=1)
                
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        append_linear_metrics(fold_dir, model_type, epoch, train_loss_avg, val_loss_avg, train_acc, val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best checkpoint based on validation loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_epoch = epoch
            
            # Save best checkpoint
            save_linear_checkpoint(
                classifier, encoder, model_type, optimizer, epoch,
                train_loss_avg, val_loss_avg, train_acc, val_acc, fold_dir, checkpoint_type='best'
            )
    
    # Save last checkpoint
    save_linear_checkpoint(
        classifier, encoder, model_type, optimizer, epoch,
        train_loss_avg, val_loss_avg, train_acc, val_acc, fold_dir, checkpoint_type='last'
    )
    
    # Load best checkpoint for evaluation
    classifier.load_state_dict(best_classifier_state)
    encoder.load_state_dict(best_encoder_state)
    
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Linear checkpoints saved (best and last)")
    
    return classifier, best_val_loss, best_epoch


def evaluate_linear(classifier, encoder, train_loader, val_loader, test_loader, model_type):
    """Evaluate linear classifier on train, val, and test sets"""
    classifier.eval()
    encoder.eval()
    
    results = {}
    
    for loader, split_name in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Testing linear classifier on {split_name}'):
                images = batch[0].to(device)
                labels = batch[1]
                
                if model_type == 'mae':
                    latent = encoder.forward_encoder(images, idx_keep=None)
                    features = latent[:, 0, :]
                elif model_type == 'simmim':
                    feats = encoder.vit.conv_proj(images)
                    feats = feats.flatten(2).transpose(1, 2)
                    cls_token = encoder.vit.class_token.expand(feats.shape[0], -1, -1)
                    feats = torch.cat([cls_token, feats], dim=1)
                    feats = encoder.vit.encoder(feats)
                    features = feats[:, 0]
                else:
                    features = encoder(images).flatten(start_dim=1)
                
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        top1_accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        results[split_name] = {
            'top1_accuracy': top1_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    print(f"\n{'='*70}")
    print("LINEAR EVALUATION RESULTS")
    print(f"{'='*70}")
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()} SET:")
        print(f"  Top-1 Accuracy: {results[split]['top1_accuracy']:.4f} ({results[split]['top1_accuracy']*100:.2f}%)")
        print(f"  Precision:      {results[split]['precision']:.4f}")
        print(f"  Recall:         {results[split]['recall']:.4f}")
        print(f"  F1-Score:       {results[split]['f1_score']:.4f}")
    print(f"{'='*70}")
    
    return results


# ============================================================================
# 3. Fine-tuning Evaluation (Top-1 Accuracy)
# ============================================================================

class FineTuneClassifier(nn.Module):
    """Classifier with fine-tunable encoder"""
    def __init__(self, encoder, feature_dim, num_classes, model_type):
        super(FineTuneClassifier, self).__init__()
        self.encoder = encoder
        self.model_type = model_type
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        if self.model_type == 'mae':
            latent = self.encoder.forward_encoder(x, idx_keep=None)
            features = latent[:, 0, :]
        elif self.model_type == 'simmim':
            feats = self.encoder.vit.conv_proj(x)
            feats = feats.flatten(2).transpose(1, 2)
            cls_token = self.encoder.vit.class_token.expand(feats.shape[0], -1, -1)
            feats = torch.cat([cls_token, feats], dim=1)
            feats = self.encoder.vit.encoder(feats)
            features = feats[:, 0]
        else:
            features = self.encoder(x).flatten(start_dim=1)
        return self.fc(features)


def finetune_model(encoder, train_loader, val_loader, num_classes, feature_dim, 
                   model_type, fold_dir, epochs=50, lr=0.0001):
    """Fine-tune entire model"""
    print(f"\n{'='*70}")
    print("FINE-TUNING EVALUATION")
    print(f"{'='*70}")
    
    model = FineTuneClassifier(encoder, feature_dim, num_classes, model_type).to(device)
    
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    metrics_path = os.path.join(fold_dir, f'{model_type}_finetune_metrics.txt')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    
    print(f"\nFine-tuning model for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Finetune Epoch {epoch}/{epochs}', leave=False)
        for batch in pbar:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        append_finetune_metrics(fold_dir, model_type, epoch, train_loss_avg, val_loss_avg, train_acc, val_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best checkpoint based on validation loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
            # Save best checkpoint
            save_finetune_checkpoint(
                model, optimizer, epoch, train_loss_avg, val_loss_avg, 
                train_acc, val_acc, fold_dir, model_type, checkpoint_type='best'
            )
    
    # Save last checkpoint
    save_finetune_checkpoint(
        model, optimizer, epoch, train_loss_avg, val_loss_avg, 
        train_acc, val_acc, fold_dir, model_type, checkpoint_type='last'
    )
    
    # Load best checkpoint for evaluation
    model.load_state_dict(best_model_state)
    
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Fine-tune checkpoints saved (best and last)")
    
    return model, best_val_loss, best_epoch


def evaluate_finetuned(model, train_loader, val_loader, test_loader):
    """Evaluate fine-tuned model on train, val, and test sets"""
    model.eval()
    
    results = {}
    
    for loader, split_name in [(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Testing fine-tuned model on {split_name}'):
                images = batch[0].to(device)
                labels = batch[1]
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        top1_accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        results[split_name] = {
            'top1_accuracy': top1_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    print(f"\n{'='*70}")
    print("FINE-TUNING RESULTS")
    print(f"{'='*70}")
    for split in ['train', 'val', 'test']:
        print(f"{split.upper()} SET:")
        print(f"  Top-1 Accuracy: {results[split]['top1_accuracy']:.4f} ({results[split]['top1_accuracy']*100:.2f}%)")
        print(f"  Precision:      {results[split]['precision']:.4f}")
        print(f"  Recall:         {results[split]['recall']:.4f}")
        print(f"  F1-Score:       {results[split]['f1_score']:.4f}")
    print(f"{'='*70}")
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confusion_matrix(cm, class_names, fold_num, method_name, save_path, model_type):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_type.upper()} - {method_name} - Fold {fold_num}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved: {save_path}")
    plt.close()


def plot_k_comparison(k_results, fold_num, save_path, model_type):
    """Plot comparison of different k values"""
    k_values = [r['k'] for r in k_results]
    train_accuracies = [r['train']['top1_accuracy'] for r in k_results]
    val_accuracies = [r['val']['top1_accuracy'] for r in k_results]
    test_accuracies = [r['test']['top1_accuracy'] for r in k_results]
    val_f1_scores = [r['val']['f1_score'] for r in k_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'KNN Performance vs K - {model_type.upper()} - Fold {fold_num}', 
                 fontsize=16, fontweight='bold')
    
    # Train accuracy
    ax1.plot(k_values, train_accuracies, 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax1.set_ylabel('Top-1 Accuracy', fontsize=12)
    ax1.set_title('Train Set Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Val accuracy
    ax2.plot(k_values, val_accuracies, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax2.set_ylabel('Top-1 Accuracy', fontsize=12)
    ax2.set_title('Validation Set Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Test accuracy
    ax3.plot(k_values, test_accuracies, 'o-', linewidth=2, markersize=8, color='red')
    ax3.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax3.set_ylabel('Top-1 Accuracy', fontsize=12)
    ax3.set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Val F1-score
    ax4.plot(k_values, val_f1_scores, 'o-', linewidth=2, markersize=8, color='orange')
    ax4.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Validation Set F1-Score', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"K comparison plot saved: {save_path}")
    plt.close()


def plot_method_comparison(results, fold_num, save_path, model_type):
    """Plot comparison of evaluation methods"""
    methods = list(results.keys())
    metrics = ['top1_accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Top-1 Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Methods Comparison - {model_type.upper()} - Fold {fold_num}', 
                fontsize=16, fontweight='bold')
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        values = [results[method][metric] for method in methods]
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=np.mean(values), color='r', linestyle='--', 
                  alpha=0.5, label=f'Average: {np.mean(values):.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Method comparison plot saved: {save_path}")
    plt.close()


# ============================================================================
# Main Testing Function
# ============================================================================

def test_fold(fold_num, args, all_results):
    """Test a single fold with all three evaluation methods"""
    
    print("\n" + "="*80)
    print(f"TESTING FOLD {fold_num}/{args.num_folds} - MODEL: {args.model_type.upper()}")
    print("="*80)
    
    checkpoint_fold_dir = get_fold_checkpoint_dir(args.checkpoint_dir, args.model_type, fold_num)
    print(f"\nCheckpoint directory: {checkpoint_fold_dir}")
    
    fold_dir = os.path.join(args.data_path, f"fold{fold_num}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    test_dir = os.path.join(fold_dir, "test")
    
    for directory, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(directory):
            raise ValueError(f"{name} directory not found: {directory}")
    
    ssl_checkpoint_path = os.path.join(
        checkpoint_fold_dir,
        f'{args.model_type}_{"best" if args.use_best else "last"}_checkpoint.pth'
    )
    
    if not os.path.exists(ssl_checkpoint_path):
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_checkpoint_path}")
    
    print(f"\nSSL Checkpoint: {ssl_checkpoint_path}")
    
    checkpoint = torch.load(ssl_checkpoint_path, map_location=device, weights_only=False)
    
    model = load_model(args.model_type, args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded SSL checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint.get('train_loss', 0):.4f}")
    print(f"Val loss: {checkpoint.get('val_loss', 0):.4f}")
    
    test_transform = get_test_transform(args.image_size)
    train_transform = get_train_transform(args.image_size)
    
    # Use LightlyDataset instead of ImageFolder
    train_dataset = LightlyDataset(input_dir=train_dir, transform=test_transform)
    val_dataset = LightlyDataset(input_dir=val_dir, transform=test_transform)
    test_dataset = LightlyDataset(input_dir=test_dir, transform=test_transform)
    
    train_dataset_aug = LightlyDataset(input_dir=train_dir, transform=train_transform)
    val_dataset_aug = LightlyDataset(input_dir=val_dir, transform=test_transform)
    
    # Get class information
    # LightlyDataset doesn't have .classes attribute, so we get it from the directory
    temp_imagefolder = ImageFolder(root=train_dir)
    num_classes = len(temp_imagefolder.classes)
    class_names = temp_imagefolder.classes
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader_aug = DataLoader(val_dataset_aug, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    encoder = get_encoder(model, args.model_type)
    feature_dim = get_feature_dim(model, args.model_type)
    
    print(f"\nFeature dimension: {feature_dim}")
    
    fold_results = {}
    linear_best_epoch = None
    finetune_best_epoch = None
    
    # KNN EVALUATION
    if args.eval_knn:
        print("\n" + "="*80)
        print("METHOD 1: KNN EVALUATION WITH MULTIPLE K VALUES")
        print("="*80)
        
        print("\nExtracting embeddings for train, val, and test sets...")
        train_embeddings, train_labels = extract_embeddings_dispatch(
            args.model_type, model, train_loader, 'train'
        )
        val_embeddings, val_labels = extract_embeddings_dispatch(
            args.model_type, model, val_loader, 'val'
        )
        test_embeddings, test_labels = extract_embeddings_dispatch(
            args.model_type, model, test_loader, 'test'
        )
        
        print(f"Train embeddings shape: {train_embeddings.shape}")
        print(f"Val embeddings shape: {val_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")
        
        k_values = [int(k) for k in args.k_values.split(',')]
        k_results, best_k_result = evaluate_knn_multiple_k(
            train_embeddings, train_labels,
            val_embeddings, val_labels,
            test_embeddings, test_labels,
            k_values=k_values
        )
        
        fold_results['knn'] = best_k_result
        fold_results['knn_all_k'] = k_results
        
        save_knn_detailed_results(checkpoint_fold_dir, args.model_type, k_results, best_k_result)
        
        # ── Per-class detailed metrics for KNN (best k) ──
        for split, split_labels, split_preds in [
            ('train', train_labels, best_k_result['train']['predictions']),
            ('val',   val_labels,   best_k_result['val']['predictions']),
            ('test',  test_labels,  best_k_result['test']['predictions']),
        ]:
            print_detailed_metrics(split_labels, split_preds, class_names,
                                   split, 'KNN', fold_num, args.model_type,
                                   checkpoint_name=f"knn_k{best_k_result['k']}")
            save_detailed_metrics_to_file(split_labels, split_preds, class_names,
                                          split, 'knn', fold_num, args.model_type,
                                          args.output_dir,
                                          checkpoint_name=f"knn_k{best_k_result['k']}")

        # ── Inference timing for KNN (uses encoder + NN lookup) ──
        print(f"\n{'='*80}\nINFERENCE TIMING - KNN (fold{fold_num})\n{'='*80}")
        _batch_sizes = [int(x) for x in args.inference_batch_sizes.split(',')]

        def _knn_infer_fn(img_batch):
            with torch.no_grad():
                feats = extract_embeddings_dispatch(
                    args.model_type, model,
                    [(img_batch, torch.zeros(img_batch.shape[0], dtype=torch.long))],
                    'inference'
                )[0]
            return feats

        # Simpler timing: just measure encoder embedding extraction
        class _SimpleLoader:
            def __init__(self, imgs, bs):
                self.imgs = imgs
                self.bs   = bs
            def __iter__(self):
                for i in range(0, len(self.imgs), self.bs):
                    yield self.imgs[i:i+self.bs], torch.zeros(self.bs, dtype=torch.long)

        # Collect all test images as tensors
        _all_test_imgs = torch.cat([b[0] for b in test_loader], dim=0)

        def _encoder_fn(img_batch):
            return extract_embeddings_dispatch(
                args.model_type, model,
                [{'img': img_batch, 'label': torch.zeros(img_batch.shape[0], dtype=torch.long)}],
                'inference'
            )

        # Use raw encoder for timing (embedding extraction is the bottleneck)
        encoder_for_timing = encoder
        encoder_for_timing.eval()

        def _enc_forward(img_batch):
            img_batch = img_batch.to(device)
            if args.model_type == 'mae':
                latent = encoder_for_timing.forward_encoder(img_batch, idx_keep=None)
                return latent[:, 0, :]
            elif args.model_type == 'simmim':
                feats = encoder_for_timing.vit.conv_proj(img_batch)
                feats = feats.flatten(2).transpose(1, 2)
                cls_token = encoder_for_timing.vit.class_token.expand(feats.shape[0], -1, -1)
                feats = torch.cat([cls_token, feats], dim=1)
                feats = encoder_for_timing.vit.encoder(feats)
                return feats[:, 0]
            else:
                return encoder_for_timing(img_batch).flatten(start_dim=1)

        class _FakeModel(nn.Module):
            def __init__(self, fn): super().__init__(); self._fn = fn
            def forward(self, x): return self._fn(x)
            def eval(self): return self
            def parameters(self): return iter([])

        _timing_model = _FakeModel(_enc_forward)
        _dummy_loader = [(_all_test_imgs[i:i+32], None) for i in range(0, len(_all_test_imgs), 32)]

        knn_timing = measure_inference_time(_timing_model, _dummy_loader, _batch_sizes, device)
        print_inference_time_report(knn_timing, fold_num, args.model_type, 'KNN', 'test',
                                    checkpoint_name=f"knn_k{best_k_result['k']}_encoder")
        save_inference_time_to_file(knn_timing, fold_num, args.model_type, 'knn', 'test',
                                    args.output_dir, checkpoint_name=f"knn_k{best_k_result['k']}_encoder")

        if args.save_plots:
            # Save confusion matrices for all three datasets
            for split, labels, predictions in [
                ('train', train_labels, best_k_result['train']['predictions']),
                ('val', val_labels, best_k_result['val']['predictions']),
                ('test', test_labels, best_k_result['test']['predictions'])
            ]:
                cm = confusion_matrix(labels, predictions)
                cm_path = os.path.join(args.output_dir,
                                      f'{args.model_type}_fold{fold_num}_knn_k{best_k_result["k"]}_{split}_confusion.png')
                plot_confusion_matrix(cm, class_names, fold_num, f'KNN (k={best_k_result["k"]}) - {split.upper()}', 
                                    cm_path, args.model_type)
            
            k_comp_path = os.path.join(args.output_dir,
                                      f'{args.model_type}_fold{fold_num}_k_comparison.png')
            plot_k_comparison(k_results, fold_num, k_comp_path, args.model_type)
    
    # LINEAR EVALUATION
    if args.eval_linear:
        print("\n" + "="*80)
        print("METHOD 2: LINEAR EVALUATION")
        print("="*80)
        
        if args.load_linear_checkpoint:
            # Load existing checkpoint without training
            print(f"\nLoading existing linear checkpoint ({args.linear_checkpoint_type})...")
            
            classifier = LinearClassifier(feature_dim, num_classes).to(device)
            linear_checkpoint_path = os.path.join(
                checkpoint_fold_dir, 
                f'{args.model_type}_{args.linear_checkpoint_type}_linear_checkpoint.pth'
            )
            
            checkpoint_data = load_linear_checkpoint(linear_checkpoint_path, classifier, encoder)
            linear_best_epoch = checkpoint_data['epoch']
            
            print(f"Loaded {args.linear_checkpoint_type} linear checkpoint for evaluation only")
            
        else:
            # Train linear classifier
            linear_classifier, best_val_loss, linear_best_epoch = train_linear_classifier(
                encoder, train_loader, val_loader, num_classes, feature_dim,
                args.model_type, checkpoint_fold_dir, epochs=args.linear_epochs, lr=args.linear_lr
            )
            
            linear_checkpoint_path = os.path.join(checkpoint_fold_dir, f'{args.model_type}_best_linear_checkpoint.pth')
            print(f"\nLoading best linear checkpoint for testing...")
            load_linear_checkpoint(linear_checkpoint_path, linear_classifier, encoder)
            
            classifier = linear_classifier
        
        linear_results = evaluate_linear(classifier, encoder, train_loader, val_loader, test_loader, args.model_type)
        fold_results['linear'] = linear_results
        
        # ── Per-class detailed metrics for Linear ──
        for split in ['train', 'val', 'test']:
            print_detailed_metrics(
                linear_results[split]['labels'], linear_results[split]['predictions'],
                class_names, split, 'LINEAR', fold_num, args.model_type,
                checkpoint_name=f"{args.model_type}_{args.linear_checkpoint_type}_linear_checkpoint.pth"
            )
            save_detailed_metrics_to_file(
                linear_results[split]['labels'], linear_results[split]['predictions'],
                class_names, split, 'linear', fold_num, args.model_type, args.output_dir,
                checkpoint_name=f"{args.model_type}_{args.linear_checkpoint_type}_linear_checkpoint.pth"
            )

        # ── Inference timing for Linear ──
        print(f"\n{'='*80}\nINFERENCE TIMING - LINEAR (fold{fold_num})\n{'='*80}")
        _batch_sizes = [int(x) for x in args.inference_batch_sizes.split(',')]

        class _LinearInferModel(nn.Module):
            def __init__(self, enc, clf, _model_type, _device):
                super().__init__()
                self.enc        = enc
                self.clf        = clf
                self._mtype     = _model_type
                self._dev       = _device
            def forward(self, x):
                x = x.to(self._dev)
                if self._mtype == 'mae':
                    latent = self.enc.forward_encoder(x, idx_keep=None)
                    feats  = latent[:, 0, :]
                elif self._mtype == 'simmim':
                    feats = self.enc.vit.conv_proj(x)
                    feats = feats.flatten(2).transpose(1, 2)
                    cls_token = self.enc.vit.class_token.expand(feats.shape[0], -1, -1)
                    feats = torch.cat([cls_token, feats], dim=1)
                    feats = self.enc.vit.encoder(feats)
                    feats = feats[:, 0]
                else:
                    feats = self.enc(x).flatten(start_dim=1)
                return self.clf(feats)
            def eval(self):
                self.enc.eval(); self.clf.eval(); return self
            def parameters(self): return iter([])

        _lin_model = _LinearInferModel(encoder, classifier, args.model_type, device)
        _all_test_imgs_lin = torch.cat([b[0] for b in test_loader], dim=0)
        _dummy_loader_lin  = [(_all_test_imgs_lin[i:i+32], None)
                               for i in range(0, len(_all_test_imgs_lin), 32)]

        lin_timing = measure_inference_time(_lin_model, _dummy_loader_lin, _batch_sizes, device)
        print_inference_time_report(lin_timing, fold_num, args.model_type, 'LINEAR', 'test',
                                    checkpoint_name=f"{args.model_type}_{args.linear_checkpoint_type}_linear_checkpoint.pth")
        save_inference_time_to_file(lin_timing, fold_num, args.model_type, 'linear', 'test',
                                    args.output_dir,
                                    checkpoint_name=f"{args.model_type}_{args.linear_checkpoint_type}_linear_checkpoint.pth")

        if args.save_plots:
            # Save confusion matrices for all three datasets
            for split in ['train', 'val', 'test']:
                cm = confusion_matrix(linear_results[split]['labels'], linear_results[split]['predictions'])
                cm_path = os.path.join(args.output_dir,
                                      f'{args.model_type}_fold{fold_num}_linear_{split}_confusion.png')
                plot_confusion_matrix(cm, class_names, fold_num, f'Linear - {split.upper()}', cm_path, args.model_type)
    
    # FINE-TUNING EVALUATION
    if args.eval_finetune:
        print("\n" + "="*80)
        print("METHOD 3: FINE-TUNING EVALUATION")
        print("="*80)
        
        if args.load_finetune_checkpoint:
            # Load existing checkpoint without training
            print(f"\nLoading existing fine-tune checkpoint ({args.finetune_checkpoint_type})...")
            
            model_ft = load_model(args.model_type, args).to(device)
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            encoder_ft = get_encoder(model_ft, args.model_type)
            
            finetuned_model = FineTuneClassifier(encoder_ft, feature_dim, num_classes, args.model_type).to(device)
            
            finetune_checkpoint_path = os.path.join(
                checkpoint_fold_dir, 
                f'{args.model_type}_{args.finetune_checkpoint_type}_finetune_checkpoint.pth'
            )
            
            checkpoint_data = load_finetune_checkpoint(finetune_checkpoint_path, finetuned_model)
            finetune_best_epoch = checkpoint_data['epoch']
            
            print(f"Loaded {args.finetune_checkpoint_type} fine-tune checkpoint for evaluation only")
            
        else:
            # Train fine-tuned model
            model_ft = load_model(args.model_type, args).to(device)
            model_ft.load_state_dict(checkpoint['model_state_dict'])
            encoder_ft = get_encoder(model_ft, args.model_type)
            
            finetuned_model, best_val_loss, finetune_best_epoch = finetune_model(
                encoder_ft, train_loader_aug, val_loader_aug, num_classes, feature_dim,
                args.model_type, checkpoint_fold_dir, epochs=args.finetune_epochs, lr=args.finetune_lr
            )
            
            finetune_checkpoint_path = os.path.join(checkpoint_fold_dir, f'{args.model_type}_best_finetune_checkpoint.pth')
            print(f"\nLoading best fine-tuned checkpoint for testing...")
            load_finetune_checkpoint(finetune_checkpoint_path, finetuned_model)
        
        finetune_results = evaluate_finetuned(finetuned_model, train_loader, val_loader, test_loader)
        fold_results['finetune'] = finetune_results
        
        # ── Per-class detailed metrics for Fine-tune ──
        for split in ['train', 'val', 'test']:
            print_detailed_metrics(
                finetune_results[split]['labels'], finetune_results[split]['predictions'],
                class_names, split, 'FINETUNE', fold_num, args.model_type,
                checkpoint_name=f"{args.model_type}_{args.finetune_checkpoint_type}_finetune_checkpoint.pth"
            )
            save_detailed_metrics_to_file(
                finetune_results[split]['labels'], finetune_results[split]['predictions'],
                class_names, split, 'finetune', fold_num, args.model_type, args.output_dir,
                checkpoint_name=f"{args.model_type}_{args.finetune_checkpoint_type}_finetune_checkpoint.pth"
            )

        # ── Inference timing for Fine-tune ──
        print(f"\n{'='*80}\nINFERENCE TIMING - FINETUNE (fold{fold_num})\n{'='*80}")
        _batch_sizes = [int(x) for x in args.inference_batch_sizes.split(',')]

        class _FTInferModel(nn.Module):
            def __init__(self, m, _device):
                super().__init__()
                self.m    = m
                self._dev = _device
            def forward(self, x):
                return self.m(x.to(self._dev))
            def eval(self):
                self.m.eval(); return self
            def parameters(self): return iter([])

        _ft_model = _FTInferModel(finetuned_model, device)
        _all_test_imgs_ft = torch.cat([b[0] for b in test_loader], dim=0)
        _dummy_loader_ft  = [(_all_test_imgs_ft[i:i+32], None)
                              for i in range(0, len(_all_test_imgs_ft), 32)]

        ft_timing = measure_inference_time(_ft_model, _dummy_loader_ft, _batch_sizes, device)
        print_inference_time_report(ft_timing, fold_num, args.model_type, 'FINETUNE', 'test',
                                    checkpoint_name=f"{args.model_type}_{args.finetune_checkpoint_type}_finetune_checkpoint.pth")
        save_inference_time_to_file(ft_timing, fold_num, args.model_type, 'finetune', 'test',
                                    args.output_dir,
                                    checkpoint_name=f"{args.model_type}_{args.finetune_checkpoint_type}_finetune_checkpoint.pth")

        if args.save_plots:
            # Save confusion matrices for all three datasets
            for split in ['train', 'val', 'test']:
                cm = confusion_matrix(finetune_results[split]['labels'], finetune_results[split]['predictions'])
                cm_path = os.path.join(args.output_dir,
                                      f'{args.model_type}_fold{fold_num}_finetune_{split}_confusion.png')
                plot_confusion_matrix(cm, class_names, fold_num, f'Fine-tune - {split.upper()}', cm_path, args.model_type)
    
    # SUMMARY
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nTest Set Top-1 Accuracy Comparison:")
    for method, results in fold_results.items():
        if method != 'knn_all_k':
            if isinstance(results, dict) and 'test' in results:
                test_acc = results['test']['top1_accuracy']
                print(f"  {method.upper():12}: {test_acc:.4f} ({test_acc*100:.2f}%)")
            elif isinstance(results, dict) and 'top1_accuracy' in results:
                print(f"  {method.upper():12}: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
    
    if args.save_plots and len([k for k in fold_results.keys() if k != 'knn_all_k']) > 1:
        comparison_path = os.path.join(args.output_dir,
                                      f'{args.model_type}_fold{fold_num}_comparison.png')
        # Extract test results for comparison plot
        plot_methods = {}
        for k, v in fold_results.items():
            if k != 'knn_all_k':
                if isinstance(v, dict) and 'test' in v:
                    plot_methods[k] = v['test']
                else:
                    plot_methods[k] = v
        if plot_methods:
            plot_method_comparison(plot_methods, fold_num, comparison_path, args.model_type)
    
    knn_res = fold_results.get('knn', None)
    linear_res = fold_results.get('linear', None)
    finetune_res = fold_results.get('finetune', None)
    save_final_metrics(checkpoint_fold_dir, args.model_type, knn_res, linear_res, finetune_res,
                      linear_best_epoch, finetune_best_epoch)
    
    results_to_save = {}
    for method, results in fold_results.items():
        if method == 'knn_all_k':
            results_to_save[method] = [{
                'k': r['k'],
                'train': {
                    'top1_accuracy': float(r['train']['top1_accuracy']),
                    'precision': float(r['train']['precision']),
                    'recall': float(r['train']['recall']),
                    'f1_score': float(r['train']['f1_score'])
                },
                'val': {
                    'top1_accuracy': float(r['val']['top1_accuracy']),
                    'precision': float(r['val']['precision']),
                    'recall': float(r['val']['recall']),
                    'f1_score': float(r['val']['f1_score'])
                },
                'test': {
                    'top1_accuracy': float(r['test']['top1_accuracy']),
                    'precision': float(r['test']['precision']),
                    'recall': float(r['test']['recall']),
                    'f1_score': float(r['test']['f1_score'])
                }
            } for r in results]
        else:
            # Handle both old format (single dict) and new format (train/val/test dicts)
            if isinstance(results, dict) and 'test' in results:
                results_to_save[method] = {
                    'train': {
                        'top1_accuracy': float(results['train']['top1_accuracy']),
                        'precision': float(results['train']['precision']),
                        'recall': float(results['train']['recall']),
                        'f1_score': float(results['train']['f1_score'])
                    },
                    'val': {
                        'top1_accuracy': float(results['val']['top1_accuracy']),
                        'precision': float(results['val']['precision']),
                        'recall': float(results['val']['recall']),
                        'f1_score': float(results['val']['f1_score'])
                    },
                    'test': {
                        'top1_accuracy': float(results['test']['top1_accuracy']),
                        'precision': float(results['test']['precision']),
                        'recall': float(results['test']['recall']),
                        'f1_score': float(results['test']['f1_score'])
                    }
                }
            else:
                results_to_save[method] = {
                    'top1_accuracy': float(results['top1_accuracy']),
                    'precision': float(results['precision']),
                    'recall': float(results['recall']),
                    'f1_score': float(results['f1_score'])
                }
    
    all_results[f'fold{fold_num}'] = results_to_save
    
    return fold_results


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SSL Testing with Lightly (Lightly-Only Version)')
    
    # Model selection
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['simclr', 'moco', 'simsiam', 'mae', 'nnclr', 'simmim', 'byol', 'dinov2'],
                       help='SSL method to test')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help="Folder containing fold1..foldK, each with train/, val/, test/")
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--folds', type=str, default=None,
                       help='Specific folds to test (e.g., "1,3,5" or "all")')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/selfsupervised')
    parser.add_argument('--use_best', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Evaluation method selection
    parser.add_argument('--eval_knn', action='store_true')
    parser.add_argument('--eval_linear', action='store_true')
    parser.add_argument('--eval_finetune', action='store_true')
    parser.add_argument('--eval_all', action='store_true')
    
    # KNN arguments
    parser.add_argument('--k_values', type=str, default='1,3,5,7,9,11,15,20,30,50',
                       help='Comma-separated list of k values for KNN')
    
    # Linear evaluation arguments
    parser.add_argument('--linear_epochs', type=int, default=200)
    parser.add_argument('--linear_lr', type=float, default=0.001)
    parser.add_argument('--load_linear_checkpoint', action='store_true',
                       help='Load existing linear checkpoint instead of training')
    parser.add_argument('--linear_checkpoint_type', type=str, default='best',
                       choices=['best', 'last'],
                       help='Which linear checkpoint to load (best or last)')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--finetune_lr', type=float, default=0.001)
    parser.add_argument('--load_finetune_checkpoint', action='store_true',
                       help='Load existing fine-tune checkpoint instead of training')
    parser.add_argument('--finetune_checkpoint_type', type=str, default='best',
                       choices=['best', 'last'],
                       help='Which fine-tune checkpoint to load (best or last)')
    
    # Model-specific arguments
    parser.add_argument('--encoder', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--queue_size', type=int, default=4096)
    parser.add_argument('--moco_momentum', type=float, default=0.999)
    parser.add_argument('--simsiam_proj_dim', type=int, default=2048)
    parser.add_argument('--simsiam_pred_dim', type=int, default=512)
    parser.add_argument('--mae_patch_size', type=int, default=32)
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75)
    parser.add_argument('--mae_decoder_embed_dim', type=int, default=512)
    parser.add_argument('--mae_decoder_depth', type=int, default=1)
    parser.add_argument('--mae_decoder_num_heads', type=int, default=16)
    parser.add_argument('--simmim_patch_size', type=int, default=32)
    parser.add_argument('--simmim_mask_ratio', type=float, default=0.6)
    parser.add_argument('--byol_proj_dim', type=int, default=256)
    parser.add_argument('--byol_hidden_dim', type=int, default=1024)
    parser.add_argument('--dinov2_patch_size', type=int, default=16)
    parser.add_argument('--dinov2_embed_dim', type=int, default=384)
    parser.add_argument('--dinov2_drop_path_rate', type=float, default=0.1)
    parser.add_argument('--dinov2_separate_head', action='store_true')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results/selfsupervised')
    parser.add_argument('--save_plots', action='store_true')
    parser.add_argument('--inference_batch_sizes', type=str, default='1,2,4,8,16,32',
                       help='Comma-separated batch sizes for inference timing (e.g. "1,2,4,8,16,32")')
    
    args = parser.parse_args()
    
    if args.eval_all:
        args.eval_knn = True
        args.eval_linear = True
        args.eval_finetune = True
    
    if not (args.eval_knn or args.eval_linear or args.eval_finetune):
        print("No evaluation method selected, defaulting to KNN")
        args.eval_knn = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.folds is None or args.folds.lower() == 'all':
        folds_to_test = list(range(1, args.num_folds + 1))
    else:
        folds_to_test = [int(f) for f in args.folds.split(',')]
    
    print("\n" + "="*80)
    print(f"SSL TESTING WITH LIGHTLY (LIGHTLY-ONLY VERSION) - {args.model_type.upper()}")
    print("="*80)
    print(f"\nEvaluation Methods:")
    print(f"  KNN:        {'✓' if args.eval_knn else '✗'}")
    print(f"  Linear:     {'✓' if args.eval_linear else '✗'} {f'(Load {args.linear_checkpoint_type} checkpoint)' if args.load_linear_checkpoint else '(Train)'}")
    print(f"  Fine-tune:  {'✓' if args.eval_finetune else '✗'} {f'(Load {args.finetune_checkpoint_type} checkpoint)' if args.load_finetune_checkpoint else '(Train)'}")
    print(f"\nFolds to test: {folds_to_test}")
    print("="*80)
    
    all_results = {}
    
    for fold_num in folds_to_test:
        try:
            fold_results = test_fold(fold_num, args, all_results)
        except Exception as e:
            print(f"\n❌ Error testing fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        
        method_names = []
        if args.eval_knn:
            method_names.append('knn')
        if args.eval_linear:
            method_names.append('linear')
        if args.eval_finetune:
            method_names.append('finetune')
        
        for method in method_names:
            print(f"\n{method.upper()} Results:")
            print("-" * 70)
            
            # Collect test set accuracies from all folds
            train_accs = []
            val_accs = []
            test_accs = []
            
            for fold_key in all_results:
                if method in all_results[fold_key]:
                    method_results = all_results[fold_key][method]
                    # Handle nested structure (train/val/test)
                    if isinstance(method_results, dict) and 'test' in method_results:
                        train_accs.append(method_results['train']['top1_accuracy'])
                        val_accs.append(method_results['val']['top1_accuracy'])
                        test_accs.append(method_results['test']['top1_accuracy'])
                    # Handle old flat structure (for backward compatibility)
                    elif isinstance(method_results, dict) and 'top1_accuracy' in method_results:
                        test_accs.append(method_results['top1_accuracy'])
            
            if train_accs and val_accs and test_accs:
                print(f"  Train Set:")
                print(f"    Average Top-1 Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
                print(f"    Best:  {np.max(train_accs):.4f}")
                print(f"    Worst: {np.min(train_accs):.4f}")
                
                print(f"  Validation Set:")
                print(f"    Average Top-1 Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
                print(f"    Best:  {np.max(val_accs):.4f}")
                print(f"    Worst: {np.min(val_accs):.4f}")
                
                print(f"  Test Set:")
                print(f"    Average Top-1 Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
                print(f"    Best:  {np.max(test_accs):.4f}")
                print(f"    Worst: {np.min(test_accs):.4f}")
            elif test_accs:
                # Only test accuracies available (old format)
                print(f"  Average Top-1 Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
                print(f"  Best:  {np.max(test_accs):.4f}")
                print(f"  Worst: {np.min(test_accs):.4f}")
        
        results_path = os.path.join(args.output_dir, f'{args.model_type}_lightly_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*80)
    print("LIGHTLY-ONLY TESTING COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()