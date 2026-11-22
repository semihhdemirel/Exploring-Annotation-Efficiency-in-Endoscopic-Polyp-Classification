"""Shared evaluation metrics used by the supervised and semi-supervised
pipelines, so both report identical definitions.

Definitions
-----------
* Overall precision / recall / F1 / specificity are *support-weighted*
  averages over classes (weights = number of true samples per class).
* Specificity of class c is TN_c / (TN_c + FP_c) in a one-vs-rest sense.
* 95% confidence intervals are percentile bootstrap intervals
  (default 1000 resamples, fixed seed).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


# --------------------------------------------------------------------------
# Confusion-matrix helpers
# --------------------------------------------------------------------------
def per_class_counts(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """TP / FP / FN / TN per class from a confusion matrix (rows = truth)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def specificity_per_class(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    c = per_class_counts(cm)
    denom = c["tn"] + c["fp"]
    return np.divide(c["tn"], denom, out=np.zeros_like(denom), where=denom > 0)


def specificity_overall(y_true, y_pred, num_classes: int,
                        average: str = "weighted") -> float:
    """Overall specificity. average = 'weighted' (by class support) or 'macro'."""
    spec = specificity_per_class(y_true, y_pred, num_classes)
    if average == "macro":
        return float(spec.mean())
    if average != "weighted":
        raise ValueError("average must be 'weighted' or 'macro'")
    support = np.bincount(np.asarray(y_true), minlength=num_classes)
    if support.sum() == 0:
        return 0.0
    return float(np.average(spec, weights=support))


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric_fn: Callable, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42):
    """Percentile bootstrap CI for metric_fn(y_true, y_pred) -> float."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:  # degenerate resample
            continue
    if not scores:
        return float("nan"), float("nan")
    a = (1 - ci) / 2
    return (float(np.percentile(scores, a * 100)),
            float(np.percentile(scores, (1 - a) * 100)))


# --------------------------------------------------------------------------
# Full metric computation
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names: Sequence[str],
                    n_bootstrap: int = 1000,
                    specificity_average: str = "weighted") -> dict:
    """Overall + per-class metrics with bootstrap CIs and confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(class_names)
    labels = list(range(k))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    counts = per_class_counts(cm)

    overall_fns = {
        "accuracy": lambda t, p: accuracy_score(t, p),
        "precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
        "recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
        "f1": lambda t, p: f1_score(t, p, average="weighted", zero_division=0),
        "specificity": lambda t, p: specificity_overall(t, p, k, specificity_average),
    }
    overall = {}
    for name, fn in overall_fns.items():
        lo, hi = bootstrap_ci(y_true, y_pred, fn, n_bootstrap)
        overall[name] = {"value": float(fn(y_true, y_pred)), "ci": (lo, hi)}

    def _bin(fn_name, c):
        def f(t, p):
            tb, pb = (t == c).astype(int), (p == c).astype(int)
            if fn_name == "precision":
                return precision_score(tb, pb, zero_division=0)
            if fn_name == "recall":
                return recall_score(tb, pb, zero_division=0)
            if fn_name == "f1":
                return f1_score(tb, pb, zero_division=0)
            return specificity_per_class(t, p, k)[c]
        return f

    per_class = {}
    for c, cname in enumerate(class_names):
        entry = {"support": int(cm[c].sum()),
                 "tp": int(counts["tp"][c]), "fp": int(counts["fp"][c]),
                 "fn": int(counts["fn"][c]), "tn": int(counts["tn"][c])}
        for m in ("precision", "recall", "f1", "specificity"):
            fn = _bin(m, c)
            lo, hi = bootstrap_ci(y_true, y_pred, fn, n_bootstrap)
            entry[m] = {"value": float(fn(y_true, y_pred)), "ci": (lo, hi)}
        per_class[cname] = entry

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, labels=labels,
                                        target_names=list(class_names),
                                        digits=4, zero_division=0),
        "specificity_average": specificity_average,
        "n_samples": int(len(y_true)),
    }


def format_metrics(res: dict, class_names: Sequence[str], title: str = "") -> str:
    """Human-readable text block."""
    sep = "=" * 100
    out: List[str] = [sep, title, sep, "",
                      f"-- Overall (support-weighted; specificity: {res['specificity_average']}) --"]
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
              "f1": "F1 Score", "specificity": "Specificity"}
    for key, lab in labels.items():
        m = res["overall"][key]
        out.append(f"  {lab + ':':<13}{m['value']:.4f}  (95% CI: [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}])")
    out += ["", "-- Per-class --"]
    for cname in class_names:
        pc = res["per_class"][cname]
        short = {"precision": "prec", "recall": "rec", "f1": "f1", "specificity": "spec"}
        cells = "  ".join(f"{short[m]}={pc[m]['value']:.4f} [{pc[m]['ci'][0]:.4f},{pc[m]['ci'][1]:.4f}]"
                          for m in short)
        out.append(f"  {cname:<16}n={pc['support']:<6} {cells}")
        out.append(f"  {'':<16}TP={pc['tp']} FP={pc['fp']} FN={pc['fn']} TN={pc['tn']}")
    out += ["", "-- Confusion matrix (rows = true, cols = predicted) --",
            "  " + " " * 16 + "".join(f"{n:>12}" for n in class_names)]
    for i, n in enumerate(class_names):
        out.append(f"  {n:<16}" + "".join(f"{v:>12}" for v in res["confusion_matrix"][i]))
    out += ["", "-- Classification report --", res["report"], sep]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Inference benchmark (forward pass only; data already in memory)
# --------------------------------------------------------------------------
def benchmark_forward(forward_fn: Callable, images, batch_sizes: Sequence[int],
                      device, n_warmup: int = 10) -> Dict[int, dict]:
    """Time forward_fn over an in-memory image tensor for several batch sizes.

    Data loading / decoding / augmentation is deliberately excluded so the
    numbers reflect model cost only.
    """
    import torch  # local import keeps common.metrics importable without torch

    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    n = len(images)
    results = {}
    for bs in batch_sizes:
        with torch.no_grad():
            warm = images[:min(bs, n)].to(device)
            for _ in range(n_warmup):
                forward_fn(warm)
            sync()
            t0 = time.perf_counter()
            for i in range(0, n, bs):
                forward_fn(images[i:i + bs].to(device))
            sync()
            dt = time.perf_counter() - t0
        results[bs] = {"total_time_s": dt, "total_images": n,
                       "ms_per_image": 1000 * dt / n, "throughput_img_s": n / dt}
    return results
