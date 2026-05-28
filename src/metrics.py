"""Metryki segmentacji liczone tylko w FOV (accuracy, sens, spec, G-mean, ...)."""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import confusion_matrix


def _counts_in_fov(
    pred: np.ndarray,
    gt: np.ndarray,
    fov: np.ndarray,
) -> tuple[int, int, int, int]:
    """(TN, FP, FN, TP) wewnątrz FOV."""
    if pred.shape != gt.shape or pred.shape != fov.shape:
        raise ValueError(
            f"Niezgodność rozmiarów: pred={pred.shape}, gt={gt.shape}, fov={fov.shape}"
        )
    f = (fov > 0).ravel()
    if not f.any():
        raise ValueError("Maska FOV jest pusta.")

    y_true = (gt.ravel()[f] > 0).astype(np.uint8)
    y_pred = (pred.ravel()[f] > 0).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def confusion_matrix_fov(
    pred: np.ndarray,
    gt: np.ndarray,
    fov: np.ndarray,
) -> dict[str, int]:
    """{'tp', 'fp', 'tn', 'fn'} w obrębie FOV."""
    tn, fp, fn, tp = _counts_in_fov(pred, gt, fov)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    fov: np.ndarray,
) -> dict[str, float]:
    # TP - naczynie poprawnie wykryte
    # FP - tło błędnie wykryte jako naczynie,
    # TN - tło poprawnie wykryte,
    # FN - naczynie błędnie wykryte jako tło.
    tn, fp, fn, tp = _counts_in_fov(pred, gt, fov)
    total = tp + tn + fp + fn

    accuracy    = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision   = tp / (tp + fp) if (tp + fp) else 0.0
    balanced    = 0.5 * (sensitivity + specificity)
    g_mean      = math.sqrt(max(0.0, sensitivity * specificity))

    return {
        "accuracy":          float(accuracy),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "precision":         float(precision),
        "g_mean":            float(g_mean),
        "balanced_accuracy": float(balanced),
        "arith_mean":        float(balanced),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }


def aggregate(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Mean/std po obrazach (ddof=1). Liczniki pikselowe sumowane jako *_total."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    out: dict[str, float] = {}
    n = len(metrics_list)
    for k in keys:
        values = np.array([m[k] for m in metrics_list], dtype=np.float64)
        if k in {"tp", "fp", "tn", "fn"}:
            out[f"{k}_total"] = float(values.sum())
        else:
            out[f"{k}_mean"] = float(values.mean())
            out[f"{k}_std"] = float(values.std(ddof=1)) if n >= 2 else 0.0
    return out
