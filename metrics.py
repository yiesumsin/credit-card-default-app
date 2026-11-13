"""Small evaluation helpers (confusion matrix, ROC curve, AUC) written with plain numpy.

Writing them here means the app needs no scikit-learn, which keeps the hosted app small and quick to start.
"""
import numpy as np


def confusion_counts(y_true, probability, threshold):
    """Returns (tp, fp, tn, fn) when everyone with probability >= threshold is called a defaulter."""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(probability) >= threshold
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    return tp, fp, tn, fn


def scores_from_counts(tp, fp, tn, fn):
    """Recall, precision, F1 and accuracy from the four confusion-matrix counts (0 when undefined)."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return dict(recall=recall, precision=precision, f1=f1, accuracy=accuracy)


def roc_points(y_true, probability):
    """False-positive rate and true-positive rate for every cutoff, plus the area under the curve (AUC)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probability, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    y_sorted, p_sorted = y[order], p[order]
    # Keep one point per distinct probability so ties do not create fake steps
    distinct = np.where(np.diff(p_sorted))[0]
    cut = np.r_[distinct, len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[cut]
    fps = (1 + cut) - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc
