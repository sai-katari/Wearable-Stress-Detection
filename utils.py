import numpy as np
from sklearn.metrics import (f1_score, recall_score, precision_score,
                              accuracy_score, roc_auc_score, confusion_matrix)


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc'] = 0.0
    return metrics


def print_metrics(metrics, prefix=''):
    print(f'{prefix}F1: {metrics["f1"]:.4f} | '
          f'Recall: {metrics["recall"]:.4f} | '
          f'Precision: {metrics["precision"]:.4f} | '
          f'Accuracy: {metrics["accuracy"]:.4f}' +
          (f' | AUC: {metrics["auc"]:.4f}' if 'auc' in metrics else ''))


def get_class_weights(labels):
    # compute class weights to handle stress/non-stress imbalance
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts)
    return weights.astype(np.float32)