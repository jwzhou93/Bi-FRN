"""Utility helpers for computing classification metrics."""

from typing import Tuple

import torch


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a confusion matrix.

    Args:
        preds: Tensor containing predicted class indices (shape: ``[N]``).
        targets: Tensor containing ground-truth class indices (shape: ``[N]``).
        num_classes: Number of classes in the classification task.

    Returns:
        A ``num_classes x num_classes`` tensor with counts where ``[i, j]``
        corresponds to the number of examples whose ground-truth label is ``i``
        and predicted label is ``j``.
    """

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def precision_recall_f1(cm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision, recall and F1-score from a confusion matrix."""

    tp = cm.diag().float()
    precision = torch.where(
        cm.sum(0) > 0, tp / cm.sum(0).float(), torch.zeros_like(tp)
    )
    recall = torch.where(
        cm.sum(1) > 0, tp / cm.sum(1).float(), torch.zeros_like(tp)
    )
    f1_denominator = precision + recall
    f1 = torch.where(
        f1_denominator > 0,
        2 * precision * recall / f1_denominator,
        torch.zeros_like(tp),
    )
    return precision, recall, f1

