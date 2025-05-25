import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from data.module import NamedDataLoader
from utils import LOGGER, display_metrics


class StatsMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class EvaluationMetrics:
    auc: float
    best_acc: float
    average_precision: float
    f1_based_thresh_acc: float
    raw_acc: float
    real_acc: float
    fake_acc: float


def run_test(
    model: Module,
    test_loader: DataLoader,
    cfg: CfgNode,
    device: torch.device,
) -> dict:
    """Run testing/evaluation loop and log/display metrics."""
    start_time = time.time()

    with torch.no_grad():
        metrics = compute_metrics(model, test_loader, device)

    test_metrics = {
        "test/roc_auc": metrics.auc,
        "test/accuracy": metrics.best_acc,
        "test/average_precision": metrics.average_precision,
        "test/f1_based_threshold_accuracy": metrics.f1_based_thresh_acc,
        "test/raw_accuracy": metrics.raw_acc,
        "test/real_accuracy": metrics.real_acc,
        "test/fake_accuracy": metrics.fake_acc,
    }

    if cfg.wandb:
        wandb.log(test_metrics)

    display_metrics(metrics=test_metrics, elapsed=time.time() - start_time)
    return test_metrics


def run_validation(
    model: Module,
    val_loader: DataLoader,
    cfg: CfgNode,
    step: int,
    epoch: int,
    device: torch.device,
    start_time: float,
) -> EvaluationMetrics:
    """Runs validation and log/display metrics."""
    LOGGER.info(f"Running validation at epoch {epoch}, step {step}...")
    metrics = compute_metrics(model, val_loader, device, step)

    val_metrics = {
        "val/epoch": epoch,
        "val/roc_auc": metrics.auc,
        "val/accuracy": metrics.best_acc,
        "val/average_precision": metrics.average_precision,
        "val/f1_based_threshold_accuracy": metrics.f1_based_thresh_acc,
        "val/raw_accuracy": metrics.raw_acc,
        "val/real_accuracy": metrics.real_acc,
        "val/fake_accuracy": metrics.fake_acc,
    }

    if cfg.wandb:
        wandb.log(val_metrics, step=step)

    display_metrics(metrics=val_metrics, elapsed=time.time() - start_time)
    return metrics


def compute_metrics(
    model: Module,
    data_loader: NamedDataLoader,
    device: torch.device,
    step: int | None = None,
) -> EvaluationMetrics:
    LOGGER.info("Evaluation Started!")
    gt_labels, prob_scores = [], []

    model.eval()
    with torch.no_grad():
        desc = f"Step {step}" if step is not None else data_loader.name
        for images, labels in tqdm(data_loader, desc=desc, unit="batch"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            prob = torch.softmax(logits, dim=-1)[:, 1]

            gt_labels.append(labels)
            prob_scores.append(prob)

    gt = torch.cat(gt_labels).cpu().numpy()
    prob = torch.cat(prob_scores).cpu().numpy()

    # Compute ROC-based threshold
    fpr, tpr, thresholds_roc = roc_curve(gt, prob)
    LOGGER.info(f"Thresh Old (ROC mid): {thresholds_roc[len(thresholds_roc) // 2]}")

    # Compute precision-recall-based threshold
    precision, recall, thresholds_pr = precision_recall_curve(gt, prob)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    f1_score_for_thresh = f1_score[:-1]
    best_idx = np.argmax(f1_score_for_thresh)
    best_pr_thresh = thresholds_pr[best_idx]
    LOGGER.info(f"Thresh New (PR best-F1): {best_pr_thresh}")

    # Apply threshold to get predicted labels
    preds = np.where(prob > best_pr_thresh, 1, 0)
    f1_based_thresh_acc = accuracy_score(gt, preds)

    auc = roc_auc_score(gt, prob)
    ap = average_precision_score(gt, prob)
    best_acc, best_thresh = search_best_acc(gt, prob)
    LOGGER.info(f"Search ACC: {best_acc}, Search Thresh: {best_thresh}")

    real_acc = accuracy_score(gt[gt == 0], prob[gt == 0] > 0.5)
    fake_acc = accuracy_score(gt[gt == 1], prob[gt == 1] > 0.5)
    raw_acc = accuracy_score(gt, prob > 0.5)

    model.train()

    return EvaluationMetrics(
        auc=auc,
        best_acc=best_acc,
        average_precision=ap,
        f1_based_thresh_acc=f1_based_thresh_acc,
        raw_acc=raw_acc,
        real_acc=real_acc,
        fake_acc=fake_acc,
    )


def compute_accuracy(
    output: Tensor | Sequence[Tensor],
    target: Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[Tensor]:
    """Computes the accuracy over the k top predictions for specified values of k.

    Args:
        output: Prediction matrix (batch_size, num_classes) or tuple with first
            element as predictions.
        target: Ground truth labels (batch_size).
        topk: The top-k values to compute accuracy for.

    Returns:
        List of accuracies at each top-k.
    """
    if isinstance(output, (tuple, list)):
        output = output[0]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def search_best_acc(
    gt_labels: np.ndarray, pred_probs: np.ndarray
) -> tuple[float, float]:
    """Searches for the threshold that gives the highest accuracy.

    Args:
        gt_labels: Ground truth binary labels.
        pred_probs: Predicted probabilities for the positive class.

    Returns:
        A tuple containing the best accuracy and its corresponding threshold.
    """
    thresholds = np.unique(pred_probs)
    best_acc = -1.0
    best_threshold = -1.0

    for thresh in thresholds:
        preds = (pred_probs > thresh).astype(int)
        acc = accuracy_score(gt_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    return best_acc, best_threshold
