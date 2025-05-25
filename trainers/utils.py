import os
import time
from typing import Callable

import torch
import wandb
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from utils import LOGGER, display_metrics, save_checkpoint

from .evaluation import StatsMeter, compute_accuracy, run_validation


def train_one_epoch(
    model: Module,
    step: int,
    best_val_acc: float,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    loss_meter: StatsMeter,
    acc_meter: StatsMeter,
    cfg: CfgNode,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
    criterion: Callable,
    clip_grad: bool,
    max_grad_norm: float = 5.0,
) -> tuple[int, float]:
    """Trains model for one epoch."""
    model.train()

    best_step = 0
    loss_meter.reset()
    acc_meter.reset()
    start_time = time.time()

    for batch in tqdm(train_data_loader, desc=f"Epoch {epoch}", unit="batch"):
        outputs, labels, loss = train_step(
            model,
            batch,
            optimizer,
            device,
            use_amp,
            scaler,
            criterion,
            clip_grad,
            max_grad_norm,
        )

        acc = compute_accuracy(outputs, labels)[0].item()
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))
        step += 1

        if step % cfg.log_every == 0:
            log_train_metrics(
                epoch, step, cfg, loss_meter, acc_meter, optimizer, start_time
            )

        if step % cfg.eval_every == 0:
            save_dir = os.path.join(cfg.output_dir, "latest.pth")
            save_checkpoint(save_dir, model, optimizer, epoch, step, scaler, cfg)

            val_metrics = run_validation(
                model,
                val_data_loader,
                cfg,
                step,
                epoch,
                device,
                start_time,
            )

            if val_metrics.best_acc <= best_val_acc:
                continue

            best_val_acc = val_metrics.best_acc
            best_step = step

            info = {
                "best_val_acc": best_val_acc,
                "best_step": best_step,
            }
            save_dir = os.path.join(cfg.output_dir, "best.pth")
            save_checkpoint(
                save_dir, model, optimizer, epoch, step, scaler, cfg, **info
            )
            msg = f"Epoch {epoch}, Step {step}: New best val accuracy, model saved."
            LOGGER.info(msg)

    return step, best_val_acc


def train_step(
    model: Module,
    batch: tuple[Tensor, Tensor],
    optimizer: Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
    criterion: Callable[[Tensor, Tensor], Tensor],
    clip_grad: bool,
    max_grad_norm: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Performs a single training step.

    Args:
        model: The model to train.
        batch: A tuple of input tensors and corresponding labels.
        optimizer: Optimizer used for parameter updates.
        device: The device to move data and model to.
        use_amp: Whether to use automatic mixed precision (AMP).
        scaler: Gradient scaler for AMP. Must be provided if use_amp is True.
        loss_fn: Loss function to compute training loss.
        clip_grad: Whether to apply gradient clipping.
        max_grad_norm: Maximum norm for gradient clipping.

    Returns:
        A tuple of (model output, target labels, loss).
    """
    if use_amp:
        assert scaler is not None, "AMP enabled but GradScaler is None."

    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    if use_amp and scaler is not None:
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()

        if clip_grad:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if clip_grad:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

    return outputs, targets, loss


def log_train_metrics(
    epoch: int,
    step: int,
    cfg: CfgNode,
    loss_meter: StatsMeter,
    acc_meter: StatsMeter,
    optimizer: Optimizer,
    start_time: float,
):
    """Logs training metrics."""
    metrics = {
        "train/epoch": epoch,
        "train/acc": acc_meter.avg,
        "train/loss": loss_meter.avg,
        "train/lr": get_lr(optimizer),
        "train/step": step,
    }

    if cfg.wandb:
        wandb.log(metrics, step=step)

    elapsed_time = time.time() - start_time
    display_metrics(metrics=metrics, elapsed=elapsed_time, title="Training Metrics")
    loss_meter.reset()
    acc_meter.reset()


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]
