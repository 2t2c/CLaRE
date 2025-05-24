"""
Script to train the LaRE + CLIPping Fusion architecture pipeline.
"""

import os
import sys

os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import get_model
from datetime import datetime
from datetime import timedelta
from zoneinfo import ZoneInfo
import time
from tqdm import tqdm
import yaml
import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.data.distributed
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from rich import print as rprint
import wandb
from loss import LabelSmoothingLoss
from utils import set_seed, get_device, display_metrics, display_model_summary, load_config, display_args
from dataset import describe_dataloader, LARE
from yacs.config import CfgNode as CN
import logging

SCHEDULERS = ["single_step", "multi_step", "cosine"]
OPTIMIZERS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]

# fetch logger
logger = logging.getLogger("fomo_logger")

test_best = -1
test_best_close = -1


class StatsMeter(object):
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def build_optimizer(model, optim_cfg, param_groups=None):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
        param_groups: If provided, directly optimize param_groups and abandon model
    """
    optim = optim_cfg.name
    lr = optim_cfg.lr
    weight_decay = optim_cfg.weight_decay
    momentum = optim_cfg.momentum
    sgd_dampening = optim_cfg.sgd_dampning
    sgd_nesterov = optim_cfg.sgd_nesterov
    rmsprop_alpha = optim_cfg.rmsprop_alpha
    adam_beta1 = optim_cfg.adam_beta1
    adam_beta2 = optim_cfg.adam_beta2
    staged_lr = optim_cfg.staged_lr
    new_layers = optim_cfg.new_layers
    base_lr_mult = optim_cfg.base_lr_mult
    if optim not in OPTIMIZERS:
        raise ValueError(
            f"optim must be one of {OPTIMIZERS}, but got {optim}"
        )
    if param_groups is not None and staged_lr:
        logger.warning(
            "staged_lr will be ignored, if you need to use staged_lr, "
            "please bind it with param_groups yourself."
        )
    if param_groups is None:
        if staged_lr:
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "When staged_lr is True, model given to "
                    "build_optimizer() must be an instance of nn.Module"
                )
            if isinstance(model, nn.DataParallel):
                model = model.module
            if isinstance(new_layers, str):
                if new_layers is None:
                    logger.warning("new_layers is empty (staged_lr is useless)")
                new_layers = [new_layers]

            base_params = []
            base_layers = []
            new_params = []
            for name, module in model.named_children():
                if name in new_layers:
                    new_params += [p for p in module.parameters()]
                else:
                    base_params += [p for p in module.parameters()]
                    base_layers.append(name)
            param_groups = [
                {
                    "params": base_params,
                    "lr": lr * base_lr_mult
                },
                {
                    "params": new_params
                },
            ]
        else:
            if isinstance(model, nn.Module):
                param_groups = model.parameters()
            else:
                param_groups = model

    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )
    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )
    elif optim == "radam":
        optimizer = torch.optim.RAdam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer


class _BaseWarmupScheduler(_LRScheduler):
    def __init__(
            self, optimizer, successor, warmup_epoch, last_epoch=-1, verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    def __init__(
            self, optimizer, successor, warmup_epoch, cons_lr, last_epoch=-1, verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):
    def __init__(
            self, optimizer, successor, warmup_epoch, min_lr, last_epoch=-1, verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs]


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    :param:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.lr_scheduler
    stepsize = optim_cfg.stepsize
    gamma = optim_cfg.gamma
    max_epoch = optim_cfg.max_epoch

    if lr_scheduler not in SCHEDULERS:
        raise ValueError(
            f"scheduler must be one of {SCHEDULERS}, but got {lr_scheduler}"
        )
    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]
        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )
        if stepsize <= 0:
            stepsize = max_epoch
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )
    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    if optim_cfg.warmup_epoch > 0:
        if not optim_cfg.warmup_epoch:
            scheduler.last_epoch = optim_cfg.warmup_epoch

        if optim_cfg.warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.warmup_epoch, optim_cfg.warmup_cons_lr
            )
        elif optim_cfg.warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.warmup_epoch, optim_cfg.warmup_min_lr
            )
        else:
            raise ValueError
    return scheduler


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k. Yoinked from DASSL.

    :param:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    :returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def train_one_epoch(model, train_data_loader, val_data_loader,
                    optimizer, epoch,
                    loss_meter, auc_meter,
                    args, cfg, device, scaler,
                    step):
    # set model to training mode
    model.train()
    # set variables
    loss_meter.reset()
    auc_meter.reset()
    best_val, best_step = 0, 0
    start_time = time.time()
    epoch += 1
    pbar = tqdm(train_data_loader, desc=f"Epoch {epoch}", unit="batch")

    for batch in pbar:
        images, labels, loss_maps = batch
        images = images.to(device)
        labels = labels.to(device).flatten().squeeze()
        loss_maps = loss_maps.to(device)

        # autograd
        if cfg.clipping.coop.prec == "amp":
            with autocast():
                outputs = model(images, loss_maps)
                loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, loss_maps)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = compute_accuracy(outputs, labels)[0].item()

        # update variables
        loss_meter.update(loss.item(), images.shape[0])
        auc_meter.update(train_acc, images.shape[0])
        step += 1

        # training statistics
        if step % args.log_every == 0:
            lr = get_lr(optimizer)
            metrics = {
                "train/epoch": epoch,
                "train/acc": auc_meter.avg,
                "train/loss": loss_meter.avg,
                "train/lr": lr,
                "train/step": step,
            }
            if args.logging:
                wandb.log(metrics, step=step)
            elasped = time.time() - start_time
            display_metrics(metrics=metrics, elasped=elasped, title="Training Metrics")
            auc_meter.reset()
            loss_meter.reset()

        # validation statistics
        if step % args.eval_every == 0:
            # save directly after training to avoid errors and wasted training
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'latest.pt'))
            auc, ap, acc, r_acc, f_acc, raw_acc, raw_r_acc, raw_f_acc, best_thresh = validation_contrastive(model, val_data_loader,
                                                                                                 step, device)
            if auc > best_val:
                best_val = auc
                best_step = step
                ckpt = {
                    "state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_step": best_step
                }
                name = "best.pth"
                torch.save(ckpt, os.path.join(args.log_dir, name))
                logger.info(f'Epoch {epoch}, Step {step}: New best val accuracy, model saved.')

            # log metrics to wandb
            val_metrics = {
                "val/epoch": epoch,
                "val/roc_auc": auc,
                "val/average_precision": ap,
                "val/accuracy": acc,
                "val/real_accuracy": r_acc,
                "val/fake_accuracy": f_acc,
                "val/raw_accuracy": raw_acc,
                "val/raw_real_accuracy": raw_r_acc,
                "val/raw_fake_accuracy": raw_f_acc,
                "val/best_thresh": best_thresh,
                "val/step": step,
            }
            if args.logging:
                wandb.log(val_metrics, step=step)
            elasped = time.time() - start_time
            display_metrics(metrics=val_metrics, elasped=elasped)

            model.train()  # switch back to training mode after validation

    return step


def search_best_acc(gt_labels, pred_probs):
    best_acc = -1
    best_threshold = -1
    acc_dict = {}
    for thresh in sorted(pred_probs.tolist()):
        pred_probs_copy = np.array(pred_probs)
        pred_probs_copy[pred_probs_copy > thresh] = 1
        pred_probs_copy[pred_probs_copy <= thresh] = 0
        acc = accuracy_score(gt_labels, pred_probs_copy)
        acc_dict[thresh] = acc
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    return best_acc, best_threshold


def validation_contrastive(model, data_loader, step, device):
    """
    Validation function for the model.
    """
    logger.info('Evaluation Started!')
    model.eval()

    gt_labels_list, pred_labels_list, prob_labels_list = [], [], []
    pbar = tqdm(data_loader, desc=f"Step {step}", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            images, labels, loss_maps = batch
            images = images.to(device)
            b, C, H, W = images.shape
            images = images.reshape(b, C, H, W)
            labels = labels.flatten().squeeze().to(device)
            loss_maps = loss_maps.to(device)
            try:
                with torch.no_grad():
                    logits = model(images, loss_maps)
                    prob = torch.softmax(logits, dim=-1)  # bs * 2
            except:  # skip last batch
                logger.warning('Bad evaluation batch!', exc_info=True)
                raise
                # continue
            gt_labels_list.append(labels)
            prob_labels_list.append(prob[:, 1])

    # gather ground truth labels and predicted labels
    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)
    gt_labels_list = gt_labels.cpu().numpy()
    prob_labels_list = prob_labels.cpu().numpy()

    fpr, tpr, thres = roc_curve(gt_labels_list, prob_labels_list)
    thresh = thres[len(thres) // 2]
    logger.info(f'Thresh Old: {thresh}')
    precision, recall, thresholds = precision_recall_curve(gt_labels_list, prob_labels_list)
    f_score = precision * recall / (precision + recall)
    thresh = thresholds[np.argmax(f_score)]
    # thresh = 0.5
    logger.info(f'Thresh New: {thresh}')

    pred_labels_list = np.array(prob_labels_list)
    pred_labels_list[pred_labels_list > thresh] = 1
    pred_labels_list[pred_labels_list <= thresh] = 0

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)

    # best thresh accuracy
    best_acc, best_thresh = search_best_acc(gt_labels_list, prob_labels_list)
    logger.info(f'Search ACC: {best_acc}, Search Thresh: {best_thresh}')
    r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > best_thresh)
    f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > best_thresh)

    raw_r_acc = accuracy_score(gt_labels_list[gt_labels_list == 0], prob_labels_list[gt_labels_list == 0] > 0.5)
    raw_f_acc = accuracy_score(gt_labels_list[gt_labels_list == 1], prob_labels_list[gt_labels_list == 1] > 0.5)
    raw_acc = accuracy_score(gt_labels_list, prob_labels_list > 0.5)

    return auc, ap, best_acc, r_acc, f_acc, raw_acc, raw_r_acc, raw_f_acc, best_thresh


def train(args):
    # load the config file
    with open(args.config, 'r') as f:
        config_file = yaml.safe_load(f)
    # convert to yacs
    cfg = load_config(config_file)
    # add args inside cfg as CfgNode
    cfg.args = CN(vars(args))
    # pretty print args
    display_args(args, title="Config Arguments")

    # setup wandb
    if args.logging:
        wandb.init(
            project=args.project,
            entity="FoMo",
            name=args.run_name + "/" + args.uid,
            config={
                "architecture": args.model,
                "clip_type": args.clip_type,
                "batch_size": args.batch_size,
                "log_dir": args.log_dir,
                "seed": args.seed,
                "mode": args.mode,
                "device": args.device,
                "eval_every": args.eval_every,
                "log_every": args.log_every,
                "epochs": args.epochs,
                "train_dataset": cfg.dataset.train_dataset,
                "test_dataset": cfg.dataset.test_dataset,
            },
            settings=wandb.Settings(_service_wait=300, init_timeout=120))

    # load model
    model = get_model(name=args.model, clip_type=args.clip_type, cfg=cfg)
    # freeze parameters
    for name, param in model.named_parameters():
        if not any([
            "prompt_learner" in name,
            "conv" in name,
            "conv_align" in name,
            "attn_pool" in name,
            "channel_align" in name,
            "image_proj" in name,
            "text_proj" in name,
            "roi_pool" in name,
        ]):
            param.requires_grad_(False)
    device = get_device(args.device)
    model.to(device)
    display_model_summary(model, input_shape=(1, 3, cfg.dataset.resolution,
                                              cfg.dataset.resolution), device=device)

    # load training data
    train_dataset = LARE(config=cfg.dataset,
                        mode="train",
                        jpeg_quality=cfg.dataset.jpeg_quality,
                        debug=args.debug)
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=None)
    describe_dataloader(train_data_loader, title="Train DataLoader Summary")

    # load validation data
    val_dataset = LARE(config=cfg.dataset,
                      mode="test",
                      jpeg_quality=cfg.dataset.jpeg_quality,
                      debug=args.debug)
    val_data_loader = DataLoader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=None)
    describe_dataloader(val_data_loader, title="Val DataLoader Summary")

    # setting optimizer, scaler, and scheduler
    scaler = GradScaler() if cfg.clipping.coop.prec == "amp" else None
    # change default config
    cfg.clipping.optim.max_epoch = args.epochs
    optimizer = build_optimizer(model.prompt_learner, cfg.clipping.optim)
    scheduler = build_lr_scheduler(optimizer, cfg.clipping.optim)

    # setting scheduler
    loss_meter = StatsMeter()
    auc_meter = StatsMeter()

    # starting the training
    logger.info(f"Training Started! - Debugging: {args.debug}")
    step = 0
    for epoch in range(args.epochs):
        # train one epoch
        step = train_one_epoch(model, train_data_loader, val_data_loader,
                               optimizer, epoch,
                               loss_meter, auc_meter, args, cfg,
                               device, scaler, step)
        # scheduler step
        scheduler.step()

    # dump the config
    with open(f"{args.log_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())

    if args.logging:
        # exit session
        wandb.finish()
