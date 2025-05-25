import torch
from torch.optim.lr_scheduler import _LRScheduler

SCHEDULERS = ["single_step", "multi_step", "cosine"]


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
