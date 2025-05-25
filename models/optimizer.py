import torch
from torch import nn

from utils import LOGGER

OPTIMIZERS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]


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
        raise ValueError(f"optim must be one of {OPTIMIZERS}, but got {optim}")
    if param_groups is not None and staged_lr:
        LOGGER.warning(
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
                    LOGGER.warning("new_layers is empty (staged_lr is useless)")
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
                {"params": base_params, "lr": lr * base_lr_mult},
                {"params": new_params},
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
