import torch
from torch.nn import Module
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from yacs.config import CfgNode

from .logger import LOGGER


def save_checkpoint(
    path: str,
    model: Module,
    optimizer: Optimizer,
    epoch: int | None = None,
    step: int | None = None,
    scaler: GradScaler | None = None,
    cfg: CfgNode | None = None,
    **additional_info,
):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if cfg is not None:
        checkpoint["cfg"] = cfg.dump()

    checkpoint.update(additional_info)
    torch.save(checkpoint, path)

    desc = f"Checkpoint saved to {path}"
    if epoch is not None and step is not None:
        desc += f"at epoch {epoch}, step {step}"

    LOGGER.info(desc)


def load_checkpoint(
    path: str,
    model: Module,
    optimizer: Optimizer,
    scaler: GradScaler | None = None,
) -> tuple[int, int, str | None]:
    LOGGER.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch: int = checkpoint.get("epoch", -1)
    step: int = checkpoint.get("step", -1)
    cfg_str = checkpoint.get("cfg")

    LOGGER.info(f"Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step, cfg_str
