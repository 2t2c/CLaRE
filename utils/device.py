import torch

from .logger import LOGGER


def get_device(type: str):
    if type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif type == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    LOGGER.info(f"Using device: {device}")

    return device
