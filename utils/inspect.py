from collections import Counter
from datetime import timedelta

import torch
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from torchinfo import summary

from utils.logger import LOGGER


def display_cfg(args, title="Arguments"):
    """Nicely print argparse.Namespace, CfgNode, or dict using rich."""
    if hasattr(args, "dump"):  # handle CfgNode
        args = yaml.safe_load(args.dump())
    elif not isinstance(args, dict):
        args = vars(args)

    table = Table(title=title)
    table.add_column("Argument", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in args.items():
        table.add_row(str(key), str(value))

    console = Console()
    console.print(table)


def display_metrics(metrics: dict, elapsed: float, title: str = "Validation Metrics"):
    """
    Nicely print metric dictionary using rich.

    :param:
        metrics: Dictionary of metrics (e.g., {"AUC": 0.95, "ACC": 0.88, ...})
    :param:
        title: Optional table title
    """
    rprint("Time Elasped:", str(timedelta(seconds=elapsed)))
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(
            str(key), f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
        )

    console = Console()
    console.print(table)


def display_model_summary(model, input_shape=None, title="Model Summary", device="cpu"):
    """
    Display model parameter summary using rich. Supports multiple inputs.

    :param:
        model: PyTorch model
    :param:
        input_shape: A single shape tuple or list/tuple e.g., (1, 3, 224, 224)
    :param:
        title: Optional table title
    :param:
        device: Device for dummy inputs
    """
    model = model.to(device)
    table = Table(title=title)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    table.add_row("Model", model.__class__.__name__)
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Trainable (Million)", f"{trainable_params / (1024**2):.2f} Million")
    if hasattr(model, "clip_model"):
        arch = str(summary(model.clip_model, depth=1, verbose=0))
    else:
        arch = str(summary(model, depth=1, verbose=0))
    table.add_row("Summary", arch)

    if input_shape is not None:
        if hasattr(model, "clip_model"):
            table.add_row("Input Shape", str(input_shape))
            visual_arch = str(
                summary(
                    model.clip_model.visual, input_size=input_shape, depth=1, verbose=0
                )
            )
            table.add_row("CLIP.Visual", visual_arch)

    console = Console()
    console.print(table)


def describe_dataloader(dataloader):
    """
    Method to print dataset statistics from a PyTorch DataLoader:
    - Total number of samples
    - Total number of batches
    - Class distribution (if available)
    - Sample data shape and dtype
    """
    console = Console()
    dataset = dataloader.dataset
    total_samples = len(dataset)
    total_batches = len(dataloader)

    table = Table(title="DataLoader Summary")

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total samples", str(total_samples))
    table.add_row(
        f"Total batches (batch_size={dataloader.batch_size})", str(total_batches)
    )
    table.add_row("Num Workers", str(dataloader.num_workers))

    # class info
    class_info_found = False
    if hasattr(dataset, "classes"):
        table.add_row("Classes", str(dataset.classes))
        class_info_found = True
    if hasattr(dataset, "class_to_idx"):
        table.add_row("Class to index mapping", str(dataset.class_to_idx))
        class_info_found = True
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        label_counts = Counter(targets.tolist())
        table.add_row("Label counts", str(dict(label_counts)))
        class_info_found = True
    if not class_info_found:
        table.add_row(
            "Class/Label info", "No class/label info found in dataset attributes."
        )

    # sample data shape and dtype
    try:
        first_batch = next(iter(dataloader))
        if isinstance(first_batch, (list, tuple)):
            # Show shape of first input and sample label summary
            shape_info = str(first_batch[0].shape)
            label_info = str(first_batch[1])
            table.add_row("Input sample shape", shape_info)
            table.add_row("Label sample", label_info)
        elif isinstance(first_batch, dict):
            table.add_row("Sample keys", str(list(first_batch.keys())))
            for key, value in first_batch.items():
                if hasattr(value, "shape"):
                    shape = tuple(value.shape)
                else:
                    shape = "N/A"
                dtype = getattr(value, "dtype", type(value).__name__)
                table.add_row(
                    f"{key.capitalize()} shape & dtype",
                    str(shape) + f", ({str(dtype)})",
                )
        else:
            table.add_row("Sample", str(type(first_batch)))
    except Exception as e:
        table.add_row("Sample inspection error", str(e))
        LOGGER.warning(str(e))

    console.print(table)
