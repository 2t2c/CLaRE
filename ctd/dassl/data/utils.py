import numpy as np
from PIL import Image
import torch
import logging
from rich.table import Table
from rich.console import Console
from collections import Counter
from io import BytesIO


logger = logging.getLogger("fomo_logger")


def png2jpg(img, quality):
    # check if the img in right
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    out = BytesIO()
    # ranging from 0-95, 75 is default
    img.save(out, format="jpeg", quality=quality)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()

    return img


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

    console.print(table)
