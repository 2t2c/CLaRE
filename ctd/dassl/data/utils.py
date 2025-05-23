import logging
from concurrent.futures import ThreadPoolExecutor
import json
from functools import partial
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table

LOGGER = logging.getLogger("dataset_manifest")
LOGGER.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

if not LOGGER.hasHandlers():
    LOGGER.addHandler(console_handler)


def get_image_paths(path: Path, images_per_video: Optional[int] = None) -> list[str]:
    """Recursively collects paths to all image files in a directory and its subdirectories.

    Parameters:
        path: The root directory to search for images.
        images_per_video: Optional limit on how many images to return.

    Returns:
        List of image file paths relative to `path`.
    """
    LOGGER.debug(f'Retrieving images of "{path.name}"...')
    extensions = {".jpg", ".jpeg", ".png"}
    paths = []

    if not path.exists():
        LOGGER.warning(f'Can\'t read images from "{path}" as it does not exist!')
        return path

    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p.relative_to(path).as_posix())

    LOGGER.debug(f'Retrieved images of "{path.name}"')
    return paths


def create_dataset_manifest(
    dataset_root: Path,
    manifest_path: Path,
    subsets_to_include: dict[str, list[str]],
    images_per_video: Optional[int] = None,
    max_workers: Optional[int] = None,
):
    LOGGER.info("** Creating dataset manifest **")
    data = {"root": str(dataset_root.absolute())}

    processors = {
        "train": process_train_datasets,
        "test": process_test_datasets,
    }

    inputs = [
        (
            processors[split],
            (
                dataset_root / split,
                subsets_to_include[split],
                images_per_video,
                max_workers,
            ),
        )
        for split in ["train", "test"]
    ]

    def run_processor(args):
        processor, inputs = args
        return processor(*inputs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(run_processor, inputs)

    for split, result in zip(["train", "test"], results):
        data[split] = result

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing to file...")

    with manifest_path.open("w") as f:
        json.dump(data, f, indent=2)

    LOGGER.info(f'Created dataset manifest at "{manifest_path.as_posix()}"')


def process_train_datasets(
    split_dir: Path,
    subset_names: list[str],
    images_per_video: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> dict:
    LOGGER.info("Processing train subsets...")
    data = {}

    def include_subset(subset: Path, split_subsets: list[str]) -> bool:
        if not subset.is_dir():
            return False

        return subset.name in split_subsets

    def process_subset(subset: Path) -> tuple[str, list[str]]:
        LOGGER.info(f'- Processing subset "{subset.name}"...')
        image_paths = get_image_paths(subset, images_per_video)
        return subset.name, image_paths

    for classname in ["fake", "real"]:
        LOGGER.info(f'Scanning for subsets for class "{classname}"...')
        data[classname] = {}
        path = split_dir / classname

        if not path.exists():
            LOGGER.warning(f'Directory for "{classname}" does not exist! "{path}"')
            continue

        subsets = [p for p in path.iterdir() if include_subset(p, subset_names)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_subset, subsets)

            for subset_name, image_paths in results:
                data[classname][subset_name] = image_paths
                LOGGER.info(f' - Loaded subset "{subset_name}"')

    LOGGER.info("Finished processing train subsets!")
    return data


def process_test_datasets(
    split_dir: Path,
    subset_names: list[str],
    images_per_video: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> dict:
    LOGGER.info("Processing test subsets...")
    data = {}

    def preload_real_images() -> dict[str, list[str]]:
        LOGGER.info("Preloading real images...")
        real_imgs_cache = {}

        for subset in subset_names:
            if subset.endswith("-ff") and "ff" not in real_imgs_cache:
                ff_path = split_dir / "real" / "face-forensics"
                real_imgs_cache["ff"] = get_image_paths(ff_path, images_per_video)
            elif subset.endswith("-cdf") and "cdf" not in real_imgs_cache:
                cdf_path = split_dir / "real" / "celeb-df"
                real_imgs_cache["cdf"] = get_image_paths(cdf_path, images_per_video)

        LOGGER.info("Preloaded real images")
        return real_imgs_cache

    def process_subset(subset: str, real_imgs_cache: dict) -> tuple[str, dict]:
        LOGGER.info(f'- Processing subset "{subset}"...')
        subset_path = split_dir / "fake" / subset
        if not subset_path.exists():
            LOGGER.warning(f'Directory for "{subset}" does not exist! "{subset_path}"')
            return subset, {}

        subset_data = {}

        if subset.endswith("-ff"):
            subset_data["fake"] = get_image_paths(subset_path, images_per_video)
            subset_data["real"] = real_imgs_cache["ff"]
        elif subset.endswith("-cdf"):
            subset_data["fake"] = get_image_paths(subset_path, images_per_video)
            subset_data["real"] = real_imgs_cache["cdf"]
        else:
            subset_data["fake"] = get_image_paths(
                subset_path / "fake", images_per_video
            )
            subset_data["real"] = get_image_paths(
                subset_path / "real", images_per_video
            )

        LOGGER.info(f' - Loaded subset "{subset}"')
        return subset, subset_data

    real_imgs_cache = preload_real_images()
    process_fn = partial(process_subset, real_imgs_cache=real_imgs_cache)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_fn, subset_names)

    for subset_name, subset_data in results:
        data[subset_name] = subset_data

    LOGGER.info("Finished processing test subsets!")
    return data


def process_test_datasets_x(
    split_dir: Path,
    subset_names: list[str],
    images_per_video: Optional[int] = None,
) -> dict:
    data = {}
    real_imgs_cache = {}

    # Preload all necessary real images
    for subset in subset_names:
        if subset.endswith("-ff") and "ff" not in real_imgs_cache:
            ff_dataset_path = split_dir / "real" / "face-forensics"
            real_imgs_cache["ff"] = get_image_paths(ff_dataset_path, images_per_video)

        elif subset.endswith("-cdf") and "cdf" not in real_imgs_cache:
            cdf_dataset_path = split_dir / "real" / "celeb-df"
            real_imgs_cache["cdf"] = get_image_paths(cdf_dataset_path, images_per_video)

    for subset in subset_names:
        LOGGER.info(f'- Processing subset "{subset}"...')
        subset_path = split_dir / "fake" / subset
        data[subset] = {}

        if not subset_path.exists():
            LOGGER.warning(f'Directory for "{subset}" does not exist! {subset_path}')
            continue

        if subset.endswith("-ff"):
            fake_image_paths = get_image_paths(subset_path, images_per_video)
            data[subset]["fake"] = fake_image_paths
            data[subset]["real"] = real_imgs_cache["ff"]

        elif subset.endswith("-cdf"):
            fake_image_paths = get_image_paths(subset_path, images_per_video)
            data[subset]["fake"] = fake_image_paths
            data[subset]["real"] = real_imgs_cache["cdf"]

        else:
            fake_image_paths = get_image_paths(subset_path / "fake", images_per_video)
            real_image_paths = get_image_paths(subset_path / "real", images_per_video)
            data[subset]["fake"] = fake_image_paths
            data[subset]["real"] = real_image_paths

        LOGGER.info(f' - Loaded subset "{subset}"')

    return data


def load_dataset_manifest(path: Path) -> dict[str, dict]:
    with path.open("r") as f:
        return json.load(f)


def png_to_jpg(img, quality):
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
