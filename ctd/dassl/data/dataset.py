import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .utils import png2jpg

LOGGER = logging.getLogger("fomo_logger")


def get_image_paths(path: Path, images_per_video: Optional[int]) -> list[str]:
    """Recursively collects paths to all image files in a directory and its
    subdirectories.

    Parameters:
        path: The root directory to search for images.

    Returns:
        List of image file paths.
    """
    extensions = {".jpg", ".jpeg", ".png"}

    def is_image(p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in extensions

    return [str(p.relative_to(path)) for p in path.rglob("*") if is_image(p)]


def create_dataset_manifest(
    dataset_root: Path,
    manifest_path: Path,
    images_per_video: Optional[int] = None,
    subsets_to_include: Optional[list[str]] = None,
):
    data = {"root": str(dataset_root.absolute())}
    for split in ["train", "test", "val"]:
        split_path = dataset_root / split
        data[split] = {}

        if not split_path.exists():
            msg = (
                f'Could not find a directory for split "{split}" at {str(split_path)}!'
            )
            LOGGER.warning(msg)
            continue

        for classname in ["fake", "real"]:
            data[split][classname] = {}
            path = split_path / classname

            if not path.exists():
                continue

            def should_include_subset(subset: Path) -> bool:
                if not subset.is_dir():
                    return False
                if subsets_to_include is None:
                    return True
                return subset.name in subsets_to_include

            subsets = [p for p in path.iterdir() if should_include_subset(p)]
            for subset in subsets:
                LOGGER.info(f'Loading subset "{subset}"')
                image_paths = get_image_paths(subset, images_per_video)
                data[split][classname][subset.name] = image_paths

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing to file...")
    with manifest_path.open("w") as f:
        json.dump(data, f, indent=2)

    LOGGER.info(f'Create dataset manifest at "{str(manifest_path)}"')


def load_dataset_manifest(path: Path) -> dict[str, dict]:
    with path.open("r") as f:
        return json.load(f)


class DF40(Dataset):
    def __init__(
        self,
        config: dict,
        jpeg_quality: Optional[int] = None,
        debug: bool = False,
        mode: str = "train",
    ):
        """Initializes the dataset object.

        Args:
            config: A dictionary containing configuration parameters.
            mode: A string indicating the mode (train or test).
        """
        self.mode = mode
        self.debug = debug
        self.frame_num = config["frame_num"][mode]
        self.clip_size = config["clip_size"]
        self.jpeg_quality = jpeg_quality
        self.classname_to_label = config["classname_to_label"]
        self.transform = None

        manifest_path = config["dataset_manifest"]
        dataset_manifest = load_dataset_manifest(manifest_path)
        self.image_paths, self.labels = self.load_dataset(dataset_manifest, mode)

        msg = f"Could not find any images or labels for {self.mode} mode!"
        assert self.image_paths and self.labels, msg

        if self.debug:
            self.image_paths = self.image_paths[: min(1000, len(self.image_paths))]
            self.labels = self.labels[: min(1000, len(self.image_paths))]

    def load_dataset(self, manifest: dict, mode: str) -> tuple[list[str], list[int]]:
        """Collects image and label lists.

        Args:
            manifest_path: Path to the manifest JSON file.
            mode: Dataset split to load ('train', 'test', 'val').

        Returns:
            Tuple of (list of image paths, list of labels).
        """
        image_paths = []
        labels = []

        dataset_root = Path(manifest["root"])
        for classname, subsets in manifest[mode].items():
            label = self.classname_to_label[classname]

            for subset, rel_image_paths in subsets.items():
                # Build full image paths
                path_to_prepend = Path(dataset_root) / classname / subset
                full_paths = [str(path_to_prepend / img) for img in rel_image_paths]
                image_paths.extend(full_paths)
                labels.extend([label] * len(full_paths))

        return image_paths, labels

    def load_image(self, image_path: str):
        """Load an RGB image from a file path and resize it to the configured resolution.

        Raises an error if the image cannot be loaded.

        Returns the image as a PIL Image.
        """
        size = self.config["resolution"]
        assert os.path.exists(image_path), f"{image_path} does not exist"

        img = cv2.imread(image_path)
        if img is None:
            # fallback: open with PIL and convert to OpenCV format
            img = Image.open(image_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            if img is None:
                raise ValueError(f"Loaded image is None: {image_path}")

        if self.jpeg_quality is not None and image_path.endswith(".png"):
            img = Image.fromarray(img)
            img = png2jpg(img, self.jpeg_quality)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(np.uint8(img))

    def to_tensor(self, img):
        """Convert an image to a PyTorch tensor."""
        return transforms.ToTensor()(img)

    def normalize(self, img):
        """Normalize an image tensor."""
        mean = self.config["mean"]
        std = self.config["std"]
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(img)

    def augment_data(self, img, augmentation_seed: Optional[int] = None):
        """Apply data augmentation to an image.

        Args:
            img: Input image.
            augmentation_seed: Seed for reproducible augmentation.

        Returns:
            Augmented image.
        """
        if self.transform is None:
            msg = "augment_data is set to True but there are no defined augmentations!"
            raise ValueError(msg)

        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        augmented_img = self.transform(image=img)["image"]

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img

    @staticmethod
    def collate_fn(batch: list):
        """Collate a batch of data points.

        Args:
            batch: A list of tuples containing the image tensor and the label tensor.

        Returns:
            A tuple containing the image tensor and the label tensor.
        """
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        return {
            "images": images,
            "labels": labels,
        }

    def __getitem__(self, index, normalize: bool = True):
        """Retrieve a data point by index.

        Returns:
            An image and its label.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = np.array(self.load_image(image_path))

        if self.mode == "train" and self.config["use_data_augmentation"]:
            image = self.augment_data(image)

        if normalize:
            image = self.normalize(self.to_tensor(image))

        return image, label

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)


class CTD(DF40):
    def __init__(
        self,
        config,
        mode,
        img_size=224,
        jpeg_quality=None,
        debug=False,
    ):
        super().__init__(
            config,
            jpeg_quality=jpeg_quality,
            debug=debug,
            mode=mode,
        )
        self.transform = A.Compose(
            [
                A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.GaussNoise(p=1.0),
                    ],
                    p=0.5,
                ),
                A.RandomRotate90(p=0.33),
            ],
            p=1.0,
        )
