import itertools
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from utils import LOGGER, png_to_jpg

from .manifest import load_dataset_manifest
from .transforms import get_clip_transform


class DF40(Dataset):
    def __init__(
        self,
        config: dict,
        jpeg_quality: Optional[int] = None,
        debug: bool = False,
        mode: str = "train",
        test_subset=None,
    ):
        """Initializes the dataset object.

        Args:
            config: A dictionary containing configuration parameters.
            mode: A string indicating the mode (train or test).
        """
        self.mode = mode
        self.debug = debug
        self.config = config
        self.frames_per_video = config["frames_per_video"][mode]
        self.clip_size = config["clip_size"]
        self.jpeg_quality = jpeg_quality
        self.classname_to_label = config["class_to_label"]
        self.image_resolution = config["resolution"]
        self.transform = None
        self.test_subset = test_subset
        manifest_path = Path(config["manifest"])
        dataset_manifest = load_dataset_manifest(manifest_path)
        self.image_paths, self.labels = self.load_dataset(dataset_manifest, mode)

        msg = f"Could not find any images or labels for {self.mode} mode!"
        assert self.image_paths and self.labels, msg

        if self.debug:
            self.image_paths = self.image_paths[: min(10_000, len(self.image_paths))]
            self.labels = self.labels[: min(10_000, len(self.image_paths))]

    def load_dataset(self, manifest: dict, mode: str) -> tuple[list[str], list[int]]:
        if mode == "train":
            image_paths, labels = self.load_train_dataset(manifest)
        elif mode == "test":
            image_paths, labels = self.load_test_dataset(manifest, self.test_subset)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)
        return list(image_paths), list(labels)

    def load_train_dataset(self, manifest: dict) -> tuple[tuple[str], tuple[int]]:
        """Collects image and label lists.

        Args:
            manifest_path: Path to the manifest JSON file.

        Returns:
            Tuple of (list of image paths, list of labels).
        """
        LOGGER.info("Loading dataset...")

        mode = "train"
        dataset_root = Path(manifest["root"])
        real_img_paths = []
        fake_img_paths = []

        for classname, subsets in manifest[mode].items():
            image_paths = fake_img_paths if classname == "fake" else real_img_paths

            for subset, subset_img_paths in subsets.items():
                if self.frames_per_video is not None:
                    subset_img_paths = self.sample_sorted_paths(
                        subset_img_paths, self.frames_per_video
                    )

                # Build full image paths
                path_to_prepend = Path(dataset_root) / mode / classname / subset
                full_paths = [str(path_to_prepend / img) for img in subset_img_paths]
                image_paths.extend(full_paths)

        image_paths, labels = self.balance_dataset(real_img_paths, fake_img_paths)
        return image_paths, labels

    def load_test_dataset(
        self, manifest: dict, subset: str
    ) -> tuple[list[str], list[int]]:
        """Collects image and label lists.

        Args:
            manifest: Dataset manifest dictionary.
            subset: Specific subset name within the mode to load.

        Returns:
            Tuple of (list of image paths, list of labels).
        """
        LOGGER.info("Loading dataset...")

        mode = "test"
        fake_label = self.classname_to_label["fake"]
        real_label = self.classname_to_label["real"]
        dataset_root = Path(manifest["root"])
        subset_prefix = dataset_root / mode / "fake" / subset

        fake_imgs = manifest[mode][subset]["fake"]
        subset_is_split_in_videos = subset.endswith(("-cdf", "-ff"))

        if self.frames_per_video is not None and subset_is_split_in_videos:
            fake_imgs = self.sample_sorted_paths(fake_imgs, self.frames_per_video)

        fake_paths = [str(subset_prefix / img) for img in fake_imgs]
        fake_labels = [fake_label] * len(fake_paths)

        if subset.endswith("-cdf"):
            real_prefix = dataset_root / mode / "real" / "celeb-df"
        elif subset.endswith("-ff"):
            real_prefix = dataset_root / mode / "real" / "face-forensics"
        else:
            real_prefix = subset_prefix / "real"

        real_imgs = manifest[mode][subset]["real"]

        if self.frames_per_video is not None and subset_is_split_in_videos:
            real_imgs = self.sample_sorted_paths(real_imgs, self.frames_per_video)

        real_paths = [str(real_prefix / img) for img in real_imgs]
        real_labels = [real_label] * len(real_paths)

        image_paths = fake_paths + real_paths
        labels = fake_labels + real_labels

        return image_paths, labels

    def balance_dataset(
        self, real_image_paths: list[str], fake_image_paths: list[str]
    ) -> tuple[list[str], list[int]]:
        repeat_factor = len(fake_image_paths) // len(real_image_paths)

        balanced_real_paths = real_image_paths * repeat_factor
        remaining = len(fake_image_paths) - len(balanced_real_paths)
        balanced_real_paths += random.choices(real_image_paths, k=remaining)

        image_paths = balanced_real_paths + fake_image_paths
        real_label = self.classname_to_label["real"]
        fake_label = self.classname_to_label["fake"]
        real_labels = [real_label] * len(balanced_real_paths)
        fake_labels = [fake_label] * len(fake_image_paths)
        labels = real_labels + fake_labels

        return image_paths, labels

    def load_image(self, image_path: str):
        """Load an RGB image from a file path and resize it to the configured resolution.

        Raises an error if the image cannot be loaded.

        Returns the image as a PIL Image.
        """
        size = self.image_resolution
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
            img = png_to_jpg(img, self.jpeg_quality)

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
    def sample_sorted_paths(
        image_paths: list[str], frames_per_video: int, seed: Optional[int] = None
    ) -> list[str]:
        if seed is not None:
            random.seed(seed)

        def group_key(path: str) -> str:
            path.rsplit("/", 1)[0]

        sampled = []

        # Group paths by video folder
        for folder, group in itertools.groupby(image_paths, key=group_key):
            group = list(group)

            if len(group) > frames_per_video:
                group = random.sample(group, frames_per_video)

            sampled.extend(group)
        return sampled

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
            "image": images,
            "label": labels,
        }

    def __getitem__(self, index, normalize: bool = True):
        """Retrieve a data point by index.

        Returns:
            An image and its label.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = np.array(self.load_image(image_path))

        if self.mode == "train":
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
        test_subset=None,
    ):
        super().__init__(
            config,
            jpeg_quality=jpeg_quality,
            debug=debug,
            mode=mode,
            test_subset=test_subset,
        )
        self.transform = get_clip_transform(img_size)


class LARE(DF40):
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
        self.transform = get_clip_transform(img_size)

    def __getitem__(self, index, normalize: bool = True):
        """Retrieve a data point by index.

        Returns:
            An image, its label, and its loss map.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = np.array(self.load_image(image_path))
        loss_map_path = image_path.replace("frames", "loss_maps").replace(".png", ".pt")

        if self.mode == "train" and self.config["use_data_augmentation"]:
            image = self.augment_data(image)

        if normalize:
            image = self.normalize(self.to_tensor(image))

        return image, label, loss_map_path

    @staticmethod
    def collate_fn(batch):
        images, labels, loss_maps = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        loss_maps = torch.stack(loss_maps, dim=0)

        return {"image": images, "label": labels, "loss_map": loss_maps}
