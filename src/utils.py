"""
This module contains utility functions for data processing and whole.
"""

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from io import BytesIO
import random
import logging
import cv2
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops.functional import crop
from datetime import timedelta
from rich import print as rprint
from rich.table import Table
from rich.console import Console
import torch
from torchinfo import summary
from yacs.config import CfgNode as CN
import yaml

# fetch logger
logger = logging.getLogger("fomo_logger")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(type):
    if type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif type == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device =  torch.device("cpu")
    logger.info(f"Using device: {device}")

    return device


def load_config(config):
    """
    Convert a dictionary to a CfgNode object.
    :param:
        config: yaml file
    :return:
        cfg: CfgNode object
    """
    if isinstance(config, dict):
        node = CN()
        for k, v in config.items():
            node[k] = load_config(v)
        return node

    return config

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale),
                         interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists,
              self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(
            mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"


def png2jpg(img, quality):
    # check if the img in right
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    out = BytesIO()
    # ranging from 0-95, 75 is default
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()

    return img


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def create_bbox_face(image, landmarks, margin=0):
    # Convert landmarks to a NumPy array if not already
    landmarks = np.array(landmarks)

    # Find the minimum and maximum x and y coordinates
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Find the maximum of the two to get the side length of the square
    max_side = max(width, height)

    # Adjust the bounding box to be a square
    min_x = min_x - ((max_side - width) / 2)
    max_x = min_x + max_side
    min_y = min_y - ((max_side - height) / 2)
    max_y = min_y + max_side

    # Add margin
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(image.shape[1], max_x + margin)
    max_y = min(image.shape[0], max_y + margin)

    # Convert coordinates to integers
    min_x, min_y, max_x, max_y = map(int, [min_x, min_y, max_x, max_y])

    # Crop original image within the bbox
    face = image[min_y:max_y, min_x:max_x]

    return face


def display_args(args, title="Arguments"):
    """
    Nicely print argparse.Namespace or dict using rich.

    :param:
        args: argparse.Namespace or dict
    :param:
        title: Optional table title
    """
    if not isinstance(args, dict):
        args = vars(args)

    # supress big output
    try:
        args.dataset.label_dict = "[SUPPRESSED]"
    except:
        pass

    table = Table(title=title)
    table.add_column("Argument", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in args.items():
        table.add_row(str(key), str(value))

    console = Console()
    console.print(table)

def display_metrics(metrics: dict, elasped, title="Validation Metrics"):
    """
    Nicely print metric dictionary using rich.

    :param:
        metrics: Dictionary of metrics (e.g., {"AUC": 0.95, "ACC": 0.88, ...})
    :param:
        title: Optional table title
    """
    rprint("Time Elasped:", str(timedelta(seconds=elasped)))
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(str(key), f"{value:.6f}" if isinstance(value, (int, float)) else str(value))

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
    table.add_row("Trainable (Million)", f"{trainable_params / (1024 ** 2):.2f} Million")
    if hasattr(model, "clip_model"):
        arch = str(summary(model.clip_model, depth=1, verbose=0))
    else:
        arch = str(summary(model, depth=1, verbose=0))
    table.add_row("Summary", arch)

    if input_shape is not None:
        if hasattr(model, "clip_model"):
            table.add_row(f"Input Shape", str(input_shape))
            visual_arch = str(summary(model.clip_model.visual, input_size=input_shape, depth=1, verbose=0))
            table.add_row("CLIP.Visual", visual_arch)

    console = Console()
    console.print(table)