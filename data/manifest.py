from concurrent.futures import ThreadPoolExecutor
import json
from functools import partial
from pathlib import Path
from typing import Optional


from utils.logger import LOGGER


def construct_subset_entry(path: Path, label: str) -> dict:
    return {
        "root": str(path.resolve()),
        "label": label,
        "images": get_image_paths(path),
    }


def create_manifest(train: list[str], val: list[str], test: list[str]): ...


def get_image_paths(path: Path) -> list[str]:
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


def load_dataset_manifest(path: Path) -> dict[str, dict]:
    with path.open("r") as f:
        return json.load(f)
