import os
import json
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

try:
    # Relative import for package use
    from .utils import IsotropicResize, png2jpg
except ImportError:
    # Fallback for direct script execution
    from utils import IsotropicResize, png2jpg
import albumentations as A
import logging

# fetch logger
logger = logging.getLogger("fomo_logger")

# global variables
SEED = 0
MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}

STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}

FFpp_pool = [
    "FaceForensics++",
    "FaceShifter",
    "DeepFakeDetection",
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
]

# dataset directory
DATASET_DIR = "/scratch-shared/scur0555/datasets"


class DF40(Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(
        self,
        config,
        jpeg_quality=None,
        debug=False,
        mode="train",
    ):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.debug = debug
        self.mode = mode
        self.compression = config["compression"]
        self.frame_num = config["frame_num"][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get("video_mode", False)
        self.clip_size = config.get("clip_size", None)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        # Image settings
        self.jpeg_quality = jpeg_quality

        # Set the dataset dictionary based on the mode
        if self.mode == "train":
            dataset_list = config["train_dataset"]
            print(f"Dataset list: {dataset_list}")

            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = (
                    self.collect_img_and_label_for_one_dataset(one_data)
                )
                print(f"TMP Image: {tmp_image}")
                print(f"TMP Label: {tmp_label}")
                print(f"TMP Name: {tmp_name}")

                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif self.mode == "test":
            one_data = config["test_dataset"]
            print(f"Dataset list: {dataset_list}")

            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = (
                self.collect_img_and_label_for_one_dataset(one_data)
            )
        else:
            raise NotImplementedError("Only train and test modes are supported.")

        assert len(image_list) != 0 and len(label_list) != 0, (
            f"Collect nothing for {self.mode} mode!"
        )
        self.image_list, self.label_list = image_list, label_list

        # for debugging
        if self.debug:
            self.image_list = self.image_list[: min(10_000, len(self.image_list))]
            self.label_list = self.label_list[: min(10_000, len(self.image_list))]

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            "image": self.image_list,
            "label": self.label_list,
        }

        self.transform = self.augmentations()

    def augmentations(self):
        resize_block = []
        if not self.config["with_landmark"]:
            resize_block = [
                random.choice(
                    [
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC,
                        ),
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                    ]
                )
            ]

        transformations = A.Compose(
            [
                A.HorizontalFlip(p=self.config["data_aug"]["flip_prob"]),
                A.Rotate(
                    limit=self.config["data_aug"]["rotate_limit"],
                    p=self.config["data_aug"]["rotate_prob"],
                ),
                A.GaussianBlur(
                    blur_limit=self.config["data_aug"]["blur_limit"],
                    p=self.config["data_aug"]["blur_prob"],
                ),
                *resize_block,
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=self.config["data_aug"][
                                "brightness_limit"
                            ],
                            contrast_limit=self.config["data_aug"]["contrast_limit"],
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_range=(
                        self.config["data_aug"]["quality_lower"],
                        self.config["data_aug"]["quality_upper"],
                    ),
                    p=0.5,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy")
            if self.config["with_landmark"]
            else None,
        )

        return transformations

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # Record video name for video-level metrics
        video_name_list = []

        # Try to get the dataset information from the JSON file
        try:
            with open(
                os.path.join(
                    self.config["dataset_json_folder"], dataset_name + ".json"
                ),
                "r",
            ) as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f"dataset {dataset_name} not exist!")

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == "FaceForensics++_c40":
            dataset_name = "FaceForensics++"
            cp = "c40"
        elif dataset_name == "FF-DF_c40":
            dataset_name = "FF-DF"
            cp = "c40"
        elif dataset_name == "FF-F2F_c40":
            dataset_name = "FF-F2F"
            cp = "c40"
        elif dataset_name == "FF-FS_c40":
            dataset_name = "FF-FS"
            cp = "c40"
        elif dataset_name == "FF-NT_c40":
            dataset_name = "FF-NT"
            cp = "c40"
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp is None and dataset_name in [
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
                "FaceForensics++",
                "DeepFakeDetection",
                "FaceShifter",
            ]:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == "c40" and dataset_name in [
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
                "FaceForensics++",
                "DeepFakeDetection",
                "FaceShifter",
            ]:
                sub_dataset_info = sub_dataset_info["c40"]

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # Unique video name
                unique_video_name = video_info["label"] + "_" + video_name

                # Get the label and frame paths for the current video
                if video_info["label"] not in self.config["label_dict"]:
                    raise ValueError(
                        f"Label {video_info['label']} is not found in the configuration file."
                    )
                label = self.config["label_dict"][video_info["label"]]
                frame_paths = video_info["frames"]
                if len(frame_paths) == 0:
                    print(f"{unique_video_name} is None. Let's skip it.")
                    continue
                # sorted video path to the lists
                if self.video_level:
                    if "\\" in frame_paths[0]:
                        frame_paths = sorted(
                            frame_paths,
                            key=lambda x: int(x.split("\\")[-1].split(".")[0]),
                        )
                    else:
                        frame_paths = sorted(
                            frame_paths,
                            key=lambda x: int(x.split("/")[-1].split(".")[0]),
                        )

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        # Select clip_size continuous frames
                        start_frame = random.randint(0, total_frames - self.frame_num)
                        # update total_frames
                        frame_paths = frame_paths[
                            start_frame : start_frame + self.frame_num
                        ]
                    else:
                        # Select self.frame_num frames evenly distributed throughout the video
                        step = total_frames // self.frame_num
                        frame_paths = [
                            frame_paths[i] for i in range(0, total_frames, step)
                        ][: self.frame_num]

                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError(
                            "clip_size must be specified when video_level is True."
                        )
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames - self.clip_size) // (
                                num_clips - 1
                            )

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = random.randrange(
                                    i * clip_step,
                                    min(
                                        (i + 1) * clip_step,
                                        total_frames - self.clip_size + 1,
                                    ),
                                )
                                continuous_frames = frame_paths[
                                    start_frame : start_frame + self.clip_size
                                ]
                                assert len(continuous_frames) == self.clip_size, (
                                    "clip_size is not equal to the length of frame_path_list"
                                )
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = random.randrange(
                                0, total_frames - self.clip_size + 1
                            )
                            continuous_frames = frame_paths[
                                start_frame : start_frame + self.clip_size
                            ]
                            assert len(continuous_frames) == self.clip_size, (
                                "clip_size is not equal to the length of frame_path_list"
                            )
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend(
                            [unique_video_name] * len(selected_clips)
                        )

                    else:
                        print(
                            f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames})."
                        )

                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(frame_paths))

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)

        return frame_path_list, label_list, video_name_list

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)

        if img is None:
            img = Image.open(file_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError("Loaded image is None: {}".format(file_path))
        # image processing
        if self.jpeg_quality is not None and file_path.endswith(".png"):
            img = Image.fromarray(img)
            img = png2jpg(img, self.jpeg_quality)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config["resolution"]
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
        else:
            return np.zeros((size, size, 1))
        # resize the mask to the specified resolution
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
        else:
            return np.zeros((81, 2))

        return np.float32(landmark)

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return transforms.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config["mean"]
        std = self.config["std"]
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        # Create a dictionary of arguments
        kwargs = {"image": img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs["keypoints"] = landmark
            kwargs["keypoint_params"] = A.KeypointParams(format="xy")
        if mask is not None:
            kwargs["mask"] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed["image"]
        augmented_landmark = transformed.get("keypoints")
        augmented_mask = transformed.get("mask")

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict["image"][index]
        label = self.data_dict["label"][index]

        if not isinstance(image_paths, list):
            # for the image-level IO, only one frame is used
            image_paths = [image_paths]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace("frames", "masks")  # Use .png for mask
            landmark_path = image_path.replace("frames", "landmarks").replace(
                ".png", ".npy"
            )  # Use .npy for landmark

            # Load the image
            try:
                # replace the hard-coded paths from the config
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/FaceForensics++",
                    f"{DATASET_DIR}/face_forensics/FaceForensics++",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/Celeb-DF-v2",
                    f"{DATASET_DIR}/celeb_df/Celeb-DF-v2",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/DF40_train",
                    f"{DATASET_DIR}/df40/{self.mode}",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/DF40",
                    f"{DATASET_DIR}/df40/{self.mode}",
                )
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index, image_path}: {e}")
                return self.__getitem__(0)
            # Convert to numpy array for data augmentation
            image = np.array(image)

            # Load mask and landmark (if needed)
            if self.mode == "train" and self.config["with_mask"]:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.mode == "train" and self.config["with_landmark"]:
                landmarks = self.load_landmark(landmark_path)
            elif self.config["model_name"] == "sbi" and self.config["with_landmark"]:
                try:
                    landmarks = self.load_landmark(landmark_path)
                except Exception as _:
                    landmarks = None
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == "train" and self.config["use_data_augmentation"]:
                image_trans, landmarks_trans, mask_trans = self.data_aug(
                    image, landmarks, mask, augmentation_seed
                )
            else:
                image_trans, landmarks_trans, mask_trans = (
                    deepcopy(image),
                    deepcopy(landmarks),
                    deepcopy(mask),
                )

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.mode == "train" and self.config["with_landmark"]:
                    landmarks_trans = torch.from_numpy(landmarks_trans)
                if self.mode == "train" and self.config["with_mask"]:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark)
                for landmark in landmark_tensors
            ):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(
                m is None or (isinstance(m, list) and None in m) for m in mask_tensors
            ):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark)
                for landmark in landmark_tensors
            ):
                landmark_tensors = landmark_tensors[0]
            if not any(
                m is None or (isinstance(m, list) and None in m) for m in mask_tensors
            ):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if not any(
            landmark is None or (isinstance(landmark, list) and None in landmark)
            for landmark in landmarks
        ):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict["image"] = images
        data_dict["label"] = labels
        data_dict["landmark"] = landmarks
        data_dict["mask"] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), (
            "Number of images and labels are not equal"
        )
        return len(self.image_list)


class CTD(DF40):
    def __init__(
        self,
        config,
        mode,
        img_size=224,
        jpeg_quality=None,
        gaussian_sigma=None,
        debug=False,
    ):
        super().__init__(
            config,
            jpeg_quality=jpeg_quality,
            gaussian_sigma=gaussian_sigma,
            debug=debug,
            mode=mode,
        )
        self.img_size = img_size
        self.train_list = []
        self.transform = A.Compose(
            [
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, p=1.0),
                A.RandomCrop(height=self.img_size, width=self.img_size, p=1.0),
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

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), (
            "Number of images and labels are not equal"
        )
        return len(self.image_list)

    def __getitem__(self, index, no_norm=False):
        image_paths = self.data_dict["image"][index]
        label = self.data_dict["label"][index]

        # for the image-level IO, only one frame is used
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_tensors = []
        for image_path in image_paths:
            try:
                # replace the hard-coded paths from the config
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/FaceForensics++",
                    f"{DATASET_DIR}/face_forensics/FaceForensics++",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/Celeb-DF-v2",
                    f"{DATASET_DIR}/celeb_df/Celeb-DF-v2",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/DF40_train",
                    f"{DATASET_DIR}/df40/{self.mode}",
                )
                image_path = image_path.replace(
                    "deepfakes_detection_datasets/DF40",
                    f"{DATASET_DIR}/df40/{self.mode}",
                )
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                logger.warning(f"Error loading image at index {index, image_path}: {e}")
                return self.__getitem__(0)

            # Convert to numpy array for data augmentation
            image = np.array(image)

            # Do Data Augmentation
            if self.mode == "train" and self.config["use_data_augmentation"]:
                image_trans, _, _ = self.data_aug(image)
            else:
                image_trans = deepcopy(image)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))

            image_tensors.append(image_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]

        return image_tensors, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        return {"image": images, "label": labels}
