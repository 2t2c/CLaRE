"""
Collection of datasets for deepfake detection.
"""

import sys
import argparse
import yaml
import os
import math
import yaml
import glob
import json
import pickle
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
# from landmark_extraction import extract_rois

try:
    # Relative import for package use
    from .utils import *
except ImportError:
    # Fallback for direct script execution
    from utils import *
import albumentations as A
import logging
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from collections import Counter

# fetch logger
logger = logging.getLogger("fomo_logger")

# global variables
SEED = 0
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

FFpp_pool = ['FaceForensics++', 'FaceShifter',
             'DeepFakeDetection', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']

# dataset directory
DATASET_DIR = "/scratch-shared/scur0555/datasets"
HOME_DATASET_DIR= "/home/scur0555/datasets"

class UFD(Dataset):
    def __init__(self, real_path,
                 fake_path,
                 data_mode,
                 max_sample,
                 arch,
                 jpeg_quality=None,
                 gaussian_sigma=None):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(
                real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(
                    real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list

        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def recursively_read(self, rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
        out = []
        for r, d, f in os.walk(rootdir):
            for file in f:
                if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                    out.append(os.path.join(r, file))
        return out

    def get_list(self, path, must_contain=''):
        if ".pickle" in path:
            with open(path, 'rb') as f:
                image_list = pickle.load(f)
            image_list = [item for item in image_list if must_contain in item]
        else:
            image_list = self.recursively_read(path, must_contain)
        return image_list

    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = self.get_list(real_path, must_contain='0_real')
            fake_list = self.get_list(fake_path, must_contain='1_fake')
        else:
            real_list = self.get_list(real_path)
            fake_list = self.get_list(fake_path)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label


class DF40(Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, jpeg_quality=None,
                 gaussian_sigma=None, debug=False, mode='train'):
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
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        # Image settings
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.subset = config.subset
        self.rois = None

        # Set the dataset dictionary based on the mode
        if self.mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(
                    one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif self.mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(
                one_data)
        else:
            raise NotImplementedError(
                'Only train and test modes are supported.')

        assert len(image_list) != 0 and len(
            label_list) != 0, f"Collect nothing for {self.mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        # for debugging
        if self.debug: 
            self.image_list = self.image_list[: min(5_000, len(self.image_list))]
            self.label_list = self.label_list[: min(5_000, len(self.image_list))]

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        self.transform = self.augmentations()

    def augmentations(self):
        resize_block = []
        if not self.config['with_landmark']:
            resize_block = [random.choice([
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_LINEAR,
                    interpolation_up=cv2.INTER_LINEAR),
            ])]

        transformations = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'],
                     p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(
                blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            *resize_block,
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['data_aug']['brightness_limit'],
                    contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_range=(self.config['data_aug']['quality_lower'],
                                              self.config['data_aug']['quality_upper']), p=0.5)
        ],
            keypoint_params=A.KeypointParams(
                format='xy') if self.config['with_landmark'] else None
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
        unique_datasets = []
        # Record video name for video-level metrics
        video_name_list = []

        # Try to get the dataset information from the JSON file
        try:
            with open(os.path.join(self.config['configs_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        # Get the information for the current dataset
        logger.info(f"Config: {dataset_name}")
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                               'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                                  'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # use subset if provided
                if self.subset:
                    if not any(sub in video_name for sub in self.subset):
                        # skip dataset
                        continue
                # Unique video name
                unique_video_name = video_info['label'] + '_' + video_name

                # Get the label and frame paths for the current video
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(
                        f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']
                if len(frame_paths) == 0:
                    # logger.warning(f"{unique_video_name} is None. Let's skip it.")
                    continue
                # sorted video path to the lists
                if self.video_level:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(
                            x.split('\\')[-1].split('.')[0]))
                    else:
                        frame_paths = sorted(frame_paths, key=lambda x: int(
                            x.split('/')[-1].split('.')[0]))

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        # Select clip_size continuous frames
                        start_frame = random.randint(
                            0, total_frames - self.frame_num)
                        # update total_frames
                        frame_paths = frame_paths[start_frame:start_frame +
                                                              self.frame_num]
                    else:
                        # Select self.frame_num frames evenly distributed throughout the video
                        step = total_frames // self.frame_num
                        frame_paths = [frame_paths[i] for i in range(
                            0, total_frames, step)][:self.frame_num]

                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError(
                            'clip_size must be specified when video_level is True.')
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames -
                                         self.clip_size) // (num_clips - 1)

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = random.randrange(
                                    i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1))
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(
                                    continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = random.randrange(
                                0, total_frames - self.clip_size + 1)
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(
                                continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend(
                            [unique_video_name] * len(selected_clips))

                    else:
                        print(
                            f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")

                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend(
                        [unique_video_name] * len(frame_paths))
                        
                unique_datasets.append(video_name)

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        unique_subsets = set([name.split("_")[-1] for name in unique_datasets])
        # logger.info(f'Unique Subsets in "{self.mode}" mode: "{dataset_name}" - "{unique_subsets}"')

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
        size = self.config['resolution']  # if self.mode == "train" else self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)

        if img is None:
            img = Image.open(file_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if img is None:
                raise ValueError(
                    'Loaded image is None: {}'.format(file_path))
        # image processing
        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
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
        size = self.config['resolution']
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
        mean = self.config['mean']
        std = self.config['std']
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
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

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
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

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
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace(
                'frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace(
                '.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                # replace the hard-coded paths from the config
                image_path = image_path.replace("deepfakes_detection_datasets/FaceForensics++",
                                                f"{DATASET_DIR}/face_forensics/FaceForensics++")
                image_path = image_path.replace("deepfakes_detection_datasets/Celeb-DF-v2",
                                                f"{DATASET_DIR}/celeb_df/Celeb-DF-v2")
                image_path = image_path.replace("deepfakes_detection_datasets/DF40_train",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                image_path = image_path.replace("deepfakes_detection_datasets/DF40",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                logger.warning(f"Error loading image at index {index, image_path}: {e}")
                return self.__getitem__(0)
            # Convert to numpy array for data augmentation
            image = np.array(image)

            # Load mask and landmark (if needed)
            if self.mode == 'train' and self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.mode == 'train' and self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            elif self.config['model_name'] == 'sbi' and self.config['with_landmark']:
                try:
                    landmarks = self.load_landmark(landmark_path)
                except:
                    landmarks = None
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(
                    image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(
                    image), deepcopy(landmarks), deepcopy(mask)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.mode == 'train' and self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks_trans)
                if self.mode == 'train' and self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in
                       landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
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
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
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
        assert len(self.image_list) == len(
            self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


class LARE(DF40):
    def __init__(self, config, mode,
                 img_size=224, jpeg_quality=None,
                 gaussian_sigma=None, debug=False):
        # initialize DF40 first (inherits data loading, image_list, label_list, etc.)
        super().__init__(config, jpeg_quality=jpeg_quality, 
                        gaussian_sigma=gaussian_sigma, debug=debug,
                        mode=mode)
        self.img_size = img_size
        self.train_list = []
        self.transform = A.Compose([
            # A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, p=1.0),
            # A.RandomCrop(height=self.img_size, width=self.img_size, p=1.0),
            A.RandomResizedCrop(size=(self.img_size, self.img_size),
                                      scale=(0.8, 1.0),
                                      ratio=(0.95, 1.05), p=1.0),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
        ], p=1.0)
        self.landmarks = config.landmarks


    def load_loss_maps(self, map_file):
        map_paths = []
        # opening the map file
        with open(map_file, 'r') as f:
            for line in f:
                image_path, _ = line.strip().split('\t')
                map_paths.append(image_path)

        return map_paths

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), f'Number of images ({len(self.image_list)}) and labels ({len(self.label_list)}) are not equal'
        # assert len(self.image_list) == len(self.map_paths), f'Number of images ({len(self.image_list)}) and loss maps ({len(self.map_paths)}) are not equal'
        return len(self.image_list)

    def __getitem__(self, index, no_norm=False):
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            # for the image-level IO, only one frame is used
            image_paths = [image_paths]

        image_tensors = []
        loss_map_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            # Load the image
            try:
                # replace the hard-coded paths from the config
                image_path = image_path.replace("deepfakes_detection_datasets/FaceForensics++",
                                                f"{DATASET_DIR}/face_forensics/FaceForensics++")
                image_path = image_path.replace("deepfakes_detection_datasets/Celeb-DF-v2",
                                                f"{DATASET_DIR}/celeb_df/Celeb-DF-v2")
                if "Celeb-DF-v2" not in image_path and "sd2.1" not in image_path:
                    image_path = image_path.replace("Celeb-real", "Fake_from_Celeb-real", 1)
                    image_path = image_path.replace("YouTube-real", "Fake_from_Youtube-real", 1)
                filename = os.path.basename(image_path)
                image_path = image_path.replace(filename, filename.replace("Fake_from_", "").replace("Youtube", "YouTube"))
                image_path = image_path.replace("deepfakes_detection_datasets/DF40_train",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                image_path = image_path.replace("deepfakes_detection_datasets/DF40",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                # handle EFSALL_ff
                if self.mode == "train":
                    image_path = image_path.replace("/ff", "")
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                # logger.info(image_path)
                logger.error(f"Error loading image at index {index, image_path}: {e}")
                return self.__getitem__(0)

            # Convert to numpy array for data augmentation
            image = np.array(image)
            
            # extract ROIs if needed
            if self.landmarks:
                self.rois = extract_rois(image, merge_landmarks=False)

            # Load the loss map
            try:
                # hardcoded for now (TODO: fix lare extraction script)
                image_path = image_path.replace("/frames", "")
                image_path = image_path.replace("/ff", "")
                parts = image_path.split("/")
                dataset = parts[-3] # VQGAN
                folder = parts[-2]  # '335'
                file = parts[-1].split(".")[0] + ".pt"
                if "face_forensics" in image_path:
                    loss_map_path = f"{HOME_DATASET_DIR}/df40/loss_maps/face_forensics/{folder}/{file}"
                elif "cdf" in image_path:
                    loss_map_path = image_path.replace("test/", "loss_maps/").replace("train/", "loss_maps/")
                    filename = os.path.basename(loss_map_path)
                    if any(substring in image_path for substring in [
                            "ddim", "DiT", "pixart", "rddm", "StyleGANXL", "sd2.1"
                        ]):
                        loss_map_path = loss_map_path.replace(filename, filename.split(".")[0] + ".pt")
                elif "Celeb-DF-v2" in image_path:
                    loss_map_path = f"{DATASET_DIR}/df40/loss_maps/celeb_df/{dataset}/{folder}/{file}"
                else:
                    if any(substring in image_path for substring in [
                            "MidJourney", "whichisreal", "stargan", 
                            "starganv2", "styleclip", "CollabDiff"
                        ]):
                        image_path = image_path.replace("/test", "")
                        parts = image_path.split("/")
                        dataset = parts[-2]
                        file = parts[-1].split(".")[0] + ".pt"
                        if "real" in image_path:
                            prefix = "real"
                            dataset = parts[-3]
                        if "fake" in image_path:
                            prefix = "fake"
                            dataset = parts[-3]
                        if "CollabDiff" in image_path and "fake" in image_path:
                            file = parts[-2]
                            loss_map_path = f"{HOME_DATASET_DIR}/df40/loss_maps/CollabDiff/{prefix}_{file}.pt"
                        else:
                            loss_map_path = f"{HOME_DATASET_DIR}/df40/loss_maps/{dataset}/{prefix}_{file}"
                    elif "heygen" in image_path:
                        dataset, folder, subfolder, file = parts[-4:]
                        loss_map_path = f"{HOME_DATASET_DIR}/df40/loss_maps/{dataset}/{folder}/{subfolder}/{file}.pt"
                    else:
                        loss_map_path = f"{HOME_DATASET_DIR}/df40/loss_maps/{dataset}/{folder}/{file}"
                loss_map = torch.load(loss_map_path)
            except Exception as e:
                # skip this loss map and return the first one
                # logger.info(image_path)
                logger.error(f"Error loading loss map at index {index, loss_map_path}: {e}")
                # return a zero tensor of expected shape as fallback
                # loss_map = torch.zeros((4, 32, 32)) 
                return self.__getitem__(0)

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, _, _ = self.data_aug(image, augmentation_seed)
            else:
                image_trans = deepcopy(image)

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))

            image_tensors.append(image_trans)
            loss_map_tensors.append(loss_map)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first loss map tensor
            loss_map_tensors = loss_map_tensors[0]

        return image_tensors, label, loss_map_tensors


    @staticmethod
    def collate_fn(batch):
        images, labels, loss_maps = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        loss_maps = torch.stack(loss_maps, dim=0)

        return {
            'image': images,
            'label': labels,
            'loss_map': loss_maps
        }


class CTD(DF40):
    def __init__(self, config, mode,
                 img_size=224, jpeg_quality=None,
                 gaussian_sigma=None, debug=False):
        # initialize DF40 first (inherits data loading, image_list, label_list, etc.)
        super().__init__(config, jpeg_quality=jpeg_quality,
                        gaussian_sigma=gaussian_sigma, debug=debug,
                        mode=mode)
        self.img_size = img_size
        self.train_list = []
        self.anchor = False
        self.transform = A.Compose([
            # A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, p=1.0),
            # A.RandomCrop(height=self.img_size, width=self.img_size, p=1.0),
            A.RandomResizedCrop(size=(self.img_size, self.img_size),
                                      scale=(0.8, 1.0),
                                      ratio=(0.95, 1.05), p=1.0),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.5),
            A.HorizontalFlip(p=0.5),
        ], p=1.0)

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)

    def __getitem__(self, index, no_norm=False):
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            # for the image-level IO, only one frame is used
            image_paths = [image_paths]

        image_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2 ** 32 - 1)

            # Load the image
            try:
                # replace the hard-coded paths from the config
                image_path = image_path.replace("deepfakes_detection_datasets/FaceForensics++",
                                                f"{DATASET_DIR}/face_forensics/FaceForensics++")
                image_path = image_path.replace("deepfakes_detection_datasets/Celeb-DF-v2",
                                                f"{DATASET_DIR}/celeb_df/Celeb-DF-v2")
                image_path = image_path.replace("deepfakes_detection_datasets/DF40_train",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                image_path = image_path.replace("deepfakes_detection_datasets/DF40",
                                                f"{DATASET_DIR}/df40/{self.mode}")
                # handle EFSALL_ff
                if self.mode == "train":
                    image_path = image_path.replace("/ff", "")
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                logger.warning(f"Error loading image at index {index, image_path}: {e}")
                return self.__getitem__(0)

            # Convert to numpy array for data augmentation
            image = np.array(image)

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, _, _ = self.data_aug(image, augmentation_seed)
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
        return {
            'image': images,
            'label': labels,
        }


def describe_dataloader(dataloader, title="DataLoader Summary"):
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

    table = Table(title=title)

    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total samples", str(total_samples))
    table.add_row(f"Total batches (batch_size={dataloader.batch_size})", str(total_batches))
    table.add_row(f"Num Workers", str(dataloader.num_workers))

    # class info
    class_info_found = False
    if hasattr(dataset, 'classes'):
        table.add_row("Classes", str(dataset.classes))
        class_info_found = True
    if hasattr(dataset, 'class_to_idx'):
        table.add_row("Class to index mapping", str(dataset.class_to_idx))
        class_info_found = True
    if hasattr(dataset, 'data_dict') and 'label' in dataset.data_dict:
        targets = dataset.data_dict['label']
        if isinstance(targets, tuple):
            targets = list(targets)
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        label_counts = Counter(targets.tolist())
        table.add_row("Label counts", str(dict(label_counts)))
        class_info_found = True
    if not class_info_found:
        table.add_row("Class/Label info", "No class/label info found in dataset attributes.")

    # sample data shape and dtype
    try:
        first_batch = next(iter(dataloader))
        if isinstance(first_batch, (list, tuple)):
            # Show shape of first input and sample label summary
            shape_info = str(first_batch[0].shape)
            label_info = str(first_batch[1])
            table.add_row("Label sample", label_info)
            table.add_row("Input sample shape", shape_info)
            if len(first_batch) > 2:
                loss_map_shape = str(first_batch[2].shape)
                table.add_row("Loss map shape", loss_map_shape)
        elif isinstance(first_batch, dict):
            table.add_row("Sample keys", str(list(first_batch.keys())))
            for key, value in first_batch.items():
                if hasattr(value, 'shape'):
                    shape = tuple(value.shape)
                else:
                    shape = 'N/A'
                dtype = getattr(value, 'dtype', type(value).__name__)
                table.add_row(f"{key.capitalize()} shape & dtype", str(shape) + f", ({str(dtype)})")
        else:
            table.add_row("Sample", str(type(first_batch)))
    except Exception as e:
        table.add_row("Sample inspection error", str(e))

    console.print(table)


# test the dataset pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['ufd', 'df40', 'lare'],
                        default='lare')
    # add UFD-specific arguments
    parser.add_argument('--real_path', type=str,
                        default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str,
                        default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str,
                        default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000,
                        help='only check this number of images for both fake/real')
    parser.add_argument('--arch', default="clip", type=str)
    # add DF40-specific config path
    parser.add_argument("--df40_mode", type=str,
                        choices=['train', 'test'], default="train",
                        help="DF40 dataset mode name")
    parser.add_argument("--df40_name", type=str, default=None,
                        help="DF40 dataset name")
    parser.add_argument('--df40_config', type=str,
                        default="train_config.yaml",
                        help="DF40 mode config")
    # generic params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    args = parser.parse_args()
    display_args(args)

    if args.dataset == "ufd":
        dataset = UFD(
            args.real_path,
            args.fake_path,
            args.data_mode,
            args.max_sample,
            args.arch,
            jpeg_quality=args.jpeg_quality,
            gaussian_sigma=args.gaussian_sigma,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=4
        )
    elif args.dataset == "df40":
        # load the config file
        with open("./configs/" + args.df40_config, 'r') as f:
            config = yaml.safe_load(f)
        if args.df40_name is not None:
            config[f'{args.df40_mode}_dataset'] = args.df40_name
        dataset = DF40(config=config, mode=args.df40_mode)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True if args.df40_mode == "train" else False,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
    elif args.dataset == "lare":
        # load the config file
        with open("./configs/" + args.df40_config, 'r') as f:
            config = yaml.safe_load(f)
        if args.df40_name is not None:
            config[f'{args.df40_mode}_dataset'] = args.df40_name
        dataset = LARE(config=config, 
                       mode=args.df40_mode, 
                       jpeg_quality=args.jpeg_quality)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True if args.df40_mode == "train" else False,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
    rprint(f"Loaded dataset '{args.dataset}' successfully.")
    describe_dataloader(loader)
