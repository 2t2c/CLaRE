"""
Main script to train/test the architecture pipeline.
"""

import os
import sys
import argparse
import time
from utils import set_seed, display_args
import logging

os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join("..")))
from train import train
from test import test

# logging
# wandb.login()

# create logger
logger = logging.getLogger("fomo_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def main(args):
    """
    Main function to train/test the architecture pipeline.
    """
    mode = args.mode
    if mode == "train":
        args.out_dir = os.path.join(args.out_dir, args.run_name, int(time.time()))
        # start training
        train(args)
    elif mode == "test":
        test(args)
    elif mode == "both":
        train(args)
        test(args)
    else:
        logger.error("Invalid mode. Choose 'train', 'test', 'both'.")


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    # wandb arguments
    parser.add_argument('--project', type=str,
                        default="local", help='wandb project name')
    parser.add_argument('--run_name', type=str,
                        default="debug", help='wandb run name')
    parser.add_argument('--log_every', type=int,
                        default=500, help='logging step')

    # training/testing config args
    parser.add_argument('--mode', type=str,
                        choices=['train', 'test', 'both'],
                        default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run the model on: cpu | cuda | mps")
    parser.add_argument('--eval_every', type=int,
                        default=1000, help='Evaluation step')
    # dataset args
    parser.add_argument("--data_root", type=str, default='data/',
                        help="The root folder of training set.")
    parser.add_argument("--train_file", type=str,
                        default='annotation/Train_num398700.txt')
    parser.add_argument("--val_file", type=str,
                        default='annotation/Test_MidjourneyV5_num2000.txt')
    parser.add_argument("--test_file", type=str,
                        default='annotation/Test_MidjourneyV5_num2000.txt')
    parser.add_argument("--no_strong_aug", action='store_true', default=False)

    # model configs
    parser.add_argument("--model", type=str, default='CLIP:RN50')
    parser.add_argument("--clip_type", type=str, default='wmap')
    parser.add_argument("--roi_pooling", type=str, default=False)

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='The training batch size over all gpus.')
    parser.add_argument("--out_dir", type=str, default='logdir')
    parser.add_argument("--num_classes", type=int, default=2, help='The class number of training dataset')
    parser.add_argument('--val_ratio', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument("--weights", type=str, default='out_dir', help="The folder to save models.")
    parser.add_argument('--data_size', type=int, default=256, help='The image size for training.')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--label_smooth", action='store_true', default=False)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    # pretty print args
    display_args(args)

    if args.seed is not None:
        # set seed for deterministic results
        set_seed(args.seed)
        logger.warning('You have chosen to seed training. '
                       'This will turn on the CUDNN deterministic setting, '
                       'which can slow down your training considerably! '
                       'You may see unexpected behavior when restarting '
                       'from checkpoints.')

    # call the main function
    main(args)
