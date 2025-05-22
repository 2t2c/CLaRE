"""
Main script to train/test the architecture pipeline.
"""

import os
import sys
import argparse
import time
from utils import set_seed, display_args
import logging
from rich.logging import RichHandler

os.environ["WANDB__SERVICE_WAIT"] = "300"
sys.path.append(os.path.abspath(os.path.join("..")))
from train import train
# from test import test

# logging
# wandb.login()

# create logger
logger = logging.getLogger("fomo_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False


def main(args):
    """
    Main function to train/test the architecture pipeline.
    """
    mode = args.mode
    if mode == "train":
        # create directory if not exists
        uid = str(int(time.time()))
        out_dir = os.path.join(os.path.expanduser("~"), args.out_dir, args.run_name, uid)
        os.makedirs(out_dir, exist_ok=True)
        args.uid = uid
        args.out_dir = out_dir
        # pretty print args
        display_args(args)
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
                        default=100, help='logging step')
    parser.add_argument('--logging', type=bool,
                        default=False, help='online logging')

    # training/testing config args
    parser.add_argument('--mode', type=str,
                        choices=['train', 'test', 'both'],
                        default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run the model on: cpu | cuda | mps")
    parser.add_argument('--eval_every', type=int,
                        default=100, help='Evaluation step')

    # dataset args
    parser.add_argument("--df40_name", type=str, default=None,
                        help="DF40 dataset name")
    parser.add_argument('--config', type=str,
                        default="config.yaml",
                        help="DF40 mode config")
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument('--debug', type=bool,
                        default=True, help='Debugging on few samples')

    # model configs
    # parser.add_argument("--model", type=str, default='CLIP:RN50')
    parser.add_argument("--model", type=str, default='CLIP:ViT-L/14')
    parser.add_argument("--clip_type", type=str, default='wmap')
    parser.add_argument("--roi_pooling", type=str, default=False)

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='The training batch size over all gpus.')
    parser.add_argument("--out_dir", type=str, default='fomo_logdir')
    parser.add_argument("--num_classes", type=int, default=2, help='The class number of training dataset')
    parser.add_argument('--val_ratio', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--data_size', type=int, default=256, help='The image size for training.')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--label_smooth", action='store_true', default=False)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()

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
