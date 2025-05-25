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
sys.path.append(os.path.abspath(os.path.join("..")))
from train_lare import train as train_lare
from train_clipping import train as train_clipping
from test import test
from train_fusion import train as train_fusion

# logging
os.environ["WANDB__SERVICE_WAIT"] = "300"
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
    # create directory if not exists
    uid = str(int(time.time()))
    logger.info(f"Session Started with UID '{uid}' and Mode '{mode}'")
    # log_dir = os.path.join(os.path.expanduser("~"), args.log_dir, args.module, args.run_name, uid)
    log_dir = os.path.join(os.path.expanduser("~"), args.log_dir, uid)
    os.makedirs(log_dir, exist_ok=True)
    args.uid = uid
    args.log_dir = log_dir

    if mode == "train":
        # start training
        if args.module == "clipping":
            train_clipping(args)
        elif args.module == "lare":
            train_lare(args)
        elif args.module == "fusion":
            train_fusion(args)
        else:
            logger.error("Invalid module. Choose 'lare', 'clipping', or 'fusion'.")
    elif mode == "test":
        test(args)
    # elif mode == "both": # TODO: implement both
    #     train(args)
    #     test(args)
    else:
        logger.error("Invalid mode. Choose 'train', 'test', 'both'.")


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    # global config file for training/testing
    parser.add_argument('--config', type=str,
                        default="./configs/config.yaml",
                        help="Data and Model config")
    # type of module
    parser.add_argument('--module', type=str,
                        choices=['lare', 'clipping', 'fusion'],
                        default="clipping")

    # wandb arguments
    parser.add_argument('--project', type=str,
                        default="local", help='wandb project name')
    parser.add_argument('--run_name', type=str,
                        default="debug", help='wandb run name')
    parser.add_argument('--log_every', type=int,
                        default=50, help='logging step')
    parser.add_argument('--logging', action='store_true',
                        help='online logging')

    # training/testing config args
    parser.add_argument('--mode', type=str,
                        choices=['train', 'test', 'both'],
                        default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run the model on: cpu | cuda | mps")
    parser.add_argument('--eval_every', type=int,
                        default=50, help='Evaluation step')

    # testing config args
    parser.add_argument("--test_datasets", nargs='+', default=None,
                        help="DF40 dataset name")
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='Checkpoint to evaluate')

    # dataset args
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Does not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4. Used to test robustness of the model. Does not apply if None")
    parser.add_argument('--train_ratio', type=int, default=32, help='Total train frames per video')
    parser.add_argument('--test_ratio', type=int, default=8, help='Total test frames per video')
    parser.add_argument('--debug', action='store_true', help='Debugging on few samples')

    # model configs
    # parser.add_argument("--model", type=str, default='CLIP:RN50')
    parser.add_argument("--model", type=str, default='CLIP:ViT-L/14')
    parser.add_argument("--clip_type", type=str, default='clipping')
    parser.add_argument("--roi_pooling", type=str, default=False)

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='The training batch size over all gpus.')
    parser.add_argument("--log_dir", type=str, default='fomo_logdir')
    parser.add_argument("--num_classes", type=int, default=2, help='The class number of training dataset')
    parser.add_argument('--val_ratio', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--data_size', type=int, default=256, help='The image size for training.')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--label_smooth", action='store_true', default=False)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--num_workers', default=4, type=int, metavar='N',
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
