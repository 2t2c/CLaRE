import argparse
import os
import time

from trainers import GET_TRAINER
from utils import (
    LOGGER,
    construct_config,
    display_cfg,
    set_logging_level,
    set_seed,
)


def main(args):
    cfg = construct_config(args)
    mode = cfg.mode
    cfg.uid = str(int(time.time()))
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.module, cfg.uid)
    display_cfg(cfg)

    set_logging_level(cfg.log_level)

    trainer_cls = GET_TRAINER[cfg.module]
    trainer = trainer_cls(cfg)

    if mode == "train":
        os.makedirs(cfg.output_dir, exist_ok=True)
        trainer.train()
    elif mode == "test":
        trainer.test(cfg.test_checkpoint)
    else:
        LOGGER.error("Invalid mode! Allowed values: [train, test]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and evaluation script")

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to the data and model config YAML file",
    )

    # Debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugging mode with fewer samples",
    )

    log_levels = ["debug", "info", "warning", "error"]
    parser.add_argument(
        "--log-level",
        type=str,
        choices=log_levels,
        default="info",
        help=f"Select the log level. Choices: {log_levels}",
    )

    # Module selection
    modules = ["lare", "clipping"]
    parser.add_argument(
        "--module",
        type=str,
        choices=modules,
        default="clipping",
        help=f"Select module to run. Choices: {modules}",
    )

    # Weights & Biases (wandb) options
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="local",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="debug",
        help="Weights & Biases run name",
    )

    # Mode and seed
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Run mode: train or test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional). If not set, seed is None.",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Logging frequency (steps)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluation frequency (steps)",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="CLIP:ViT-L/14",
        help="Model architecture identifier",
    )
    parser.add_argument(
        "--clip-type",
        type=str,
        default=None,
        help="Type of CLIP variant",
    )
    parser.add_argument(
        "--roi-pooling",
        action="store_true",
        help="Enable ROI pooling if applicable",
    )

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size across all GPUs",
    )
    parser.add_argument(
        "--test-checkpoint",
        type=str,
        default=None,
        help="The path to the checkpoint to load for testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/",
        help="Directory for saving checkpoints and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--label-smoothing",
        action="store_true",
        help="Enable label smoothing",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of data loading workers",
    )
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        LOGGER.warning(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main(args)
