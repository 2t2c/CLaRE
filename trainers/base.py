from abc import ABC, abstractmethod
from typing import Callable

import wandb
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from yacs.config import CfgNode

from .evaluation import run_test
from utils import LOGGER, display_model_summary, get_device, load_checkpoint
from data import DataModule


class Trainer(ABC):
    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.data = DataModule(cfg)

        self.model: Module | None = None
        self.optimizer: Optimizer | None = None
        self.criterion: Callable | None = None
        self.scaler: GradScaler | None = None
        self.scheduler: _LRScheduler | None = None

        if cfg.wandb:
            wandb.init(
                project=cfg.project,
                entity="FoMo",
                name=cfg.run_name + "/" + cfg.uid,
                config={
                    "architecture": cfg.model,
                    "clip_type": cfg.clip_type,
                    "batch_size": cfg.batch_size,
                    "out_dir": cfg.out_dir,
                    "seed": cfg.seed,
                    "mode": cfg.mode,
                    "device": cfg.device,
                    "eval_every": cfg.eval_every,
                    "log_every": cfg.log_every,
                    "epochs": cfg.epochs,
                    "train_datasets": cfg["train"],
                    "val_datasets": cfg["val"],
                    "test_datasets": cfg["test"],
                },
                settings=wandb.Settings(_service_wait=300, init_timeout=120),
            )

        self.initialization()
        msg_model = "Model not initialized in 'initialization()'"
        msg_optim = "Optimizer not initialized in 'initialization()'"
        msg_crit = "Criterion not initialized in 'initialization()'"

        assert self.model is not None, msg_model
        assert self.optimizer is not None, msg_optim
        assert self.criterion is not None, msg_crit
        LOGGER.info(f'Model "{cfg.model}" loaded successfully!')
        LOGGER.debug(f'Loading model to device "{self.device}"...')
        self.model.to(self.device)

    @abstractmethod
    def initialization(self):
        """Subclasses must override this to initialize:
        - self.model
        - self.optimizer
        - self.criterion
        - self.scaler (optional)
        - self.scheduler (optional)
        """

    def train(self, *args, **kwargs):
        input_shape = (1, 3, 224, 224)
        display_model_summary(self.model, input_shape, device=self.device)

        LOGGER.info(f"Training Started! - Debugging: {self.cfg.debug}")

        self.before_train()
        self.train_loop(*args, **kwargs)
        self.after_train()

        if self.cfg.wandb:
            wandb.finish()

    def before_train(self):
        """Optional hook to run before training."""
        pass

    @abstractmethod
    def train_loop(self, *args, **kwargs):
        """Subclasses must override this and implement their own training loop here."""
        pass

    def after_train(self):
        """Optional hook to run after training."""
        pass

    def test(self, path_to_checkpoint: str, *args, **kwargs):
        load_checkpoint(path_to_checkpoint, self.model, self.optimizer, self.scaler)
        self.model.eval()

        LOGGER.info(f"Evaluation Started! - Debugging: {self.cfg.debug}")

        self.before_test()

        for test_loader in self.data.test_loaders:
            run_test(self.model, test_loader, self.cfg, self.device)

        self.after_test()

    def before_test(self):
        """Optional hook to run before testing."""
        pass

    def after_test(self):
        """Optional hook to run after testing."""
        pass
