"""
Script to train the CLIPping prompt-tuning architecture pipeline.
"""

from torch.cuda.amp import GradScaler
from torch.nn import functional as F

from models import get_model
from models.optimizer import build_optimizer
from models.scheduler import build_lr_scheduler
from utils import LOGGER

from .base import Trainer
from .evaluation import StatsMeter
from .utils import train_one_epoch


def train_clipping(cfg):
    trainer = ClippingTrainer(cfg)

    if cfg.mode == "train":
        trainer.train()
    elif cfg.mode == "test":
        trainer.test()


class ClippingTrainer(Trainer):
    def initialization(self):
        LOGGER.info("Loading model...")
        cfg = self.cfg
        self.model = get_model(name=cfg.model_name, cfg=cfg)
        self.scaler = GradScaler() if cfg.clipping.coop.prec == "amp" else None
        self.optimizer = build_optimizer(self.model.prompt_learner, cfg.clipping.optim)
        self.scheduler = build_lr_scheduler(self.optimizer, cfg.clipping.optim)
        self.criterion = F.cross_entropy

        # freeze all except prompt_learner
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def train_loop(self, *args, **kwargs):
        cfg = self.cfg
        loss_meter = StatsMeter()
        acc_meter = StatsMeter()

        step = 0
        best_val_acc = 0.0
        use_amp = cfg.clipping.coop.prec == "amp"
        for epoch in range(cfg.epochs):
            step, best_val_acc = train_one_epoch(
                self.model,
                step,
                best_val_acc,
                self.data.train_loader,
                self.data.val_loader,
                self.optimizer,
                epoch,
                loss_meter,
                acc_meter,
                cfg,
                self.device,
                self.scaler,
                use_amp,
                self.criterion,
                cfg.clip_grad,
                cfg.max_grad_norm,
            )
            self.scheduler.step()
