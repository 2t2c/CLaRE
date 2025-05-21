import torch
from torch.utils.data import DataLoader

from .dataset import CTD


def build_data_loader(
    cfg,
    batch_size=64,
    is_train=True,
):
    mode = "train" if is_train else "test"
    dataset = CTD(cfg, mode=mode)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(dataset) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        collate_fn=dataset.collate_fn,
    )
    assert len(data_loader) > 0
    return data_loader


class DataManager:
    def __init__(self, cfg):
        train_loader = build_data_loader(
            cfg,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            is_train=True,
        )

        val_loader = None
        # if dataset_cfg.val:
        #     val_loader = build_data_loader(
        #         cfg,
        #         batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        #         is_train=False,
        #     )

        test_loader = build_data_loader(
            cfg,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            is_train=False,
        )

        # Attributes
        self._num_classes = 2

        # Dataset and data-loaders
        self.train_loader_x = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @property
    def num_classes(self):
        return self._num_classes
