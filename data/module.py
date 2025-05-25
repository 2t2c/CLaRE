from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

from utils import describe_dataloader

from .dataset import CTD, DF40, LARE

DATASET_CLASSES: dict[str, DF40] = {
    "CTD": CTD,
    "LARE": LARE,
}


class NamedDataLoader(DataLoader):
    def __init__(self, *args, name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name


class DataModule:
    def __init__(self, cfg: CfgNode):
        dataset_class = DATASET_CLASSES.get(cfg.dataset.name)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset class: {cfg.dataset.name}")

        self.dataset_class = dataset_class
        self.cfg = cfg

        self.train_dataset, self.train_loader = self._create_dataset_and_loader(
            mode="train",
        )
        self.val_dataset, self.val_loader = self._create_dataset_and_loader(
            mode="test", test_subset=cfg.val[0], shuffle=False
        )

        self.test_datasets = []
        self.test_loaders = []
        for dataset_name in cfg.test:
            dataset, dataloader = self._create_dataset_and_loader(
                mode="test", test_subset=dataset_name, shuffle=False
            )
            self.test_datasets.append(dataset)
            self.test_loaders.append(dataloader)

    def _create_dataset_and_loader(
        self,
        mode: str,
        test_subset: str | None = None,
        shuffle: bool = True,
    ) -> tuple[Dataset, NamedDataLoader]:
        dataset = self.dataset_class(
            mode=mode,
            config=self.cfg.dataset,
            debug=self.cfg.debug,
            test_subset=test_subset,
        )
        dataloader = NamedDataLoader(
            dataset,
            name=test_subset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        describe_dataloader(dataloader)
        return dataset, dataloader
