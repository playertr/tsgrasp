from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig
import os

from tsgrasp.data.acronymvid import AcronymVidDataset, minkowski_collate_fn

class LitAcronymvidDataset(pl.LightningDataModule):
    def __init__(self, data_cfg : DictConfig, batch_size):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = data_cfg.num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = AcronymVidDataset(self.data_cfg, split="train")
            self.dataset_val   = AcronymVidDataset(self.data_cfg, split="test")

            if "subset_factor" in self.data_cfg:
                self.dataset_train = Subset(
                    self.dataset_train, 
                    indices=range(0, len(self.dataset_train), self.data_cfg.subset_factor))

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=minkowski_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=minkowski_collate_fn)

    def test_dataloader(self):
        # TODO: DON'T USE VAL DATASET FOR TESTING!
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=minkowski_collate_fn)

    def prepare_data(self):
        files = os.listdir(self.data_cfg.dataroot)
        if not all( split in files for split in ['test', 'train']):
            raise FileNotFoundError(f"Dataroot <{self.data_cfg.dataroot}> not populated with data files. Download or generate dataset.")