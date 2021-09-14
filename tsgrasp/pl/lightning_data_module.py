from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tsgrasp.data.acronymvid import AcronymVidDataset


class LitDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 2

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = AcronymVidDataset(self.data_dir, split="train")
            # self.dataset_train = AcronymVidDataset(self.data_dir, split="val")
            # dataset_full = AcronymVidDataset(self.data_dir, train=True)
            # self.dataset_train, self.dataset_val = random_split(
            #     dataset_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.dataset_test = AcronymVidDataset(
                self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)