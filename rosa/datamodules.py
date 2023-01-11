from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from .datasets import AnnDataDataset
from .utils import random_split_multi


class AnnDataModule(LightningDataModule):
    def __init__(self, adata_path, item="joint", split="simple"):
        super().__init__()
        self.adata_path = adata_path
        self.item = item
        self.split = split

        if self.item in ("joint", "joint-concat"):
            self.batch_size = 2**14
        else:
            self.batch_size = 2**6

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = AnnDataDataset(self.adata_path, item=self.item)
        self.predict_dataset = dataset
        self.n_dim_1 = dataset.n_cells
        self.n_dim_2 = dataset.n_genes

        if self.split == "simple":
            self.train_dataset, self.val_dataset = random_split(
                dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
        elif self.split == "joint" and self.item in ("joint", "joint-concat"):
            shape = (self.n_dim_1, self.n_dim_2)
            self.train_dataset, self.val_dataset = random_split_multi(
                dataset, [0.8, 0.2], shape, generator=torch.Generator().manual_seed(42)
            )
            train_genes = np.unique(np.array(self.train_dataset.indices_multi)[:, 1])
            dataset.keep_genes_for_cell_embedding(train_genes)
        else:
            raise ValueError(f"Split {self.split} not recognized")
        # Use same dataset for validation and testing !!!!
        self.test_dataset = self.val_dataset

        self.n_input_1 = dataset.len_cell_embedding
        self.n_input_2 = dataset.len_gene_embedding

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def teardown(self, stage=None):
        pass
