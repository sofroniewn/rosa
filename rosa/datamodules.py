from dataclasses import dataclass
from typing import Optional

import numpy as np
from anndata import read_h5ad  # type: ignore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .config import DataModuleConfig
from .datasets import (RosaJointDataset, RosaObsDataset, RosaVarDataset,
                       rosa_dataset_factory)


class RosaDataModule(LightningDataModule):
    def __init__(
        self, adata_path: str, config: DataModuleConfig
    ):
        super().__init__()

        self.adata_path = adata_path
        self.data_config = config.data
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = read_h5ad(self.adata_path)

        # create predict dataset with all data
        self.predict_dataset = rosa_dataset_factory(adata, data_config=self.data_config)

        self.len_input = self.predict_dataset.input_dim
        self.len_target = self.predict_dataset.expression_dim

        # split train and test sets based on adata file
        # TODO - if not provided, make splits random
        if isinstance(self.predict_dataset, RosaJointDataset):
            obs_train = adata.obs["train"]
            var_train = adata.var["train"]
            adata_train = adata[obs_train, var_train]
            adata_val = adata[np.logical_not(obs_train), np.logical_not(var_train)]
        elif isinstance(self.predict_dataset, RosaObsDataset):
            obs_train = adata.obs["train"]
            adata_train = adata[obs_train]
            adata_val = adata[np.logical_not(obs_train)]
        elif isinstance(self.predict_dataset, RosaVarDataset):
            var_train = adata.var["train"]
            adata_train = adata[:, var_train]
            adata_val = adata[:, np.logical_not(var_train)]
        else:
            raise ValueError(f"Type {type(self.predict_dataset)} not recognized.")

        self.train_dataset = rosa_dataset_factory(
            adata_train, data_config=self.data_config
        )

        self.val_dataset = rosa_dataset_factory(adata_val, data_config=self.data_config)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def teardown(self, stage=None):
        pass
