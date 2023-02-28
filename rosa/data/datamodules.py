from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from anndata import read_h5ad  # type: ignore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..utils.config import DataModuleConfig
from .datasets import rosa_dataset_factory


class RosaDataModule(LightningDataModule):
    def __init__(self, adata_path: Path, config: DataModuleConfig):
        super().__init__()

        self.adata_path = adata_path
        self.data_config = config.data
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = read_h5ad(self.adata_path)

        obs_indices_train = torch.Tensor(np.where(adata.obs["train"])[0])
        var_indices_train = torch.Tensor(np.where(adata.var["train"])[0])
        obs_indices_val = torch.Tensor(np.where(np.logical_not(adata.obs["train"]))[0])
        var_indices_val = torch.Tensor(np.where(np.logical_not(adata.var["train"]))[0])

        # create predict dataset with all data
        if self.data_config.mask is not None:
            mask = "var_indices"
            var_indices_predict = var_indices_val
        else:
            mask = None
            var_indices_predict = None

        self.predict_dataset = rosa_dataset_factory(
            adata,
            data_config=self.data_config,
            var_indices=var_indices_predict,
            mask=mask,
        )

        self.len_input = self.predict_dataset.input_dim
        self.len_target = self.predict_dataset.expression_dim

        self.train_dataset = rosa_dataset_factory(
            adata,
            data_config=self.data_config,
            obs_indices=obs_indices_train,
            var_indices=var_indices_train,
            n_var_sample=self.data_config.n_var_sample,
            mask=self.data_config.mask,
        )

        self.val_dataset = rosa_dataset_factory(
            adata,
            data_config=self.data_config,
            obs_indices=obs_indices_val,
            var_indices=var_indices_val,
            mask=mask,
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.predict_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def teardown(self, stage=None):
        pass
