from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from anndata import read_h5ad  # type: ignore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..utils.config import DataModuleConfig
from .datasets import RosaDataset


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
        self.adata = read_h5ad(self.adata_path)

        obs_indices_train = torch.Tensor(np.where(self.adata.obs["train"])[0])
        var_indices_train = torch.Tensor(np.where(self.adata.var["train"])[0])
        obs_indices_val = torch.Tensor(np.where(np.logical_not(self.adata.obs["train"]))[0])
        var_indices_val = torch.Tensor(np.where(np.logical_not(self.adata.var["train"]))[0])
        split = len(obs_indices_val) / (len(obs_indices_val) + len(obs_indices_train))

        self.train_dataset = RosaDataset(
            self.adata,
            shuffle=False,
            mask_fraction=self.data_config.mask,
            pass_through=self.data_config.pass_through,
            corrupt=self.data_config.corrupt,
            var_input=self.data_config.var_input,
            obs_indices=obs_indices_train,
            var_indices=None,  # var_indices_train, None
            mask_indices=None,
            n_var_sample=self.data_config.n_var_sample,
            n_obs_sample=self.data_config.n_obs_sample,
            expression_layer=self.data_config.expression_layer,
            expression_transform_config=self.data_config.expression_transform,
        )

        if self.data_config.n_obs_sample is not None:
            n_obs_sample = int(self.data_config.n_obs_sample * split)
        else:
            n_obs_sample = None

        # self.val_dataset = RosaDataset(
        #     adata,
        #     mask_fraction=self.data_config.mask,
        #     pass_through=self.data_config.pass_through,
        #     corrupt=self.data_config.corrupt,
        #     var_input=self.data_config.var_input,
        #     # obs_indices=obs_indices_val,
        #     # var_indices=var_indices_train,
        #     # mask_indices=None,
        #     # obs_indices=obs_indices_train,
        #     # var_indices=None,
        #     # mask_indices=var_indices_val,
        #     obs_indices=obs_indices_val,
        #     var_indices=None,
        #     mask_indices=var_indices_val,
        #     n_var_sample=self.data_config.n_var_sample,
        #     n_obs_sample=n_obs_sample,
        #     expression_layer=self.data_config.expression_layer,
        #     expression_transform_config=self.data_config.expression_transform,
        # )

        self.val_dataset = RosaDataset(
            self.adata,
            shuffle=False,
            mask_fraction=1.0,
            pass_through=0.0,
            corrupt=0.0,
            var_input=self.data_config.var_input,
            obs_indices=obs_indices_val,
            var_indices=None,
            n_var_sample=None,
            n_obs_sample=None,
            mask_indices=var_indices_val,
            expression_layer=self.data_config.expression_layer,
            expression_transform_config=self.data_config.expression_transform,
        )

        # self.predict_dataset = RosaDataset(
        #     adata,
        #     mask_fraction=1.0,
        #     pass_through=0.0,
        #     corrupt=0.0,
        #     var_input=self.data_config.var_input,
        #     obs_indices=obs_indices_val,  # None
        #     var_indices=None,
        #     n_var_sample=None,
        #     n_obs_sample=None,
        #     mask_indices=var_indices_val,
        #     expression_layer=self.data_config.expression_layer,
        #     expression_transform_config=self.data_config.expression_transform,
        # )

        self.test_dataset = self.val_dataset
        self.predict_dataset = self.val_dataset
        self.var_dim = self.train_dataset.var_dim
        self.var_input = self.train_dataset.var_input

        # Set counts from train dataset
        self.counts = self.train_dataset.counts     

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.predict_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage=None):
        pass
