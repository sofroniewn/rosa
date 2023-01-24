from typing import Optional

import numpy as np
from anndata import read_h5ad  # type: ignore
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .datasets import AnnDataDataset, EmbeddingType


class AnnDataModule(LightningDataModule):
    def __init__(
        self,
        adata_path: str,
        batch_size: int,
        expression_layer: Optional[str] = None,
        var_embedding: Optional[str] = None,
        obs_embedding: Optional[str] = None,
    ):
        super().__init__()

        self.adata_path = adata_path
        self.batch_size = batch_size
        self._var_embedding = var_embedding
        self._obs_embedding = obs_embedding
        self._expression_layer = expression_layer

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = read_h5ad(self.adata_path)

        # create predict dataset with all data
        self.predict_dataset = AnnDataDataset(
            adata,
            expression_layer=self._expression_layer,
            var_embedding=self._var_embedding,
            obs_embedding=self._obs_embedding,
        )

        self.len_obs_embedding = self.predict_dataset.len_obs_embedding
        self.len_var_embedding = self.predict_dataset.len_var_embedding
        self.n_obs = self.predict_dataset.n_obs
        self.n_var = self.predict_dataset.n_var
        self.embedding_type = self.predict_dataset.embedding_type

        # split train and test sets based on adata file
        # TODO - if not provided, make splits random
        if self.embedding_type == EmbeddingType.JOINT:
            obs_train = adata.obs["train"]
            var_train = adata.var["train"]
            adata_train = adata[obs_train, var_train]
            adata_val = adata[np.logical_not(obs_train), np.logical_not(var_train)]
        elif self.embedding_type == EmbeddingType.OBS:
            obs_train = adata.obs["train"]
            adata_train = adata[obs_train]
            adata_val = adata[np.logical_not(obs_train)]
        elif self.embedding_type == EmbeddingType.VAR:
            var_train = adata.var["train"]
            adata_train = adata[:, var_train]
            adata_val = adata[:, np.logical_not(var_train)]
        else:
            raise ValueError(
                f"Type {self.embedding_type.name} not recognized, must be one of {list(EmbeddingType.__members__)}"
            )

        self.train_dataset = AnnDataDataset(
            adata_train,
            expression_layer=self._expression_layer,
            var_embedding=self._var_embedding,
            obs_embedding=self._obs_embedding,
        )

        self.val_dataset = AnnDataDataset(
            adata_val,
            expression_layer=self._expression_layer,
            var_embedding=self._var_embedding,
            obs_embedding=self._obs_embedding,
        )
        self.test_dataset = self.val_dataset

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
