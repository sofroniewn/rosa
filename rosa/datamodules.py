from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
import anndata as ad

from .datasets import JointAnnDataDataset, GeneAnnDataDataset, CellAnnDataDataset


class JointAnnDataModule(LightningDataModule):
    def __init__(self, adata_path):
        super().__init__()
        self.adata_path = adata_path
        self.batch_size = 2**14

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = ad.read_h5ad(self.adata_path)
        train_cells = adata.obs["train"]
        train_genes = adata.var["train"]

        adata_train = adata[train_cells, train_genes]
        self.train_dataset = JointAnnDataDataset(adata_train)

        adata_val = adata[np.logical_not(train_cells), np.logical_not(train_genes)]
        self.val_dataset = JointAnnDataDataset(adata_val)
        self.test_dataset = self.val_dataset

        self.predict_dataset = JointAnnDataDataset(adata)
        self.n_input_1 = self.predict_dataset.len_cell_embedding
        self.n_input_2 = self.predict_dataset.len_gene_embedding

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


class GeneAnnDataModule(LightningDataModule):
    def __init__(self, adata_path):
        super().__init__()
        self.adata_path = adata_path
        self.batch_size = 2**8

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = ad.read_h5ad(self.adata_path)
        train_genes = adata.var["train"]

        adata_train = adata[:, train_genes]
        self.train_dataset = GeneAnnDataDataset(adata_train)

        adata_val = adata[:, np.logical_not(train_genes)]
        self.val_dataset = GeneAnnDataDataset(adata_val)
        self.test_dataset = self.val_dataset

        self.predict_dataset = GeneAnnDataDataset(adata)
        self.n_input = self.predict_dataset.len_gene_embedding
        self.n_output = self.predict_dataset.n_cells

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


class CellAnnDataModule(LightningDataModule):
    def __init__(self, adata_path):
        super().__init__()
        self.adata_path = adata_path
        self.batch_size = 2**6

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        adata = ad.read_h5ad(self.adata_path)
        train_cells = adata.obs["train"]

        adata_train = adata[train_cells, :]
        self.train_dataset = CellAnnDataDataset(adata_train)

        adata_val = adata[np.logical_not(train_cells), :]
        self.val_dataset = CellAnnDataDataset(adata_val)
        self.test_dataset = self.val_dataset

        self.predict_dataset = CellAnnDataDataset(adata)
        self.n_input = self.predict_dataset.len_cell_embedding
        self.n_output = self.predict_dataset.n_genes

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
