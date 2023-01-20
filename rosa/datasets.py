import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from anndata import AnnData  # type: ignore
import numpy as np


class JointAnnDataDataset(Dataset):
    """Return gene embedding, cell embedding, and experession for a single cell and gene combination"""

    def __init__(self, adata: AnnData) -> None:
        self.adata = adata

        # Shape cell x gene
        self.expression = torch.from_numpy(adata.X).type(torch.float32)

        #  Shape gene x gene embedding
        self.gene_embedding = torch.from_numpy(adata.varm["embedding"]).type(
            torch.float32
        )

        # Shape cell x cell embedding
        self.cell_embedding = torch.from_numpy(adata.obsm["embedding"]).type(
            torch.float32
        )

        self.n_genes = self.expression.shape[1]
        self.n_cells = self.expression.shape[0]
        self.len_cell_embedding = self.cell_embedding.shape[1]
        self.len_gene_embedding = self.gene_embedding.shape[1]

    def __len__(self) -> int:
        return self.n_cells * self.n_genes

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        cell_i, gene_j = np.unravel_index(idx, (self.n_cells, self.n_genes))
        return (
            self.cell_embedding[cell_i],
            self.gene_embedding[gene_j],
            self.expression[cell_i, gene_j],
        )


class GeneAnnDataDataset(Dataset):
    """Return gene embedding and experession across all cells for a single gene"""

    def __init__(self, adata: AnnData) -> None:
        self.adata = adata

        # Shape cell x gene
        self.expression = torch.from_numpy(adata.X).type(torch.float32)

        #  Shape gene x gene embedding
        self.gene_embedding = torch.from_numpy(adata.varm["embedding"]).type(
            torch.float32
        )

        self.n_genes = self.expression.shape[1]
        self.n_cells = self.expression.shape[0]
        self.len_gene_embedding = self.gene_embedding.shape[1]

    def __len__(self) -> int:
        return self.n_genes

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.gene_embedding[idx], self.expression[:, idx]


class CellAnnDataDataset(Dataset):
    """Return cell embedding, and experession across all genes for a single cell"""

    def __init__(self, adata: AnnData) -> None:
        self.adata = adata

        # Shape cell x gene
        self.expression = torch.from_numpy(adata.X).type(torch.float32)

        # Shape cell x cell embedding
        self.cell_embedding = torch.from_numpy(adata.obsm["embedding"]).type(
            torch.float32
        )

        self.n_genes = self.expression.shape[1]
        self.n_cells = self.expression.shape[0]
        self.len_cell_embedding = self.cell_embedding.shape[1]

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.cell_embedding[idx], self.expression[idx, :]
