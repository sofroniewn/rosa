import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


import torch
from torch.utils.data import Dataset
import anndata as ad
import numpy as np


class AnnDataDataset(Dataset):
    """Return gene embedding, cell embedding, and experession for a single cell and gene combination"""

    def __init__(self, adata_path, item="joint"):
        # Load data
        adata = ad.read_h5ad(adata_path)

        # WARNING!!!!!
        adata.obsm[
            "embedding"
        ] = adata.X  # Overwrite pca embedding with expression data !!!!!!!!
        # adata.varm['embedding'] = adata.varm['PCs'] # Overwrite enformer embedding with PCs !!!!!!!!

        self.gene_embedding = torch.from_numpy(adata.varm["embedding"]).type(
            torch.float32
        )  # Shape gene x gene embedding
        # # WARNING!!!!!
        # # zero mean expression data across genes
        # X = adata.X
        # X = X - np.expand_dims(X.mean(axis=0),axis=0)
        self.expression = torch.from_numpy(adata.X).type(
            torch.float32
        )  # Shape cell x gene
        self.cell_embedding = torch.from_numpy(adata.obsm["embedding"]).type(
            torch.float32
        )  # Shape cell x cell embedding
        self.n_genes = self.expression.shape[1]
        self.n_cells = self.expression.shape[0]
        self.len_cell_embedding = self.cell_embedding.shape[1]
        self.len_gene_embedding = self.gene_embedding.shape[1]

        # specify return item
        self.item = item

    def __len__(self):
        if self.item in ("joint", "joint-concat"):
            return self.n_cells * self.n_genes
        elif self.item == "cell":
            return self.n_cells
        elif self.item == "gene":
            return self.n_genes
        else:
            raise ValueError(f"Item {self.item} not recognized")

    def __getitem__(self, idx):
        if self.item in ("joint", "joint-concat"):
            cell_i, gene_j = np.unravel_index(idx, (self.n_cells, self.n_genes))
            cell_i, gene_j = torch.tensor((cell_i, gene_j), dtype=torch.long)
            return (
                cell_i,
                gene_j,
                self.cell_embedding[cell_i],
                self.gene_embedding[gene_j],
            ), self.expression[cell_i, gene_j]
        elif self.item == "cell":
            cell_i = idx
            return self.cell_embedding[cell_i], self.expression[cell_i, :]
        elif self.item == "gene":
            gene_j = idx
            return self.gene_embedding[gene_j], self.expression[:, gene_j]
        else:
            raise ValueError(f"Item {self.item} not recognized")

    def keep_genes_for_cell_embedding(self, indices):
        self.cell_embedding = self.cell_embedding[:, indices]
        self.len_cell_embedding = self.cell_embedding.shape[1]
