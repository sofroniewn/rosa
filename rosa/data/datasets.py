import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from scipy.sparse import csr_matrix
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.config import ExpressionTransformConfig
from .sequences import AdataFastaInterval
from .transforms import ExpressionTransform, ToTensor

warnings.simplefilter(action="ignore", category=FutureWarning)


class RosaDataset(Dataset):
    """Return masked expression vectors iterating over cells

    When used with a batch of cell, returns data as ((BxG, BxG), BxGxGE), BxG)
    where B is number of cells in batch, G is number of genes,
    GE is length of gene embedding. The first (BxG, BxG) is the input expression,
    and masking matrix, the BxGxGE are the gene embeddings, and the BxG is
    the target expression to be predicted.
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        var_input: str,
        n_var_sample: Optional[int] = None,
        n_obs_sample: Optional[int] = None,
        obs_indices: Optional[Tensor] = None,
        var_indices: Optional[Tensor] = None,
        mask: Optional[Union[float, List, Tensor]] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
    ) -> None:

        self.adata = adata

        # prepare expression, shape n_obs x n_var
        if expression_layer is None:
            self.expression = adata.X  # type: Union[np.ndarray, csr_matrix]
        else:
            self.expression = adata.layers[expression_layer]

        if expression_transform_config is None:
            expression_transform_config = ExpressionTransformConfig()
        self.transform = ExpressionTransform(expression_transform_config)

        # prepare var input, shape n_var x var_dim
        if var_input in adata.varm.keys():
            self.var_input = ToTensor()(adata.varm[var_input])
        elif var_input[-3:] == ".fa":
            self.var_input = AdataFastaInterval(adata, var_input)  # type: ignore
        else:
            raise ValueError(f"Unrecognized var input {var_input}")

        if not self.expression.shape[1] == self.var_input.shape[0]:
            raise ValueError(
                f"Number of genes in expression and var input must match, got {self.expression.shape[1]} and {self.var_input.shape[0]}"
            )
        self.var_dim = self.var_input.shape[1]

        # Var indices are the var that will be included in each sample
        if var_indices is None:
            self.var_indices = torch.arange(self.expression.shape[1]).long()
        else:
            self.var_indices = Tensor(var_indices).long()
        if self.var_indices.max() >= self.expression.shape[1]:
            raise ValueError(
                f"Max index {self.var_indices.max()} too long for {self.expression.shape[1]} samples"
            )
        if self.var_indices.min() < 0:
            raise ValueError(f"Min index {self.var_indices.min()} less than zero")

        # Obs indices are the obs that will be sampled
        if obs_indices is None:
            self.obs_indices = torch.arange(self.expression.shape[0]).long()
        else:
            self.obs_indices = Tensor(obs_indices).long()
        if self.obs_indices.max() >= self.expression.shape[0]:
            raise ValueError(
                f"Max index {self.obs_indices.max()} too long for {self.expression.shape[0]} samples"
            )
        if self.obs_indices.min() < 0:
            raise ValueError(f"Min index {self.obs_indices.min()} less than zero")

        self.n_var_sample = n_var_sample
        if n_obs_sample is not None and n_obs_sample >= len(self.obs_indices):
            n_obs_sample = None
        self.n_obs_sample = n_obs_sample
        self.n_var = len(self.var_indices)

        # Create mask
        if mask is None:
            self.mask = torch.zeros(
                self.var_input.shape[0], dtype=torch.bool
            )  # type: Union[float, Tensor]
        elif isinstance(mask, float):
            self.mask = mask
        elif isinstance(mask, List) or isinstance(mask, Tensor):
            self.mask = torch.zeros(self.var_input.shape[0], dtype=torch.bool)
            self.mask[torch.tensor(mask).long()] = True
        else:
            raise ValueError("Unrecognized masking type")

    def __len__(self) -> int:
        if self.n_obs_sample is not None:
            return self.n_obs_sample
        return len(self.obs_indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # type: ignore
        if self.n_obs_sample is None:
            actual_idx_obs = self.obs_indices[idx]
        else:
            actual_idx_obs = self.obs_indices[torch.randint(len(self.obs_indices), (1,))]

        if self.n_var_sample is not None:
            actual_idx_var = self.var_indices[
                torch.randperm(self.n_var)[: self.n_var_sample]
            ]
        else:
            actual_idx_var = self.var_indices

        expression = self.transform(self.expression[actual_idx_obs])[actual_idx_var]

        if not isinstance(self.mask, Tensor):
            # mask = torch.rand(expression.shape) <= self.mask

            values, counts = torch.unique(expression, return_counts=True)
            nbins = self.transform[-1].n_bins
            bin_counts = torch.zeros(nbins, dtype=torch.long)
            bin_counts[values] = counts
            mask_indices = torch.multinomial(1 / bin_counts[expression], int(self.mask * len(expression)))
            mask = torch.zeros(expression.shape, dtype=torch.bool)
            mask[mask_indices] = True
        else:
            mask = self.mask[actual_idx_var]

        item = dict()
        item["expression"] = expression
        item["mask"] = mask
        item["indices"] = actual_idx_var
        item["obs_idx"] = actual_idx_obs
        return item