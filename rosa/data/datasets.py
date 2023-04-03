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
    """Return masked expression vectors iterating over cells"""

    def __init__(
        self,
        adata: AnnData,
        *,
        var_input: str,
        pass_through: float = 0,
        corrupt: float = 0,
        mask_fraction: float = 0,
        n_var_sample: Optional[int] = None,
        n_obs_sample: Optional[int] = None,
        obs_indices: Optional[Tensor] = None,
        var_indices: Optional[Tensor] = None,
        mask_indices: Optional[Tensor] = None,
        mask_style: Optional[str] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
    ) -> None:

        self.adata = adata
        self.pass_through = pass_through
        self.corrupt = corrupt
        self.mask_fraction = mask_fraction

        # prepare expression, shape n_obs x n_var
        if expression_layer is None:
            expression = adata.X  # type: Union[np.ndarray, csr_matrix]
        else:
            expression = adata.layers[expression_layer]

        if expression_transform_config is None:
            expression_transform_config = ExpressionTransformConfig()

        self.transform = ExpressionTransform(expression_transform_config)
        self.expression = torch.stack([self.transform(x) for x in expression])
        self.n_bins = self.transform[-1].n_bins

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

        if mask_style is None or mask_style == 'random':
            self.counts = None
        elif mask_style == 'uniform':
            counts = torch.bincount(self.expression.ravel(), minlength=self.n_bins)
            self.counts = counts / counts.sum() # counts n_bins
        elif mask_style == 'uniform_var':
            counts = torch.stack([torch.bincount(x, minlength=self.n_bins) for x in self.expression.T]).T
            self.counts = counts / self.expression.shape[0] # counts var x n_bins
        elif mask_style == 'uniform_obs':
            counts = torch.stack([torch.bincount(x, minlength=self.n_bins) for x in self.expression])
            self.counts = counts / self.expression.shape[1] # counts obs x n_bins
        else:
            raise ValueError(f'Unrecognized mask style {mask_style}')

        # Mask indices are the var indices that are allowed to be masked
        if mask_indices is None:
            self.mask_bool = torch.ones(self.expression.shape[1], dtype=torch.bool)
        else:
            self.mask_bool = torch.zeros(self.expression.shape[1], dtype=torch.bool)
            self.mask_bool[mask_indices.long()] = True

    def __len__(self) -> int:
        if self.n_obs_sample is not None:
            return self.n_obs_sample
        return len(self.obs_indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self.n_obs_sample is None:
            actual_idx_obs = self.obs_indices[idx]
        else:
            actual_idx_obs = self.obs_indices[
                torch.randint(len(self.obs_indices), (1,))
            ]

        if self.n_var_sample is None:
            actual_idx_var = self.var_indices
        else:
            actual_idx_var = self.var_indices[
                torch.randperm(self.n_var)[: self.n_var_sample]
            ]

        # Transform expression
        expression = self.expression[actual_idx_obs, actual_idx_var]
        expression_target = expression.clone().detach()

        # Create mask
        mask_indices = create_mask(
            expression, self.mask_bool[actual_idx_var], self.mask_fraction, self.counts, #[:, actual_idx_var]
        )
        mask = torch.zeros(expression.shape, dtype=torch.bool)
        mask[mask_indices] = True

        # Pass, corrupt, or mask expression values
        expression = apply_mask(
            expression, mask_indices, self.n_bins, self.corrupt, self.pass_through
        )

        item = dict()
        item["expression_input"] = expression
        item["expression_target"] = expression_target
        item["mask"] = mask
        item["var_indices"] = actual_idx_var
        item["obs_idx"] = actual_idx_obs
        return item


def create_mask(
    expression: Tensor,
    mask_bool: Tensor,
    mask_fraction: float = 0.0,
    counts: Optional[Tensor] = None,
) -> Tensor:
    if mask_fraction == 0.0:
        # Mask no values
        return torch.empty(0)

    if mask_fraction == 1.0:
        # Mask all allowed values
        return torch.where(mask_bool)[0]

    mask_indices = torch.where(mask_bool)[0]
    if len(mask_indices) == 0:
        return mask_indices

    if counts is None:
        # Pick random set of allowed indices to mask
        use_mask_indices = torch.randperm(len(mask_indices))[: int(len(mask_indices) * mask_fraction)]
        return mask_indices[use_mask_indices]

    if counts.ndim == 1:
       result = counts[expression]
    else:
       result = counts[expression, torch.arange(expression.shape[0])]

    adj_values = 1 / result[mask_indices]
    # adj_values[torch.isinf(adj_values)] = 0

    use_mask_indices = torch.multinomial(
        adj_values,
        int(mask_fraction * len(expression)),
    )
    return mask_indices[use_mask_indices]


def apply_mask(
    expression: Tensor,
    mask_indices: Tensor,
    n_bins: int,
    corrupt: float = 0,
    pass_through: float = 0,
) -> Tensor:
    num_pass = int(pass_through * len(mask_indices))
    num_corrput = int(corrupt * len(mask_indices))
    if num_corrput > 0 and len(mask_indices) >= num_pass + num_corrput:
        # corrupt to values at rates proportial to observed counts ### FIX!!!!!
        values, counts = torch.unique(expression, return_counts=True)
        count_inds = torch.multinomial(counts.float(), num_corrput, replacement=True)
        expression[mask_indices[num_pass : num_pass + num_corrput]] = values[count_inds]
    if len(mask_indices) > num_pass + num_corrput:
        expression[mask_indices[num_pass + num_corrput :]] = n_bins

    return expression
