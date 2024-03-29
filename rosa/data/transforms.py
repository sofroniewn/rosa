from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from ..utils.config import ExpressionTransformConfig


class ToTensor(nn.Module):
    """Convert ``numpy.ndarray`` to tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, tensor: Union[np.ndarray, csr_matrix]) -> torch.Tensor:
        if isinstance(tensor, csr_matrix):
            tensor = tensor.toarray()
        return torch.from_numpy(tensor).type(torch.float32).squeeze(dim=0)


class CountNormalize(nn.Module):
    """Normalize a tensor to a fixed total counts."""

    def __init__(self, total_counts: int = 1) -> None:
        super().__init__()
        self.total_counts = total_counts

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.total_counts * F.normalize(tensor, p=1.0, eps=1e-12)


class Log1p(nn.Module):
    """Log1p normalize a tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.log1p(tensor)


class QuantileNormalize(nn.Module):
    """Normalize a tensor by quantiles."""

    def __init__(self, n_bins: int, zero_bin: bool) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.zero_bin = zero_bin

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.zero_bin:
            # Put all zero values in their own bin and then distribute others evenly
            boundaries = torch.quantile(
                tensor[tensor > 0], torch.linspace(0, 1, self.n_bins)
            )
            boundaries = torch.concat([torch.tensor([0]), boundaries])
            boundaries[-1] = torch.inf
            return torch.bucketize(tensor, boundaries, right=True) - 1
        else:
            # boundaries = torch.quantile(tensor, torch.linspace(0, 1, self.n_bins))

            # boundaries = [torch.tensor(0.0)]
            # for q in range(self.n_bins - 1):
            #     data_remaining = tensor[tensor>boundaries[-1]]
            #     next_val = max(boundaries[-1]+1, torch.quantile(data_remaining, 1/(self.n_bins - q)))
            #     boundaries.append(next_val)
            # boundaries = torch.stack(boundaries, dim=0)

            # max_val = tensor.max() + 1
            # boundaries = torch.exp(torch.linspace(0, torch.log(max_val), self.n_bins))

            boundaries = [torch.tensor(0.0)]
            # min_step_size = 0.1
            for q in range(self.n_bins - 2):
                data_remaining = tensor[tensor > boundaries[-1]]
                n_data_remaining = len(data_remaining)
                if n_data_remaining == 0:
                    next_val = boundaries[-1]
                else:
                    # n_bins_remaining = self.n_bins - q - 1
                    # next_quantile = 1 / n_bins_remaining
                    # next_val = max(
                    #     boundaries[-1] + min_step_size,
                    #     torch.quantile(data_remaining, next_quantile),
                    # )
                    next_quantile = 1 / 2
                    next_val = torch.quantile(data_remaining, next_quantile)
                boundaries.append(next_val)
            boundaries = torch.stack(boundaries, dim=0)
            return torch.bucketize(tensor, boundaries)


class ExpressionTransform(nn.Sequential):
    def __init__(self, cfg: ExpressionTransformConfig) -> None:
        # Add base transform
        transforms = [ToTensor()]  # type: List[nn.Module]

        if cfg.total_counts is not None:
            transforms.append(CountNormalize(cfg.total_counts))

        if cfg.log1p:
            transforms.append(Log1p())

        if cfg.n_bins is not None:
            transforms.append(QuantileNormalize(cfg.n_bins, zero_bin=cfg.zero_bin))

        super().__init__(*transforms)
