from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ExpressionTransformConfig


class ToTensor(nn.Module):
    """Convert ``numpy.ndarray`` to tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, tensor: Union[np.ndarray, csr_matrix]) -> torch.Tensor:
        if isinstance(tensor, csr_matrix):
            tensor = tensor.toarray()
        return torch.from_numpy(tensor).type(torch.float32)


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

    def __init__(self, n_bins: int, zero_bin=True) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.zero_bin = zero_bin

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.zero_bin:
            # Put all zero values in their own bin and then distribute others evenly
            boundaries = torch.quantile(tensor[tensor>0], torch.linspace(0, 1, self.n_bins))
            boundaries = torch.concat([torch.tensor([0]), boundaries])
            boundaries[-1] = torch.inf
            return torch.bucketize(tensor, boundaries, right=True) - 1
        else:
            boundaries = torch.quantile(tensor, torch.linspace(0, 1, self.n_bins))
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
            transforms.append(QuantileNormalize(cfg.n_bins))

        super().__init__(*transforms)
