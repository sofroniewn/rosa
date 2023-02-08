import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.config import DataConfig, ExpressionTransformConfig
from .transforms import ExpressionTransform, ToTensor

warnings.simplefilter(action="ignore", category=FutureWarning)


class _SingleDataset(Dataset):
    """Return a single input and experession vector"""

    def __init__(
        self,
        expression: torch.Tensor,
        input: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> None:

        if not expression.shape[0] == input.shape[0]:
            raise ValueError(
                f"Number of expression and input values must match, got {expression.shape[0]} and {input.shape[0]}"
            )

        if indices is None:
            indices = torch.arange(expression.shape[0])

        if indices.max() >= expression.shape[0]:
            raise ValueError(
                f"Max index {indices.max()} too long for {expression.shape[0]} samples"
            )

        if indices.min() < 0:
            raise ValueError(f"Min index {indices.min()} less than zero")

        self.expression = expression
        self.indices = indices.long()
        self.expression_dim = self.expression.shape[1]
        self.input = input
        self.input_dim = self.input.shape[1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        actual_idx = self.indices[idx]
        return (
            self.input[actual_idx],
            self.expression[actual_idx],
        )

    def _postprocess(self, results: List[Tensor]) -> torch.Tensor:
        return torch.concat(results)


class _JointDataset(Dataset):
    """Return two inputs and an experession value"""

    def __init__(
        self,
        expression: torch.Tensor,
        input: Tuple[torch.Tensor, torch.Tensor],
        indices: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None),
    ) -> None:

        if not expression.shape[0] == input[0].shape[0]:
            raise ValueError(
                f"Number of expression and first input values must match, got {expression.shape[0]} and {input[0].shape[0]}"
            )

        if not expression.shape[1] == input[1].shape[0]:
            raise ValueError(
                f"Number of expression and second input values must match, got {expression.shape[1]} and {input[1].shape[0]}"
            )

        if indices[0] is None:
            indices = (torch.arange(expression.shape[0]), indices[1])

        if indices[1] is None:
            indices = (indices[0], torch.arange(expression.shape[1]))

        assert indices[0] is not None and indices[1] is not None

        if indices[0].max() >= expression.shape[0]:
            raise ValueError(
                f"Max index {indices[0].max()} too long for {expression.shape[0]} samples"
            )

        if indices[1].max() >= expression.shape[1]:
            raise ValueError(
                f"Max index {indices[1].max()} too long for {expression.shape[1]} samples"
            )

        if indices[0].min() < 0:
            raise ValueError(f"Min index {indices[0].min()} less than zero")

        if indices[1].min() < 0:
            raise ValueError(f"Min index {indices[1].min()} less than zero")

        self.expression = expression
        self.expression_dim = 1
        self.indices = (
            indices[0].long(),
            indices[1].long(),
        )
        self.input = input
        self.input_dim = (
            self.input[0].shape[1],
            self.input[1].shape[1],
        )

    def __len__(self) -> int:
        return len(self.indices[0]) * len(self.indices[1])

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        idx_0, idx_1 = tuple(
            int(_idx)
            for _idx in np.unravel_index(
                idx, (len(self.indices[0]), len(self.indices[1]))
            )
        )
        actual_idx0 = self.indices[0][idx_0]
        actual_idx1 = self.indices[1][idx_1]
        return (
            (
                self.input[0][actual_idx0],
                self.input[1][actual_idx1],
            ),
            self.expression[actual_idx0, actual_idx1],
        )

    def _postprocess(self, results: List[Tensor]) -> torch.Tensor:
        return torch.concat(results).reshape(*self.expression.shape)


class RosaObsDataset(_SingleDataset):
    def __init__(
        self,
        adata: AnnData,
        *,
        obs_input: str,
        obs_indices: Optional[torch.Tensor] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
    ) -> None:
        self.adata = adata
        expression = _prepare_expression(
            adata, expression_layer, expression_transform_config
        )

        input_transform = ToTensor()
        raw_input = adata.obsm[obs_input]
        input = input_transform(raw_input)

        super().__init__(expression, input, indices=obs_indices)

    def predict(
        self, results: List[Tensor], prediction_layer: str = "prediction"
    ) -> None:
        self.adata.layers[prediction_layer] = super()._postprocess(results).numpy()


class RosaVarDataset(_SingleDataset):
    def __init__(
        self,
        adata: AnnData,
        *,
        var_input: str,
        var_indices: Optional[torch.Tensor] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
    ) -> None:
        self.adata = adata
        expression = _prepare_expression(
            adata, expression_layer, expression_transform_config
        )

        input_transform = ToTensor()
        raw_input = adata.varm[var_input]
        input = input_transform(raw_input)

        super().__init__(expression.T, input, indices=var_indices)

    def predict(
        self, results: List[Tensor], prediction_layer: str = "prediction"
    ) -> None:
        self.adata.layers[prediction_layer] = super()._postprocess(results).numpy().T


class RosaJointDataset(_JointDataset):
    def __init__(
        self,
        adata: AnnData,
        *,
        var_input: str,
        obs_input: str,
        obs_indices: Optional[torch.Tensor] = None,
        var_indices: Optional[torch.Tensor] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
    ) -> None:
        self.adata = adata
        expression = _prepare_expression(
            adata, expression_layer, expression_transform_config
        )

        input_transform = ToTensor()
        raw_var_input = adata.varm[var_input]
        raw_obs_input = adata.obsm[obs_input]
        input = (
            input_transform(raw_obs_input),
            input_transform(raw_var_input),
        )

        super().__init__(expression, input, indices=(obs_indices, var_indices))

    def predict(
        self, results: List[Tensor], prediction_layer: str = "prediction"
    ) -> None:
        self.adata.layers[prediction_layer] = super()._postprocess(results).numpy()


class RosaObsVarDataset(RosaJointDataset):
    """Iterates over cells

    When used with a batch of cell, returns data as (BxGxCE, BxGxGE), BxG
    where B is number of cells in batch, G is number of genes, CE is length
    of cell embedding, GE is length of gene embedding.
    """

    def __init__(
        self,
        adata: AnnData,
        *,
        var_input: str,
        obs_input: str,
        obs_indices: Optional[torch.Tensor] = None,
        var_indices: Optional[torch.Tensor] = None,
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
        n_var_sample: Optional[int] = None,
    ) -> None:
        super(RosaObsVarDataset, self).__init__(
            adata,
            obs_input=obs_input,
            var_input=var_input,
            obs_indices=obs_indices,
            var_indices=var_indices,
            expression_layer=expression_layer,
            expression_transform_config=expression_transform_config,
        )
        self.var_subindices = torch.arange(len(self.indices[1])).float()
        self.n_var_sample = n_var_sample

    def __len__(self) -> int:
        return len(self.indices[0])

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        actual_idx = self.indices[0][idx]
        obs_input = self.input[0][actual_idx]
        expression = self.expression[actual_idx][self.indices[1]]
        var_input = self.input[1][self.indices[1]]
        if self.n_var_sample is not None:
            sample_var = torch.multinomial(self.var_subindices, self.n_var_sample)
            expression = expression[sample_var]
            var_input = var_input[sample_var]
        full_input = (
            obs_input.expand((var_input.shape[0], obs_input.shape[0])),
            var_input,
        )
        return full_input, expression


def _prepare_expression(
    adata: AnnData,
    expression_layer: Optional[str] = None,
    expression_transform_config: Optional[ExpressionTransformConfig] = None,
) -> torch.Tensor:
    # extract expression, shape n_obs x n_var
    if expression_layer is None:
        raw_expression = adata.X  # type: np.ndarray
    else:
        raw_expression = adata.layers[expression_layer]

    if expression_transform_config is None:
        expression_transform_config = ExpressionTransformConfig()
    expression_transform = ExpressionTransform(expression_transform_config)
    return expression_transform(raw_expression)


def rosa_dataset_factory(
    adata: AnnData,
    data_config: DataConfig,
    *,
    obs_indices: Optional[torch.Tensor] = None,
    var_indices: Optional[torch.Tensor] = None,
) -> Union[RosaObsDataset, RosaVarDataset, RosaJointDataset, RosaObsVarDataset]:
    if data_config.obs_input is not None and data_config.var_input is not None:
        return RosaObsVarDataset(
            adata,
            var_input=data_config.var_input,
            obs_input=data_config.obs_input,
            obs_indices=obs_indices,
            var_indices=var_indices,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    if data_config.obs_input is None and data_config.var_input is not None:
        return RosaVarDataset(
            adata,
            var_input=data_config.var_input,
            var_indices=var_indices,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    if data_config.obs_input is not None and data_config.var_input is None:
        return RosaObsDataset(
            adata,
            obs_input=data_config.obs_input,
            obs_indices=obs_indices,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    raise ValueError("One of the var input or the obs input must not be None")
