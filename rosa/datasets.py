import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig, ExpressionTransformConfig
from .transforms import ExpressionTransform, ToTensor

warnings.simplefilter(action="ignore", category=FutureWarning)


class _SingleDataset(Dataset):
    """Return a single input and experession vector"""

    def __init__(
        self,
        expression: torch.Tensor,
        input: torch.Tensor,
    ) -> None:

        if not expression.shape[0] == input.shape[0]:
            raise ValueError(
                f"Number of expression and input values must match, got {expression.shape[0]} and {input.shape[0]}"
            )

        self.expression = expression
        self.expression_dim = self.expression.shape[1]
        self.input = input
        self.input_dim = self.input.shape[1]

    def __len__(self) -> int:
        return self.expression.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (
            self.input[idx],
            self.expression[idx],
        )

    def _postprocess(self, results: List[Tensor]) -> torch.Tensor:
        return torch.concat(results)


class _JointDataset(Dataset):
    """Return two inputs and an experession value"""

    def __init__(
        self,
        expression: torch.Tensor,
        input: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:

        if not expression.shape[0] == input[0].shape[0]:
            raise ValueError(
                f"Number of expression and first input values must match, got {expression.shape[0]} and {input[0].shape[0]}"
            )

        if not expression.shape[1] == input[1].shape[0]:
            raise ValueError(
                f"Number of expression and second input values must match, got {expression.shape[1]} and {input[1].shape[0]}"
            )

        self.expression = expression
        self.expression_dim = 1
        self.input = input
        self.input_dim = (
            self.input[0].shape[1],
            self.input[1].shape[1],
        )

    def __len__(self) -> int:
        return self.expression.shape[0] * self.expression.shape[1]

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        idx_0, idx_1 = tuple(
            int(_idx) for _idx in np.unravel_index(idx, self.expression.shape)
        )
        return (
            (
                self.input[0][idx_0],
                self.input[1][idx_1],
            ),
            self.expression[idx_0, idx_1],
        )

    def _postprocess(self, results: List[Tensor]) -> torch.Tensor:
        return torch.concat(results).reshape(*self.expression.shape)


class RosaObsDataset(_SingleDataset):
    def __init__(
        self,
        adata: AnnData,
        *,
        obs_input: str,
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

        super().__init__(expression, input)

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

        super().__init__(expression.T, input)

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

        super().__init__(expression, input)

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
        expression_layer: Optional[str] = None,
        expression_transform_config: Optional[ExpressionTransformConfig] = None,
        n_var_sample: Optional[int] = None,
    ) -> None:
        super(RosaObsVarDataset, self).__init__(adata, obs_input=obs_input, var_input=var_input, expression_layer=expression_layer, expression_transform_config=expression_transform_config)
        self.var_indices = torch.arange(self.expression.shape[1]).float()
        self.n_var_sample = n_var_sample

    def __len__(self) -> int:
        return self.expression.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        obs_input = self.input[0][idx]
        expression = self.expression[idx]
        var_input = self.input[1]
        if self.n_var_sample is not None:
            sample_var = torch.multinomial(self.var_indices, self.n_var_sample)
            expression = expression[sample_var]
            var_input = var_input[sample_var]
        full_input = (obs_input.expand((var_input.shape[0], obs_input.shape[0])), var_input)
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
    adata: AnnData, data_config: DataConfig
) -> Union[RosaObsDataset, RosaVarDataset, RosaJointDataset, RosaObsVarDataset]:
    if data_config.obs_input is not None and data_config.var_input is not None:
        return RosaObsVarDataset(
            adata,
            var_input=data_config.var_input,
            obs_input=data_config.obs_input,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    if data_config.obs_input is None and data_config.var_input is not None:
        return RosaVarDataset(
            adata,
            var_input=data_config.var_input,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    if data_config.obs_input is not None and data_config.var_input is None:
        return RosaObsDataset(
            adata,
            obs_input=data_config.obs_input,
            expression_layer=data_config.expression_layer,
            expression_transform_config=data_config.expression_transform,
        )

    raise ValueError("One of the var input or the obs input must not be None")
