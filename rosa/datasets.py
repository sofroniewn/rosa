import warnings
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset

from .preprocessing import reconstruct_expression

warnings.simplefilter(action="ignore", category=FutureWarning)


class EmbeddingType(Enum):
    JOINT = auto()
    VAR = auto()
    OBS = auto()


class RosaDataset(Dataset):
    """Return obs and var embeddings as needed along with experession"""

    def __init__(
        self,
        adata: AnnData,
        expression_layer: Optional[str] = None,
        var_embedding: Optional[str] = None,
        obs_embedding: Optional[str] = None,
    ) -> None:
        if obs_embedding is not None and var_embedding is not None:
            self.embedding_type = EmbeddingType.JOINT
        elif obs_embedding is None and var_embedding is not None:
            self.embedding_type = EmbeddingType.VAR
        elif obs_embedding is not None and var_embedding is None:
            self.embedding_type = EmbeddingType.OBS
        else:
            raise ValueError(
                "One of the var embedding or the obs embedding must not be None"
            )

        self.adata = adata

        # Store input and target keys
        self._EXPRESSION_LAYER_KEY = expression_layer
        self._VAR_EMBEDDING_KEY = var_embedding
        self._OBS_EMBEDDING_KEY = obs_embedding

        if self._EXPRESSION_LAYER_KEY == "binned":
            self._expression_type = torch.long
        else:
            self._expression_type = torch.float32

        # expression shape n_obs x n_var
        if self._EXPRESSION_LAYER_KEY is None:
            self.expression = self.adata.X  # type: np.ndarray
        else:
            self.expression = self.adata.layers[self._EXPRESSION_LAYER_KEY]
        self._n_obs, self._n_var = self.expression.shape

        if self._VAR_EMBEDDING_KEY is not None:
            # var embedding shape n_var x var embedding length
            self.var_embedding = self.adata.varm[
                self._VAR_EMBEDDING_KEY
            ]  # type: np.ndarray
        else:
            self.var_embedding = np.empty((self._n_var, 0))
        self._len_var_embedding = self.var_embedding.shape[1]

        if self._OBS_EMBEDDING_KEY is not None:
            # var embedding shape n_var x var embedding length
            self.obs_embedding = self.adata.obsm[
                self._OBS_EMBEDDING_KEY
            ]  # type: np.ndarray
        else:
            self.obs_embedding = np.empty((self._n_obs, 0))
        self._len_obs_embedding = self.obs_embedding.shape[1]

        if self.embedding_type == EmbeddingType.JOINT:
            self.len_embedding = (
                self._len_obs_embedding,
                self._len_var_embedding,
            )  # type: Union[int, Tuple[int, int]]
            self.len_target = 1
        elif self.embedding_type == EmbeddingType.VAR:
            self.len_embedding = self._len_var_embedding
            self.len_target = self._n_obs
        elif self.embedding_type == EmbeddingType.OBS:
            self.len_embedding = self._len_obs_embedding
            self.len_target = self._n_var
        else:
            raise ValueError(
                f"Type {self.embedding_type.name} not recognized, must be one of {list(EmbeddingType.__members__)}"
            )

    def __len__(self) -> int:
        if self.embedding_type == EmbeddingType.JOINT:
            return self._n_obs * self._n_var
        if self.embedding_type == EmbeddingType.VAR:
            return self._n_var
        if self.embedding_type == EmbeddingType.OBS:
            return self._n_obs
        raise ValueError(
            f"Type {self.embedding_type.name} not recognized, must be one of {list(EmbeddingType.__members__)}"
        )

    def __getitem__(self, idx) -> Tuple[Union[Tuple[Tensor, Tensor], Tensor], Tensor]:
        if self.embedding_type == EmbeddingType.JOINT:
            # Extract data
            obs_i, var_j = np.unravel_index(idx, (self._n_obs, self._n_var))
            obs = self.obs_embedding[obs_i]
            var = self.var_embedding[var_j]
            expression = self.expression[obs_i, var_j]
            # Move to torch
            expression = torch.tensor(expression).type(self._expression_type)
            obs = torch.from_numpy(obs).type(torch.float32)
            var = torch.from_numpy(var).type(torch.float32)
            return (obs, var), expression
        if self.embedding_type == EmbeddingType.VAR:
            # Extract data
            var = self.var_embedding[idx]
            expression = self.expression[:, idx]
            # Move to torch
            expression = torch.tensor(expression).type(self._expression_type)
            var = torch.from_numpy(var).type(torch.float32)
            return var, expression
        if self.embedding_type == EmbeddingType.OBS:
            # Extract data
            obs = self.obs_embedding[idx]
            expression = self.expression[idx, :]
            # Move to torch
            expression = torch.tensor(expression).type(self._expression_type)
            obs = torch.from_numpy(obs).type(torch.float32)
            return obs, expression

        raise ValueError(
            f"Type {self.embedding_type.name} not recognized, must be one of {list(EmbeddingType.__members__)}"
        )

    def postprocess(self, results: List[Any]):
        if self.embedding_type == EmbeddingType.JOINT:
            prediction = torch.concat(results).numpy().reshape(self._n_obs, self._n_var)
        elif self.embedding_type == EmbeddingType.OBS:
            prediction = torch.concat(results).numpy()
        elif self.embedding_type == EmbeddingType.VAR:
            prediction = torch.concat(results).numpy().T
        else:
            raise ValueError(
                f"Type {self.embedding_type.name} not recognized, must be one of {list(EmbeddingType.__members__)}"
            )

        if self._EXPRESSION_LAYER_KEY == "binned":
            self.adata.layers["binned_prediction"] = prediction
            reconstruct_expression(
                self.adata, input_layer="binned_prediction", output_layer="prediction"
            )
        else:
            self.adata.layers["prediction"] = prediction
