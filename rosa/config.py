from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    adata: str
    chkpt_dir: str
    chkpt: Optional[str]


@dataclass
class AdataKeys:
    expression_layer: Optional[str] # If null use adata.X else use adata.layers[expression_layer]
    var_embedding: Optional[str] # If null no var embedding used
    obs_embedding: Optional[str] # If null no obs embedding used
    n_obs_item: Optional[int] # If null return all obs, otherwise item will contain requested number of obs
    n_var_item: Optional[int] # If null return all var, otherwise item will contain requested number of var

@dataclass
class ExpressionTransforms:
    total_counts: Optional[int] # Total counts to normalize expression per cell
    lop1p: Optional[bool] # Whether to log1p normalize expression data per cell
    n_bins: Optional[int] # Number of bins to quantile normalize data into per cell.
                          # Note if quantile normalization applied then other normalizations are irrelvant.

@dataclass
class Params:
    batch_size: int
    learning_rate: float


@dataclass
class Model:
    var_head: Optional[str]
    obs_head: Optional[str]
    head: Optional[str]
    method: Optional[str]
    rank: Optional[int]


@dataclass
class RosaConfig:
    paths: Paths
    adata: Adata
    params: Params
    model: Model
