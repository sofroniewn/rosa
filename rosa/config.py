from dataclasses import dataclass
from typing import Optional


@dataclass
class PathConfig:
    adata: str
    chkpt_dir: str
    chkpt: Optional[str]


@dataclass
class ExpressionTransformConfig:
    total_counts: Optional[int] = None # Total counts to normalize expression per cell
    log1p: Optional[bool] = None # Whether to log1p normalize expression data per cell
    n_bins: Optional[int] = None # Number of bins to quantile normalize data into per cell.
                                 # Note if quantile normalization applied then other normalizations are irrelvant.

@dataclass
class ParamConfig:
    batch_size: int
    learning_rate: float
    num_workers: Optional[int] = 0


@dataclass
class DataConfig:
    expression_layer: Optional[str] = None # If null use adata.X else use adata.layers[expression_layer]
    var_input: Optional[str] = None # If null no var input used
    obs_input: Optional[str] = None # If null no obs input used
    expression_transform: Optional[ExpressionTransformConfig] = None
#     n_obs_item: Optional[int] # If null return all obs, otherwise item will contain requested number of obs
#     n_var_item: Optional[int] # If null return all var, otherwise item will contain requested number of var



@dataclass
class Model:
    var_head: Optional[str]
    obs_head: Optional[str]
    head: Optional[str]
    method: Optional[str]
    rank: Optional[int]


@dataclass
class RosaConfig:
    paths: PathConfig
    data: DataConfig
    params: ParamConfig
    model: Model
