from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    adata: str
    chkpt_dir: str
    chkpt: Optional[str]


@dataclass
class Adata:
    expression_layer: Optional[str]
    var_embedding: Optional[str]
    obs_embedding: Optional[str]


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
