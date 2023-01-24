from typing import Optional
from dataclasses import dataclass


@dataclass
class Paths:
    base: str
    adata: str
    chkpt: Optional[str]


@dataclass
class Adata:
    expression_layer: Optional[str]
    var_embedding: Optional[str]
    obs_embedding: Optional[str]


@dataclass
class Params:
    batch_size: int


@dataclass
class Model:
    head: str


@dataclass
class RosaConfig:
    paths: Paths
    adata: Adata
    params: Params
    model: Model
