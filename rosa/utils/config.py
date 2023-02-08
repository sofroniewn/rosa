from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass
class EmbeddingPathsConfig:
    gene_intervals: str
    embeddings: str


@dataclass
class PathConfig:
    adata: str
    base: str
    dataset: str
    preprocessed: str
    chkpt_dir: str
    chkpt: Optional[str]
    gene_embeddings: EmbeddingPathsConfig


@dataclass
class PreProcessingConfig:
    bulk_data: None
    splits: None


dataclass
class BulkDataConfig:
    sample_col: str
    label_col: str


dataclass
class SplitsConfig:
    seed: int
    train_fraction: float


@dataclass
class ExpressionTransformConfig:
    total_counts: Optional[int] = None  # Total counts to normalize expression per cell
    log1p: Optional[bool] = None  # Whether to log1p normalize expression data per cell
    n_bins: Optional[
        int
    ] = None  # Number of bins to quantile normalize data into per cell.
    # Note if quantile normalization applied then other normalizations are irrelvant.


@dataclass
class DataConfig:
    expression_layer: Optional[
        str
    ] = None  # If null use adata.X else use adata.layers[expression_layer]
    var_input: Optional[str] = None  # If null no var input used
    obs_input: Optional[str] = None  # If null no obs input used
    expression_transform: Optional[ExpressionTransformConfig] = None


#     n_obs_item: Optional[int] # If null return all obs, otherwise item will contain requested number of obs
#     n_var_item: Optional[int] # If null return all var, otherwise item will contain requested number of var


@dataclass
class DataModuleConfig:
    data: DataConfig
    batch_size: int
    num_workers: Optional[int] = 0


class ExpressionHeadActivations(Enum):
    SOFTPLUS = auto()
    SOFTMAX = auto()


class ExpressionHeadLikelihood(Enum):
    ZINB = auto()
    NB = auto()
    NBM = auto()


@dataclass
class ExpressionHeadConfig:
    projection: Optional[bool]
    activation: Optional[ExpressionHeadActivations]
    library_size: Optional[int]
    likelihood: Optional[ExpressionHeadLikelihood]


@dataclass
class InputEmbedConfig:
    dropout_prob: float
    layer_norm: bool
    embedding_dim: int


@dataclass
class FeedForwardConfig:
    dropout_prob: float
    hidden_dim: int


class JoinEmbedsMethods(Enum):
    ADD = auto()
    CAT = auto()
    BILINEAR = auto()
    ATTENTION = auto()
    DOT = auto()


@dataclass
class JoinEmbedsConfig:
    method: JoinEmbedsMethods
    out_dim: Optional[int]


class LossFunctions(Enum):
    MSE = auto()
    MAE = auto()
    LOGPROB = auto()


@dataclass
class CriterionConfig:
    loss_function: LossFunctions


@dataclass
class ModelConfig:
    dropout_prob: float
    layer_norm: bool
    expression_head: ExpressionHeadConfig
    feed_forward: Optional[FeedForwardConfig]
    input_embed: Optional[InputEmbedConfig]
    join_embeds: Optional[JoinEmbedsConfig]
    input_embed_1: Optional[InputEmbedConfig]
    layer_norm_1: Optional[bool]


@dataclass
class ModuleConfig:
    model: ModelConfig
    criterion: CriterionConfig
    learning_rate: float


@dataclass
class RosaConfig:
    paths: PathConfig
    preprocessing: PreProcessingConfig
    data_module: DataModuleConfig
    module: ModuleConfig
