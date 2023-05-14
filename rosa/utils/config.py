from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass
class GeneEmbeddingsConfig:
    path: str
    gene_intervals: str
    pcs: int


@dataclass
class CellEmbeddingsConfig:
    pcs: int
    key: str


@dataclass
class PathConfig:
    adata: str
    base: str
    dataset: str
    preprocessed: Optional[str]
    chkpt_dir: str
    chkpt: Optional[str]


@dataclass
class SplitConfig:
    seed: int
    fraction: float
    key: str


@dataclass
class BulkDataConfig:
    sample_col: str
    label_col: str
    min_cells: int


@dataclass
class FilterConfig:
    coding_only: bool


@dataclass
class MarkerGeneConfig:
    label_col: str
    log1p: bool
    total_counts: int
    mean_expression_threshold: float
    score_quantile: float


@dataclass
class ExpressionTransformConfig:
    total_counts: Optional[int] = None  # Total counts to normalize expression per cell
    log1p: bool = False  # Whether to log1p normalize expression data per cell
    n_bins: Optional[int] = None  # Num bins to quantile normalize data per cell.
    zero_bin: bool = True  # Whether to zero use a seperate zero bin
    # Note if quantile normalization applied then other normalizations are irrelvant.


@dataclass
class PreProcessingConfig:
    gene_embeddings: GeneEmbeddingsConfig
    cell_embeddings: Optional[CellEmbeddingsConfig]
    bulk_data: Optional[BulkDataConfig]
    split: Optional[SplitConfig]
    filter: Optional[FilterConfig]
    markers: Optional[MarkerGeneConfig]
    expression_transform: Optional[ExpressionTransformConfig]


@dataclass
class DataConfig:
    expression_layer: Optional[
        str
    ] = None  # If null use adata.X else use adata.layers[expression_layer]
    var_input: Optional[str] = None  # If null no var input used
    obs_input: Optional[str] = None  # If null no obs input used
    expression_transform: Optional[ExpressionTransformConfig] = None
    n_var_sample: Optional[int] = None
    n_obs_sample: Optional[int] = None
    mask: float = 0
    pass_through: float = 0
    corrupt: float = 0


@dataclass
class DataModuleConfig:
    data: DataConfig
    batch_size: int
    accumulate: int
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
    dropout_prob: float
    projection: Optional[bool]
    activation: Optional[ExpressionHeadActivations]
    n_bins: Optional[int]


@dataclass
class InputEmbedConfig:
    dim: int
    pre_layer_norm: bool
    layer_norm: bool
    dropout_prob: float


@dataclass
class TransformerConfig:
    dim: int
    depth: int
    heads: int
    dim_head: int
    causal: bool


class LossFunctions(Enum):
    CE = auto()
    MSE = auto()
    MAE = auto()
    LOGPROB = auto()


@dataclass
class CriterionConfig:
    loss_function: LossFunctions


@dataclass
class ModelConfig:
    n_bins: int
    dim: int
    expression_head: ExpressionHeadConfig
    var_embed: InputEmbedConfig
    expression_embed: InputEmbedConfig
    transformer: TransformerConfig


@dataclass
class OptimizerConfig:
    learning_rate: float
    var_learning_rate: float
    beta_1: float
    beta_2: float
    eps: float
    weight_decay: float
    warmup: int
    max_iters: int


@dataclass
class ModuleConfig:
    model: ModelConfig
    criterion: CriterionConfig
    optimizer: OptimizerConfig


@dataclass
class TrainerConfig:
    device: str
    precision: str
    num_devices: int
    max_epochs: int
    val_check_interval: int
    limit_val_batches: int
    log_every_n_steps: int
    gradient_clip_val: Optional[float]


@dataclass
class RosaConfig:
    paths: PathConfig
    preprocessing: PreProcessingConfig
    data_module: DataModuleConfig
    module: ModuleConfig
    trainer: TrainerConfig
