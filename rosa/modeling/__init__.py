import warnings

from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning

from .predict import predict
from .train import train

warnings.simplefilter(action="ignore", category=LightningDeprecationWarning)
