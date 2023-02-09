import warnings
from .predict import predict
from .train import train

from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning


warnings.simplefilter(action='ignore', category=LightningDeprecationWarning)
