from typing import Callable

import torch
import torch.nn.functional as F
import torchmetrics.functional as tm_F

from ....utils.config import CriterionConfig, LossFunctions  # type: ignore


def mean_log_prob_criterion(
    output: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor:
    return -output.log_prob(target).sum(-1).mean()


def criterion_factory(config: CriterionConfig) -> Callable:
    if config.loss_function == LossFunctions.MSE.name.lower():
        return tm_F.mean_squared_error
    if config.loss_function == LossFunctions.MAE.name.lower():
        return tm_F.mean_absolute_error
    if config.loss_function == LossFunctions.CE.name.lower():
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    if config.loss_function == LossFunctions.LOGPROB.name.lower():
        return mean_log_prob_criterion

    raise ValueError(f"Loss function {config.loss_function} not recognized")
