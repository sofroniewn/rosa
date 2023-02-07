from typing import Callable

import torch
import torchmetrics.functional as F

from ...utils.config import CriterionConfig, LossFunctions


def mean_log_prob_criterion(
    output: torch.distributions.Distribution, target: torch.Tensor
) -> torch.Tensor:
    return -output.log_prob(target).sum(-1).mean()


def criterion_factory(config: CriterionConfig) -> Callable:
    if config.loss_function == LossFunctions.MSE.name.lower():
        return F.mean_squared_error
    if config.loss_function == LossFunctions.MAE.name.lower():
        return F.mean_absolute_error
    if config.loss_function == LossFunctions.LOGPROB.name.lower():
        return mean_log_prob_criterion

    raise ValueError(f"Loss function {config.loss_function} not recognized")
