from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


def sample(x: torch.Tensor, nbins: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    if nbins > 1:
        confidence, prediction = nn.functional.softmax(x.float(), dim=-1).max(dim=-1)
        return prediction, confidence
    return x, torch.ones_like(x)


# from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html?highlight=warm
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def reconstruct_from_results(results, nbins):
    predicted = []
    confidence = []
    target = []
    obs_idx = []

    for batch in results:
        predicted_batch = batch["expression_predicted"]
        target_batch = batch["expression_target"]
        obs_idx_batch = batch["obs_idx"]
        # If results have been gathered across devices then have shape n_device x n_batch ....
        predicted_batch = predicted_batch.view(-1, predicted_batch.shape[-2], predicted_batch.shape[-1])
        target_batch = target_batch.view(-1, target_batch.shape[-1])
        obs_idx_batch = obs_idx_batch.view(-1)

        p, c = sample(predicted_batch, nbins=nbins)
        predicted.append(p)
        confidence.append(c)
        target.append(target_batch)
        obs_idx.append(obs_idx_batch)

    obs_idx = torch.concat(obs_idx)
    unique_values, unique_indices = torch.unique(obs_idx, return_inverse=True)
    sorted_indices = torch.argsort(unique_values)
    order = unique_indices[sorted_indices]

    confidence = torch.concat(confidence)[order]
    target = torch.concat(target)[order]
    predicted = torch.concat(predicted)[order]

    return target, predicted, confidence
