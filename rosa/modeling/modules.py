from typing import Optional, Tuple, Union

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from scanpy.plotting._matrixplot import MatrixPlot

from ..utils import merge_images, score_predictions
from ..utils.config import ModuleConfig
from .models import RosaTransformer


def sample(x: torch.Tensor, nbins: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    if nbins > 1:
        confidence, prediction = nn.functional.softmax(x, dim=-1).max(dim=-1)
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


class RosaLightningModule(LightningModule):
    def __init__(
        self,
        var_input: torch.Tensor,
        config: ModuleConfig,
        adata: ad.AnnData,
        weight: Optional[torch.Tensor] = None,
    ):
        super(RosaLightningModule, self).__init__()
        self.model = RosaTransformer(
            in_dim=var_input.shape[1],
            config=config.model,
        )
        self.register_buffer("var_input", var_input)
        self.optim_config = config.optimizer

        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.n_bins = config.model.n_bins
        self.adata = adata
        self.target = None
        self.marker_genes_dict = self.adata.obs.set_index("label").to_dict()[
            "marker_feature_name"
        ]
        sc.tl.dendrogram(self.adata, groupby="label", use_rep="X")

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, _):
        expression = batch["expression_target"]
        batch["var_input"] = self.var_input[batch["var_indices"]]
        expression_predicted = self(batch)
        expression_predicted = expression_predicted[batch["mask"]]
        expression = expression[batch["mask"]]
        loss = self.criterion(expression_predicted, expression)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # def validation_step(self, batch, _):
    #     expression = batch["expression_target"]
    #     batch["var_input"] = self.var_input[batch["var_indices"]]
    #     expression_predicted = self(batch)
    #     expression_predicted = expression_predicted[batch["mask"]]
    #     expression = expression[batch["mask"]]
    #     loss = self.criterion(expression_predicted, expression)
    #     self.log(
    #         "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
    #     )
    #     return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        expression = batch["expression_target"]
        batch["var_input"] = self.var_input[batch["var_indices"]]
        expression_predicted = self(batch)
        batch_size = batch["mask"].shape[0]

        expression_predicted = expression_predicted[batch["mask"]]
        expression = expression[batch["mask"]]
        loss = self.criterion(expression_predicted, expression)
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )

        results = {}
        results["expression_predicted"] = expression_predicted.reshape(
            batch_size, -1, self.n_bins
        )
        results["expression_target"] = expression.reshape(batch_size, -1)
        results["batch_idx"] = batch_idx
        results["dataloader_idx"] = dataloader_idx
        results["obs_idx"] = batch["obs_idx"]
        return results

    def validation_epoch_end(self, results):
        results = self.all_gather(results)
        predicted = []
        confidence = []
        target = []
        obs_idx = []
        for batch in results:
            p, c = sample(batch["expression_predicted"], nbins=self.n_bins)
            predicted.append(p)
            confidence.append(c)
            target.append(batch["expression_target"])
            obs_idx.append(batch["obs_idx"])

        obs_idx = torch.concat(obs_idx)
        order = torch.argsort(obs_idx)

        confidence = torch.concat(confidence)[order]
        target = torch.concat(target)[order]
        predicted = torch.concat(predicted)[order]

        scores = score_predictions(predicted, target, nbins=self.n_bins)
        self.log("spearman_obs_mean", scores["spearman_obs_mean"])
        self.log("spearman_var_mean", scores["spearman_var_mean"])

        cm = scores["confusion_matrix"]
        cm_norm = cm / cm.sum(dim=1)[:, None]

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_image(
            "confusion_matrix",
            torch.flip(cm_norm, (0,)),
            self.global_step,
            dataformats="HW",
        )

        if self.global_step > 0 and self.target is None:
            # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
            self.adata.layers["target"] = target.detach().cpu().numpy()
            # self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
            mp = MatrixPlot(
                self.adata,
                self.marker_genes_dict,
                groupby="label",
                gene_symbols="feature_name",
                layer="target",
                vmin=0,
                vmax=self.n_bins - 1,
                show=False,
                dendrogram=False,
            )
            mp.add_dendrogram(dendrogram_key=True)

            _color_df = mp.values_df.copy()
            if mp.var_names_idx_order is not None:
                _color_df = _color_df.iloc[:, mp.var_names_idx_order]

            if mp.categories_order is not None:
                _color_df = _color_df.loc[mp.categories_order, :]
            self.target = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
            tensorboard_logger.add_image(
                "cellXgene_1_target", self.target, self.global_step, dataformats="HW"
            )

        if self.global_step > 0:
            # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
            # self.adata.layers["target"] = target.detach().cpu().numpy()
            self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
            mp = MatrixPlot(
                self.adata,
                self.marker_genes_dict,
                groupby="label",
                gene_symbols="feature_name",
                layer="predicted",
                vmin=0,
                vmax=self.n_bins - 1,
                show=False,
                dendrogram=False,
            )
            mp.add_dendrogram(dendrogram_key=True)

            _color_df = mp.values_df.copy()
            if mp.var_names_idx_order is not None:
                _color_df = _color_df.iloc[:, mp.var_names_idx_order]

            if mp.categories_order is not None:
                _color_df = _color_df.loc[mp.categories_order, :]

            predicted = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
            tensorboard_logger.add_image(
                "cellXgene_2_predicted", predicted, self.global_step, dataformats="HW"
            )

            if self.target is not None:
                blended = merge_images(self.target, predicted)
                tensorboard_logger.add_image(
                    "cellXgene_3_blended", blended, self.global_step, dataformats="HWC"
                )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.optim_config.warmup, max_iters=self.optim_config.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    # def explain_iter(self, dataloader, explainer, indices=None):
    #     for x, y in iter(dataloader):
    #         if isinstance(x, list):
    #             x = tuple(
    #                 x_ind.reshape(-1, x_ind.shape[-1]).requires_grad_() for x_ind in x
    #             )
    #             attribution = explainer.attribute(x)
    #             yield tuple(a.reshape(y.shape[0], y.shape[1], -1) for a in attribution)
    #         else:
    #             attribution = []
    #             if indices is None:
    #                 indices = range(y.shape[-1])
    #             for target in indices:
    #                 x.requires_grad_()
    #                 attribution.append(explainer.attribute(x, target=int(target)))
    #             yield torch.stack(attribution, dim=1)
