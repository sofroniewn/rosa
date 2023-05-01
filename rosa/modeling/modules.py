from typing import Optional

import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics.functional import spearman_corrcoef
from scanpy.plotting._matrixplot import MatrixPlot

from ..utils import merge_images, score_predictions
from ..utils.helpers import sample, CosineWarmupScheduler
from ..utils.config import ModuleConfig
from .models import RosaTransformer


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
            in_dim=var_input.shape[1 + 1],
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

        # var_input = self.var_input[:, batch["var_indices"]]
        # batch["var_input"] = batch["var_indices"]
        batch["var_input"] = self.var_input[0, batch["var_indices"]]

        # sigma = 1.0
        # var_input += sigma * torch.randn_like(var_input)

        # Randomly switch between var input indices for random sampling
        # random_indices = torch.randint(0, var_input.shape[0], (var_input.shape[1],))
        # batch["var_input"] = var_input[random_indices, torch.arange(var_input.shape[1])]
        # batch["var_input"] = var_input[0]

        expression_predicted = self(batch)
        expression_predicted = expression_predicted[batch["mask"]]
        expression = expression[batch["mask"]]

        # predicted, _ = sample(expression_predicted, nbins=self.n_bins)
        # spearman_obs_mean = 1 - spearman_corrcoef(predicted.T.float(), expression.T.float()).mean()
        # spearman_var_mean = 1 - spearman_corrcoef(predicted.float(), expression.float()).mean()
        # spearman_var_mean = 0.0
        # beta_obs_loss = 1.0
        # beta_var_loss = 1.0

        loss = self.criterion(expression_predicted, expression)

        # self.log(
        #     "train_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # self.log(
        #     "train_obs_loss", spearman_obs_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # self.log(
        #     "train_var_loss", spearman_var_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # loss = ce_loss + beta_obs_loss * spearman_obs_mean + beta_var_loss * spearman_var_mean

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # n = 0
        # reg_loss = 0
        # lambda_reg = 100
        # for layer in self.model.transformer.core.layers:
        #     for head in layer.attention.heads:
        #         reg_loss += torch.linalg.norm(head.k.weight, ord=1)
        #         reg_loss += torch.linalg.norm(head.q.weight, ord=1)
        #         n += 1

        # loss += lambda_reg * reg_loss / n

        # self.log(
        #     "train_loss_reg",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

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
        # batch["var_input"] = batch["var_indices"]
        batch["var_input"] = self.var_input[0, batch["var_indices"]]
        expression_predicted = self(batch)
        batch_size = batch["mask"].shape[0]
        expression_predicted = expression_predicted[batch["mask"]]
        expression = expression[batch["mask"]]

        # predicted, _ = sample(expression_predicted, nbins=self.n_bins)
        # spearman_obs_mean = 1 - spearman_corrcoef(predicted.T.float(), expression.T.float()).mean()
        # spearman_var_mean = 0.0
        # spearman_var_mean = 1 - spearman_corrcoef(predicted.float(), expression.float()).mean()
        # beta_obs_loss = 1.0
        # beta_var_loss = 1.0

        loss = self.criterion(expression_predicted, expression)

        # self.log(
        #     "val_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # self.log(
        #     "val_obs_loss", spearman_obs_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # self.log(
        #     "val_var_loss", spearman_var_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )
        # loss = ce_loss + beta_obs_loss * spearman_obs_mean + beta_var_loss * spearman_var_mean

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # n = 0
        # reg_loss = 0
        # lambda_reg = 100
        # for layer in self.model.transformer.core.layers:
        #     for head in layer.attention.heads:
        #         reg_loss += torch.linalg.norm(head.k.weight, ord=1)
        #         reg_loss += torch.linalg.norm(head.q.weight, ord=1)
        #         n += 1

        # loss += lambda_reg * reg_loss / n

        # self.log(
        #     "val_loss_reg",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )

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
 
        if self.global_rank != 0:
            return
        
        predicted = []
        confidence = []
        target = []
        obs_idx = []
        for batch in results:
            expression_predicted_batch = batch["expression_predicted"]
            expression_target_batch = batch["expression_target"]
            obs_idx_batch = batch["obs_idx"]
            if expression_predicted_batch.ndim == 4:
                expression_predicted_batch = expression_predicted_batch.view(-1, expression_predicted_batch.shape[2], expression_predicted_batch.shape[3])
            if expression_target_batch.ndim == 3:
                expression_target_batch = expression_target_batch.view(-1, expression_target_batch.shape[2])
            if obs_idx_batch.ndim == 2:
                obs_idx_batch = obs_idx_batch.view(-1)

            p, c = sample(expression_predicted_batch, nbins=self.n_bins)
            predicted.append(p)
            confidence.append(c)
            target.append(expression_target_batch)
            obs_idx.append(obs_idx_batch)

        obs_idx = torch.concat(obs_idx)
        order = torch.argsort(obs_idx)

        confidence = torch.concat(confidence)[order]
        target = torch.concat(target)[order]
        predicted = torch.concat(predicted)[order]

        scores = score_predictions(predicted, target, nbins=self.n_bins)
        self.log("spearman_obs_mean", scores["spearman_obs_mean"])
        self.log("spearman_var_mean", scores["spearman_var_mean"])

        # cm = scores["confusion_matrix"]
        # cm_norm = cm / cm.sum(dim=1)[:, None]

        # tensorboard_logger = self.logger.experiment
        # tensorboard_logger.add_image(
        #     "confusion_matrix",
        #     torch.flip(cm_norm, (0,)),
        #     self.global_step,
        #     dataformats="HW",
        # )

        # if self.global_step > 0 and self.target is None:
        #     # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
        #     self.adata.layers["target"] = target.detach().cpu().numpy()
        #     # self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
        #     mp = MatrixPlot(
        #         self.adata,
        #         self.marker_genes_dict,
        #         groupby="label",
        #         gene_symbols="feature_name",
        #         layer="target",
        #         vmin=0,
        #         vmax=self.n_bins - 1,
        #         show=False,
        #         dendrogram=False,
        #     )
        #     mp.add_dendrogram(dendrogram_key=True)

        #     _color_df = mp.values_df.copy()
        #     if mp.var_names_idx_order is not None:
        #         _color_df = _color_df.iloc[:, mp.var_names_idx_order]

        #     if mp.categories_order is not None:
        #         _color_df = _color_df.loc[mp.categories_order, :]
        #     self.target = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
        #     tensorboard_logger.add_image(
        #         "cellXgene_1_target", self.target, self.global_step, dataformats="HW"
        #     )

        # if self.global_step > 0:
        #     # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
        #     # self.adata.layers["target"] = target.detach().cpu().numpy()
        #     self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
        #     mp = MatrixPlot(
        #         self.adata,
        #         self.marker_genes_dict,
        #         groupby="label",
        #         gene_symbols="feature_name",
        #         layer="predicted",
        #         vmin=0,
        #         vmax=self.n_bins - 1,
        #         show=False,
        #         dendrogram=False,
        #     )
        #     mp.add_dendrogram(dendrogram_key=True)

        #     _color_df = mp.values_df.copy()
        #     if mp.var_names_idx_order is not None:
        #         _color_df = _color_df.iloc[:, mp.var_names_idx_order]

        #     if mp.categories_order is not None:
        #         _color_df = _color_df.loc[mp.categories_order, :]

        #     predicted = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
        #     tensorboard_logger.add_image(
        #         "cellXgene_2_predicted", predicted, self.global_step, dataformats="HW"
        #     )

        #     if self.target is not None:
        #         blended = merge_images(self.target, predicted)
        #         tensorboard_logger.add_image(
        #             "cellXgene_3_blended", blended, self.global_step, dataformats="HWC"
        #         )

    def configure_optimizers(self):
        # # Create the list of parameter groups
        # params = []

        # # Initialize a set to collect the special parameters
        # special_params = set()

        # # Add the special parameters to separate parameter groups and collect them in the special_params set
        # for layer in self.model.transformer.core.layers:
        #     for head in layer.attention.heads:
        #         for p in [head.k, head.q]:
        #             special_params.update(p.parameters())
        #             params.append(
        #                 {
        #                     "params": list(p.parameters()),
        #                     "lr": self.optim_config.learning_rate / 1e3,
        #                     "weight_decay": self.optim_config.weight_decay * 1e3,
        #                 }
        #             )

        # # Find the non-special parameters by subtracting special_params from all model parameters
        # all_params = set(self.model.parameters())
        # non_special_params = all_params - special_params

        # # Add the non-special parameters to the first parameter group
        # params.insert(0, {"params": list(non_special_params)})

        params = self.model.parameters()

        optimizer = optim.AdamW(
            params,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta_1, self.optim_config.beta_2),
            eps=self.optim_config.eps,
            weight_decay=self.optim_config.weight_decay,
        )
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.optim_config.warmup,
            max_iters=self.optim_config.max_iters,
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
