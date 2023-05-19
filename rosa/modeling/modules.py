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
from ..utils.helpers import sample, CosineWarmupScheduler, reconstruct_from_results
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

        self.keep_obs = None
        self.keep_var = None

        # self.adata = adata
        # self.target = None
        # self.marker_genes_dict = self.adata.obs.set_index("label").to_dict()[
        #     "marker_feature_name"
        # ]
        # sc.tl.dendrogram(self.adata, groupby="label", use_rep="X")

    def forward(self, batch):
        return self.model.forward(batch)

    def _basic_step(self, batch, batch_idx, dataloader_idx=0):
        batch["var_input_encoder"] = self.var_input[0, batch["var_indices_encoder"]]
        batch["var_input_decoder"] = self.var_input[0, batch["var_indices_decoder"]]
        expression_predicted = self(batch)

        results = {}
        results["expression_predicted"] = expression_predicted
        results["expression_target"] = batch["expression_decoder"]
        results["batch_idx"] = batch_idx
        results["dataloader_idx"] = dataloader_idx
        results["obs_idx"] = batch["obs_idx"]
        return results

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        results = self._basic_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        expression_logits = results["expression_predicted"]
        target = results["expression_target"]
        loss = self.criterion(
            expression_logits.view(-1, self.n_bins),
            target.view(-1),
        )

        # predicted, _ = sample(expression_logits, nbins=self.n_bins)
        # if predicted.shape[0] == 1:
        #     spearman_obs_loss = 1 - spearman_corrcoef(predicted[0].float(), target[0].float())
        #     # spearman_var_loss = 0
        # else:
        #     spearman_obs_loss = 1 - spearman_corrcoef(predicted.T.float(), target.T.float()).mean()
        #     # spearman_var_loss = 1 - spearman_corrcoef(predicted.float(), target.float()).mean()
        # beta_obs_loss = 1.0
        # # beta_var_loss = 1.0

        # self.log(
        #     "train_ce_loss", ce_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        # )
        # self.log(
        #     "train_obs_loss", spearman_obs_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        # )
        # # self.log(
        # #     "train_var_loss", spearman_var_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # # )
        # loss = ce_loss + beta_obs_loss * spearman_obs_loss # + beta_var_loss * spearman_var_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        results = self._basic_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        loss = self.criterion(
            results["expression_predicted"].view(-1, self.n_bins),
            results["expression_target"].view(-1),
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return results

    def validation_epoch_end(self, results):
        results = self.all_gather(results)

        if self.global_rank != 0:
            return

        target, predicted, _, _ = reconstruct_from_results(results, self.n_bins)

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

        if self.global_step > 0:
            if self.keep_obs is None:
                size = min(64, target.shape[0], target.shape[1])
                self.keep_obs = torch.randperm(target.shape[0])[:size]
                self.keep_var = torch.randperm(target.shape[1])[:size]
            target = target[self.keep_obs][:, self.keep_var]
            predicted = predicted[self.keep_obs][:, self.keep_var]

            blended = merge_images(target, predicted)
            blended = blended / self.n_bins
            tensorboard_logger.add_image(
                "cellXgene_blended", blended, self.global_step, dataformats="HWC"
            )

    #     if self.global_step > 0 and self.target is None:
    #         # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
    #         self.adata.layers["target"] = target.detach().cpu().numpy()
    #         # self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
    #         mp = MatrixPlot(
    #             self.adata,
    #             self.marker_genes_dict,
    #             groupby="label",
    #             gene_symbols="feature_name",
    #             layer="target",
    #             vmin=0,
    #             vmax=self.n_bins - 1,
    #             show=False,
    #             dendrogram=False,
    #         )
    #         mp.add_dendrogram(dendrogram_key=True)

    #         _color_df = mp.values_df.copy()
    #         if mp.var_names_idx_order is not None:
    #             _color_df = _color_df.iloc[:, mp.var_names_idx_order]

    #         if mp.categories_order is not None:
    #             _color_df = _color_df.loc[mp.categories_order, :]
    #         self.target = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
    #         tensorboard_logger.add_image(
    #             "cellXgene_1_target", self.target, self.global_step, dataformats="HW"
    #         )

    #     if self.global_step > 0:
    #         # self.adata.layers["confidence"] = confidence.detach().cpu().numpy()
    #         # self.adata.layers["target"] = target.detach().cpu().numpy()
    #         self.adata.layers["predicted"] = predicted.detach().cpu().numpy()
    #         mp = MatrixPlot(
    #             self.adata,
    #             self.marker_genes_dict,
    #             groupby="label",
    #             gene_symbols="feature_name",
    #             layer="predicted",
    #             vmin=0,
    #             vmax=self.n_bins - 1,
    #             show=False,
    #             dendrogram=False,
    #         )
    #         mp.add_dendrogram(dendrogram_key=True)

    #         _color_df = mp.values_df.copy()
    #         if mp.var_names_idx_order is not None:
    #             _color_df = _color_df.iloc[:, mp.var_names_idx_order]

    #         if mp.categories_order is not None:
    #             _color_df = _color_df.loc[mp.categories_order, :]

    #         predicted = torch.from_numpy(_color_df.values) / (self.n_bins - 1)
    #         tensorboard_logger.add_image(
    #             "cellXgene_2_predicted", predicted, self.global_step, dataformats="HW"
    #         )

    #         if self.target is not None:
    #             blended = merge_images(self.target, predicted)
    #             tensorboard_logger.add_image(
    #                 "cellXgene_3_blended", blended, self.global_step, dataformats="HWC"
    #             )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        results = self._basic_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        return results

    def configure_optimizers(self):
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
