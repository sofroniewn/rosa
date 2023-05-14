import os
from typing import Tuple

import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from ..data import RosaDataModule, create_io_paths
from ..utils import RosaConfig, score_predictions
from ..utils.helpers import sample, reconstruct_from_results
from .modules import RosaLightningModule


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


def predict(config: RosaConfig, chkpt: str) -> ad.AnnData:
    _, output_path = create_io_paths(config.paths)

    rdm = RosaDataModule(
        output_path,
        config=config.data_module,
    )
    rdm.setup()

    adata = rdm.val_dataset.adata
    obs_indices = rdm.val_dataset.obs_indices.detach().numpy()
    var_bool = rdm.val_dataset.mask_bool.detach().numpy()
    adata_predict = adata[obs_indices, var_bool]
    # counts = rdm.train_dataset.counts.mean(dim=1)
    # counts = torch.bincount(rdm.train_dataset.expression.ravel(), minlength=rdm.train_dataset.n_bins)

    output_dir = str(rdm.adata_path.with_name(rdm.adata_path.stem + "__processed"))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    rlm = RosaLightningModule.load_from_checkpoint(
        chkpt,
        var_input=rdm.var_input,
        config=config.module,
        adata=adata_predict,
        weight=None,  # 1 / counts,
    )
    print(rlm)
    nbins = rlm.n_bins

    if config.trainer.num_devices > 1:
        strategy = "ddp"
    else:
        strategy = None

    pred_writer = CustomWriter(output_dir=output_dir, write_interval="epoch")
    trainer = Trainer(
        accelerator=config.trainer.device,
        devices=config.trainer.num_devices,
        strategy=strategy,
        precision=config.trainer.precision,
        callbacks=[pred_writer],
        deterministic=False,
    )
    trainer.predict(rlm, rdm, return_predictions=False)

    print("Generating predictions done")
    if rlm.global_rank == 0:
        print("Making predicted adata object")
        results = []
        for global_rank in range(config.trainer.num_devices):
            results.append(
                torch.load(os.path.join(output_dir, f"predictions_{global_rank}.pt"))[0]
            )
        results = [item for sublist in results for item in sublist]
        target, predicted, confidence, obs_idx = reconstruct_from_results(
            results, nbins
        )

        print("Scoring predictions")
        scores = score_predictions(predicted, target, nbins=nbins)
        for key in scores.keys():
            scores[key] = scores[key].detach().numpy()

        print(
            f"""
            mean spearman across genes {scores['spearman_obs_mean']:.3f}
            mean spearman across cells {scores['spearman_var_mean']:.3f}
            """
        )

        print("Assembling anndata object")
        obs_indices = obs_idx.detach().numpy()
        adata = rdm.predict_dataset.adata
        var_bool = rdm.predict_dataset.mask_bool.detach().numpy()
        adata_predict = adata[obs_indices, var_bool]
        adata_predict.layers["confidence"] = confidence.detach().numpy()
        adata_predict.layers["target"] = target.detach().numpy()
        adata_predict.layers["predicted"] = predicted.detach().numpy()
        adata_predict.uns["results"] = scores
        adata_predict.uns["nbins"] = nbins

        print("Calculate dendrogram")
        sc.tl.dendrogram(adata_predict, groupby="label")

        print("Saving predicted adata")
        output_path = str(
            rdm.adata_path.with_name(rdm.adata_path.stem + "__processed.h5ad")
        )
        print(output_path)
        adata_predict.write_h5ad(output_path)
        print("Finished")
