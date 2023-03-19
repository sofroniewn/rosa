from typing import Tuple
import anndata as ad
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import os
import torch.nn as nn

from ..data import RosaDataModule, create_io_paths
from ..utils import RosaConfig, score_predictions
from .modules import RosaLightningModule


def sample(x: torch.Tensor, nbins: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    if nbins > 1:
        confidence, prediction = nn.functional.softmax(x, dim=-1).max(dim=-1)
        return prediction, confidence
    return x, torch.ones_like(x)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def predict(config: RosaConfig, chkpt: str) -> ad.AnnData:
    _, output_path = create_io_paths(config.paths)

    # Create Data Module
    rdm = RosaDataModule(
        output_path,
        config=config.data_module,
    )
    rdm.setup()

    # Load model from checkpoint
    rlm = RosaLightningModule.load_from_checkpoint(
        chkpt,
        var_input=rdm.var_input,
        config=config.module,
    )
    print(rlm)

    if config.num_devices > 1:
        strategy = "ddp"
    else:
        strategy = None

    output_dir = str(rdm.adata_path.with_name(rdm.adata_path.stem + '__processed'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pred_writer = CustomWriter(output_dir=output_dir, write_interval="epoch")
    trainer = Trainer(accelerator=config.device, devices=config.num_devices, strategy=strategy, callbacks=[pred_writer])
    trainer.predict(rlm, rdm, return_predictions=False)

    print('Generating predictions done')
    if rlm.global_rank == 0:
        print('Making predicted adata object')
        results = []
        for global_rank in range(config.num_devices):
            results.append(torch.load(os.path.join(output_dir, f"predictions_{global_rank}.pt")))

        predicted = []
        confidence = []
        target = []
        obs_idx = []
        nbins = config.data_module.data.expression_transform.n_bins
        for device in results:
            for batch in device[0]:
                p, c = sample(batch['expression_predicted'], nbins=nbins)
                predicted.append(p)
                confidence.append(c)
                target.append(batch['expression_target'])
                obs_idx.append(batch['obs_idx'])

        obs_idx = torch.concat(obs_idx)
        order = torch.argsort(obs_idx)

        confidence = torch.concat(confidence)[order]
        target = torch.concat(target)[order]
        predicted = torch.concat(predicted)[order]

        print('Scoring predictions')
        results = score_predictions(predicted, target, nbins=nbins)

        print('Assembling anndata object')
        adata = rdm.val_dataset.adata
        obs_indices = rdm.val_dataset.obs_indices.detach().numpy()
        var_bool = rdm.val_dataset.mask_bool.detach().numpy()
        adata_predict = adata[obs_indices, var_bool]
        adata_predict.layers["confidence"] = confidence.detach().numpy()
        adata_predict.layers["target"] = target.detach().numpy()
        adata_predict.layers["predicted"] = predicted.detach().numpy()
        adata_predict.uns['results'] = results
        adata_predict.uns['nbins'] = nbins

        print('Saving predicted adata')
        output_path = str(rdm.adata_path.with_name(rdm.adata_path.stem + '__processed.h5ad'))
        print(output_path)
        adata_predict.write_h5ad(output_path)
        print('Finished')
