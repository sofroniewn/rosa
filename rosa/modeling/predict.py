import anndata as ad
from pytorch_lightning import Trainer
import torch

from ..data import RosaDataModule, create_io_paths
from ..utils import RosaConfig
from .modules import RosaLightningModule


from pytorch_lightning.callbacks import BasePredictionWriter
import os


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

    output_dir = str(rdm.adata_path.with_name(rdm.adata_path.stem + '__preprocessed'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pred_writer = CustomWriter(output_dir=output_dir, write_interval="epoch")
    trainer = Trainer(accelerator=config.device, devices=config.num_devices, strategy=strategy, callbacks=[pred_writer])
    trainer.predict(rlm, rdm, return_predictions=False)


    # predicted_bins = []
    # measured = []
    # for results in outputs:
    #     predicted_bins.append(results['expression_predicted'])
    #     measured.append(results['expression'])
    # predicted, confidence = zip(*[rlm.model.sample(y_hat) for y_hat in predicted_bins])
    # confidence = torch.concat(confidence).detach_().numpy()
    # measured = torch.concat(measured).detach_().numpy()
    # predicted = torch.concat(predicted).detach_().numpy()

    # obs_indices = rdm.val_dataset.obs_indices.detach_().numpy()
    # var_bool = rdm.val_dataset.mask.detach_().numpy()

    # adata = rdm.val_dataset.adata
    # adata_predict = adata[obs_indices, var_bool]
    # adata_predict.layers["confidence"] = confidence
    # adata_predict.layers["measured"] = measured
    # adata_predict.layers["predicted"] = predicted

    # return adata_predict, rdm, rlm
