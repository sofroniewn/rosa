import anndata as ad
from pytorch_lightning import Trainer

from ..data import RosaDataModule, create_io_paths
from ..utils import RosaConfig
from .modules import RosaLightningModule


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
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        config=config.module,
        var_input=rdm.predict_dataset.input[1],
    )
    print(rlm)

    trainer = Trainer(accelerator=config.device, devices=1, strategy=None)
    outputs = trainer.predict(rlm, rdm)
    predictions, measured = zip(*outputs)  # type: ignore
    sampled_predictions = [rlm.model.sample(y_hat) for y_hat in predictions]
    predicted, confidence = zip(*sampled_predictions)
    rdm.predict_dataset.predict(confidence, "confidence")
    rdm.predict_dataset.predict(predicted, "predicted")
    rdm.predict_dataset.predict(measured, "measured")
    return rdm.predict_dataset.adata, rdm, rlm
