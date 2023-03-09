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
        var_input=rdm.predict_dataset.input[1]
    )
    print(rlm)

    if config.num_devices > 1:
        strategy = 'ddp'
    else:
        strategy = None

    trainer = Trainer(accelerator=config.device, devices=config.num_devices, strategy=strategy)
    predictions = trainer.predict(rlm, rdm)
    rdm.predict_dataset.predict(predictions)
    return rdm.predict_dataset.adata
