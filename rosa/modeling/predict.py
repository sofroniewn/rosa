import anndata as ad
from pytorch_lightning import Trainer

from ..data import RosaDataModule
from ..utils import RosaConfig
from .modules import RosaLightningModule


def predict(config: RosaConfig, chkpt: str) -> ad.AnnData:
    # Create Data Module
    rdm = RosaDataModule(
        config.paths.adata,
        config=config.data_module,
    )
    rdm.setup()

    # Load model from checkpoint
    rlm = RosaLightningModule.load_from_checkpoint(
        chkpt,
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        config=config.module,
    )
    print(rlm)

    trainer = Trainer()
    predictions = trainer.predict(rlm, rdm)
    rdm.predict_dataset.predict(predictions)
    return rdm.predict_dataset.adata
