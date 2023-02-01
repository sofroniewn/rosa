from glob import glob

import anndata as ad
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer

from rosa import RosaConfig, RosaDataModule, RosaLightningModule

cs = ConfigStore.instance()
cs.store(name="rosa_config", node=RosaConfig)


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
