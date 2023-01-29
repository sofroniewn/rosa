from pytorch_lightning import Trainer
from glob import glob
import anndata as ad
from hydra.core.config_store import ConfigStore
from rosa import RosaDataModule, RosaLightningModule, RosaConfig


cs = ConfigStore.instance()
cs.store(name="rosa_config", node=RosaConfig)


def predict(cfg: RosaConfig, chkpt: str) -> ad.AnnData:
    # Create Data Module
    rdm = RosaDataModule(
        cfg.paths.adata,
        data_config=cfg.data,
        param_config=cfg.params,
    )
    rdm.setup()

    # Load model from checkpoint
    rlm = RosaLightningModule.load_from_checkpoint(
        chkpt,
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        model_cfg=cfg.model,
        learning_rate=cfg.params.learning_rate,
    )
    print(rlm)

    trainer = Trainer()
    predictions = trainer.predict(rlm, rdm)
    rdm.predict_dataset.predict(predictions)
    return rdm.predict_dataset.adata
