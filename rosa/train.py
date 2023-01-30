import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rosa import RosaConfig, RosaDataModule, RosaLightningModule

cs = ConfigStore.instance()
cs.store(name="rosa_config", node=RosaConfig)


def train(cfg: RosaConfig) -> None:
    rdm = RosaDataModule(
        cfg.paths.adata,
        data_config=cfg.data,
        param_config=cfg.params,
    )
    rdm.setup()

    rlm = RosaLightningModule(
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        model_config=cfg.model,
        learning_rate=cfg.params.learning_rate,
    )
    print(rlm)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True
    )

    trainer = Trainer(
        max_epochs=500_000,
        check_val_every_n_epoch=1,
        # log_every_n_steps=10_000,
        logger=TensorBoardLogger(".", "", ""),
        resume_from_checkpoint=cfg.paths.chkpt,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=10,
    )
    trainer.fit(rlm, rdm)
