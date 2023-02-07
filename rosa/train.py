from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .config import RosaConfig
from .datamodules import RosaDataModule
from .modules import RosaLightningModule


def train(config: RosaConfig) -> None:
    rdm = RosaDataModule(
        config.paths.adata,
        config=config.data_module,
    )
    rdm.setup()

    rlm = RosaLightningModule(
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        config=config.module,
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
        resume_from_checkpoint=config.paths.chkpt,
        accelerator="mps",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=10,
    )
    trainer.fit(rlm, rdm)
