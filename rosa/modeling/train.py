from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..data import RosaDataModule, create_io_paths
from ..utils import RosaConfig
from .modules import RosaLightningModule


def train(config: RosaConfig) -> None:
    _, output_path = create_io_paths(config.paths)

    rdm = RosaDataModule(
        output_path,
        config=config.data_module,
    )
    rdm.setup()

    rlm = RosaLightningModule(
        in_dim=rdm.len_input,
        out_dim=rdm.len_target,
        config=config.module,
        var_input=rdm.predict_dataset.input[1]
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
        accelerator=config.device,
        devices=1,
        # strategy='ddp',
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=config.data_module.accumulate,
        gradient_clip_val=10,
    )
    trainer.fit(rlm, rdm)
