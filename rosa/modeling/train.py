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
        var_input=rdm.var_input,
        config=config.module,
    )
    print(rlm)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True
    )

    if config.num_devices > 1:
        strategy = "ddp"
    else:
        strategy = None

    trainer = Trainer(
        max_epochs=500_000,
        check_val_every_n_epoch=1,
        # log_every_n_steps=10_000,
        logger=TensorBoardLogger(".", "", ""),
        resume_from_checkpoint=config.paths.chkpt,
        accelerator=config.device,
        devices=config.num_devices,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=config.data_module.accumulate,
        gradient_clip_val=config.gradient_clip_val,
    )
    trainer.fit(rlm, rdm)
