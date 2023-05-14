import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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

    adata = rdm.val_dataset.adata
    obs_indices = rdm.val_dataset.obs_indices.detach().numpy()
    var_bool = rdm.val_dataset.mask_bool.detach().numpy()
    adata_predict = adata[obs_indices, var_bool]
    # counts = rdm.train_dataset.counts.mean(dim=1)
    # counts = torch.bincount(rdm.train_dataset.expression.ravel(), minlength=rdm.train_dataset.n_bins)

    rlm = RosaLightningModule(
        var_input=rdm.var_input,
        config=config.module,
        adata=adata_predict,
        weight=None,  # 1 / counts,
    )
    print(rlm)
    print(
        f"Train samples {len(rdm.train_dataset)}, Val samples {len(rdm.val_dataset)}, {adata.shape[1]} genes"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    if config.trainer.num_devices > 1:
        strategy = "ddp"
    else:
        strategy = None

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        check_val_every_n_epoch=None,  # config.trainer.check_val_every_n_epoch,
        val_check_interval=config.trainer.val_check_interval,  # 1000,
        limit_val_batches=config.trainer.limit_val_batches,  # 20,
        log_every_n_steps=config.trainer.log_every_n_steps,  # 50,
        logger=TensorBoardLogger(".", "", ""),
        resume_from_checkpoint=config.paths.chkpt,
        accelerator=config.trainer.device,
        devices=config.trainer.num_devices,
        strategy=strategy,
        precision=config.trainer.precision,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        accumulate_grad_batches=config.data_module.accumulate,
        gradient_clip_val=config.trainer.gradient_clip_val,
        deterministic=False,
    )
    trainer.fit(rlm, rdm)
