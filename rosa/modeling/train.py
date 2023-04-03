import torch

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
        weight= None, #1 / counts,
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
        max_epochs=10_000,
        check_val_every_n_epoch=5,
        logger=TensorBoardLogger(".", "", ""),
        resume_from_checkpoint=config.paths.chkpt,
        accelerator=config.device,
        devices=config.num_devices,
        strategy=strategy,
        precision=config.precision,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=config.data_module.accumulate,
        gradient_clip_val=config.gradient_clip_val,
        deterministic=True
    )
    trainer.fit(rlm, rdm)
