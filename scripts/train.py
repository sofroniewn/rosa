import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from rosa import AnnDataModule, SingleEmbedding2ExpressionModel
from rosa.config import RosaConfig


@hydra.main(config_path="conf", config_name="config")
def train(cfg: RosaConfig) -> None:
    dm = AnnDataModule(
        cfg.paths.adata,
        expression_layer=cfg.adata.expression_layer,
        obs_embedding=cfg.adata.obs_embedding,
        var_embedding=cfg.adata.var_embedding,
        batch_size=cfg.params.batch_size,
    )
    dm.setup()

    model = SingleEmbedding2ExpressionModel(
        in_dim=dm.len_var_embedding, out_dim=dm.n_obs, head=cfg.model.head
    )
    print(model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True
    )
    trainer = Trainer(
        max_epochs=500_000,
        check_val_every_n_epoch=1,
        # log_every_n_steps=10_000,
        default_root_dir=cfg.paths.base + f"/Embedding2ExpressionModel_norm",
        resume_from_checkpoint=cfg.paths.chkpt,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=10,
    )
    trainer.fit(model, dm)
    # tensorboard --logdir=lightning_logs/


if __name__ == "__main__":
    train()
