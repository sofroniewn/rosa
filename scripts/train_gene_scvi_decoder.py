from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from rosa import GeneAnnDataModule
from rosa.models import SingleSCVIDecoderEmbedding2ExpressionModel


BASE_PT = "/Users/nsofroniew/Documents/data/multiomics/enformer"
TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_new_norm.h5ad"
# CKPT_PT = '/Users/nsofroniew/Documents/data/multiomics/enformer/embedding2expression_r2/lightning_logs/version_7/checkpoints/epoch=11958-step=7211277.ckpt'
CKPT_PT = None

dm = GeneAnnDataModule(TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT)
dm.setup()

model = SingleSCVIDecoderEmbedding2ExpressionModel(
    in_dim=dm.n_input, out_dim=dm.n_output
)
print(model)

checkpoint_callback = ModelCheckpoint(
    save_top_k=2, monitor="val_loss", mode="min", save_last=True
)
trainer = Trainer(
    max_epochs=500_000,
    check_val_every_n_epoch=1,
    # log_every_n_steps=10_000,
    default_root_dir=BASE_PT + f"/Embedding2ExpressionModel_norm",
    resume_from_checkpoint=CKPT_PT,
    accelerator="cpu",
    devices=1,
    callbacks=[checkpoint_callback],
    gradient_clip_val=10,
)
trainer.fit(model, dm)
# tensorboard --logdir=lightning_logs/
