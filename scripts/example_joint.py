from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from rosa import JointAnnDataModule, JointEmbedding2ExpressionModel


BASE_PT = '/Users/nsofroniew/Documents/data/multiomics/enformer'
TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT = '/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_prot.h5ad'
# CKPT_PT = '/Users/nsofroniew/Documents/data/multiomics/enformer/embedding2expression_r2/lightning_logs/version_7/checkpoints/epoch=11958-step=7211277.ckpt'
CKPT_PT = None
HEAD_1 = 'MLP'
HEAD_2 = 'MLP'
METHOD = 'bilinear'
RANK = 16

dm = JointAnnDataModule(TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT)
dm.setup()

model = JointEmbedding2ExpressionModel(in_dim_1=dm.n_input_1, in_dim_2=dm.n_input_2, rank=RANK, head_1=HEAD_1, head_2=HEAD_2, method=METHOD)
print(model)

checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor='val_loss', mode='min', save_last=True)
trainer = Trainer(max_epochs=500_000,
        check_val_every_n_epoch=1,
        # log_every_n_steps=10_000,
        default_root_dir=BASE_PT + f'/Embedding2ExpressionModel',
        resume_from_checkpoint=CKPT_PT,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=10,
    )
trainer.fit(model, dm)
# tensorboard --logdir=lightning_logs/