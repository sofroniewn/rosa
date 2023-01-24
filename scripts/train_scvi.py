import anndata as ad
from scvi.model import SCVI
import torch.nn as nn


TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_new_norm.h5ad"
TABULA_SAPIENS_BY_CELL_TYPE_SCVI_MODEL = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_scvi_model_new_norm_lsm/"

adata = ad.read_h5ad(TABULA_SAPIENS_BY_CELL_TYPE_WITH_EMBEDS_PT)

train_cells = adata.obs["train"]
adata_train = adata[train_cells].copy()

SCVI.setup_anndata(adata_train, layer="counts")

# model = SCVI(adata_train, n_layers=4, n_hidden=512, n_latent=64)
# model = SCVI(adata_train, n_layers=2, n_hidden=256, n_latent=32, dispersion='gene-cell')
model = SCVI(
    adata_train,
    n_layers=2,
    n_hidden=256,
    n_latent=32,
    dispersion="gene",
    gene_likelihood="zinb",
)

import torch
import torch.nn as nn

class Log1pSoftmax(nn.Module):
    def __init__(self, dim=None):
        super(Log1pSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        
    def forward(self, input):
        return torch.log1p(self.softmax(input))

# model = SCVI(adata_train, n_layers=2, n_hidden=64, n_latent=16)
# model = SCVI(adata_train)
# model.module.decoder.px_scale_decoder[1] = nn.Softplus()
# model.module.decoder.px_scale_decoder[1] = nn.Softmax(dim=-1)
model.module.decoder.px_scale_decoder[1] = Log1pSoftmax(dim=-1)

model.train()

model.save(TABULA_SAPIENS_BY_CELL_TYPE_SCVI_MODEL, overwrite=True)
