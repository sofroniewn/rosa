import anndata as ad
from scvi.model import SCVI
from pathlib import Path
import pandas as pd
from rosa.data.preprocessing import add_dendrogram_and_hvgs, add_de_marker_genes


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_pbulk.h5ad"

scvi_model_pt = Path(ADATA_PT[:-5] + "_scvi_model/")
scvi_de = Path(ADATA_PT[:-5] + "_scvi_de.csv")

adata = ad.read_h5ad(ADATA_PT)

train_cells = adata.obs["train"]
adata_train = adata[train_cells].copy()
SCVI.setup_anndata(adata_train, layer="counts")
model = SCVI(
    adata_train,
    n_layers=2,
    n_hidden=256,
    n_latent=32,
    dispersion="gene",
    gene_likelihood="zinb",
)

# Train model
if scvi_model_pt.exists():
    print(f'Loading Model')
    model.load(scvi_model_pt, adata=adata)
else:
    print(f'Training Model')
    model.train()
    model.save(scvi_model_pt, overwrite=True)

# Compute differential expression
if scvi_de.exists():
    print(f'Loading Differential Expression')
    df = pd.read_csv(scvi_de, index_col=0)
else:
    print(f'Computing Differential Expression')
    df = model.differential_expression(groupby='label')
    df.to_csv(scvi_de)

# Add dendrogram, hvgs, and marker genes
print(f'Adding marker genes')
adata = add_dendrogram_and_hvgs(adata)
adata = add_de_marker_genes(adata, differential_expression=df)
adata.write_h5ad(ADATA_PT)