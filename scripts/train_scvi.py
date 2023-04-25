import anndata as ad
from scvi.model import SCVI
from pathlib import Path
import pandas as pd


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens.h5ad"
LABEL = "cell_type"

scvi_model_pt = Path(ADATA_PT[:-5] + "_scvi_model/")
scvi_de = Path(ADATA_PT[:-5] + "_scvi_de.csv")

adata = ad.read_h5ad(ADATA_PT)

SCVI.setup_anndata(adata)
model = SCVI(
    adata,
    n_layers=2,
    n_hidden=256,
    n_latent=32,
    dispersion="gene",
    gene_likelihood="zinb",
)

# Train model
if scvi_model_pt.exists():
    print(f"Loading Model")
    model.load(scvi_model_pt, adata=adata)
else:
    print(f"Training Model")
    model.train()
    model.save(scvi_model_pt, overwrite=True)

# Compute differential expression
if scvi_de.exists():
    print(f"Loading Differential Expression")
    df = pd.read_csv(scvi_de, index_col=0)
else:
    print(f"Computing Differential Expression")
    df = model.differential_expression(groupby=LABEL)
    df.to_csv(scvi_de)

print("Done")
