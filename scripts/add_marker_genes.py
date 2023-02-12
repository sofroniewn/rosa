import anndata as ad
from scvi.model import SCVI
from pathlib import Path
import pandas as pd
from rosa.data.preprocessing import add_dendrogram_and_hvgs, add_de_marker_genes


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_cell_type.h5ad"

scvi_de = Path(ADATA_PT[:-5] + "_scvi_de.csv")

adata = ad.read_h5ad(ADATA_PT)

# Load differential expression
print(f'Loading Differential Expression')
df = pd.read_csv(scvi_de, index_col=0)

# Add dendrogram, hvgs, and marker genes
print(f'Adding marker genes')
adata = add_dendrogram_and_hvgs(adata)
adata = add_de_marker_genes(adata, differential_expression=df)

adata.write_h5ad(ADATA_PT[:-5] + "_2.h5ad")