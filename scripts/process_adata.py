import anndata as ad
import numpy as np

ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_prot.h5ad"

# Load anndata file
adata = ad.read_h5ad(ADATA_PT)

# Set train indicator
np.random.seed(42)
train_fraction = 0.7
adata.var["train"] = np.random.rand(adata.n_vars) <= train_fraction
adata.obs["train"] = np.random.rand(adata.n_obs) <= train_fraction

# Write anndata file
adata.write_h5ad(ADATA_PT)
