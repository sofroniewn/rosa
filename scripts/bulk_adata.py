import anndata as ad
import decoupler as dc
from rosa.preprocessing import average_expression_per_feature


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens.h5ad"
ADATA_BULK_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_pbulk.h5ad"
METHOD = "decoupler"
SAMPLE_COL = "donor_id"  # Or use 'dataset_id' to bulk across datasets
LABEL_COL = "cell_type"

adata = ad.read_h5ad(ADATA_PT)
print(adata)

# Bulk anndata
if METHOD == "decoupler":
    # Note that genes with no samples will be dropped
    padata = dc.get_pseudobulk(
        adata,
        sample_col=SAMPLE_COL,
        groups_col=LABEL_COL,
        layer=None,
        min_prop=0,
        min_cells=0,
        min_counts=0,
        min_smpls=0,
    )
    padata.var = adata.var.loc[padata.var.index]
    padata.obs['is_primary_data'] = padata.obs['is_primary_data'].astype(str)
else:
    padata = average_expression_per_feature(adata, LABEL_COL)

print(padata)

# Save anndata object
padata.write_h5ad(ADATA_BULK_PT)
