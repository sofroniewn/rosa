import anndata as ad
import decoupler as dc
from rosa.preprocessing import average_expression_per_feature


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens.h5ad"
ADATA_BULK_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features.h5ad"
METHOD = "decoupler"
SAMPLE_COL = "donor_id"  # Or use 'dataset_id' to bulk across datasets

adata = ad.read_csv(ADATA_PT)
print(adata)

# Bulk anndata
if METHOD == "decoupler":
    # Note that genes with no samples will be dropped
    padata = dc.get_pseudobulk(
        adata,
        sample_col=SAMPLE_COL,
        groups_col="cell_type",
        layer=None,
        min_prop=0,
        min_cells=0,
        min_counts=0,
        min_smpls=0,
    )
    padata.var = adata.var.loc[padata.var.index]
else:
    padata = average_expression_per_feature(adata, "cell_type")

print(padata)

# Save anndata object
padata.write(ADATA_BULK_PT)
