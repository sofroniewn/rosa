import cell_census
from rosa.preprocessing import average_expression_per_feature


ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens.h5ad"

# Open cell census
census = cell_census.open_soma()

# Fetch desired data and put into an anndata object
adata = cell_census.get_anndata(
    census,
    "Homo sapiens",
    obs_query={
        "dataset_id": "e5f58829-1a66-40b5-a624-9046778e74f5",  # tabula sapiens all
    },
)
print(adata)

# Save anndata object
adata.write(ADATA_PT)
