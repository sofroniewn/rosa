import cell_census
from pathlib import Path


BASE = "/home/ec2-user/cell_census"  # "/Users/nsofroniew/Documents/data/multiomics/cell_census"
DATASET = "tabula_sapiens"

path = Path(BASE) / (DATASET + ".h5ad")

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
print("Save adata")
adata.write(path)
