import cellxgene_census
from pathlib import Path


BASE = "/home/ec2-user/cell_census"  # "/Users/nsofroniew/Documents/data/multiomics/cell_census"
DATASET = "tabula_sapiens_2"

# dataset_id = "48b37086-25f7-4ecd-be66-f5bb378e3aea" # tabula muris all
# organism = "mus_musculus"

dataset_id = "e5f58829-1a66-40b5-a624-9046778e74f5" # tabula sapiens all
organism = "homo_sapiens"

path = Path(BASE) / (DATASET + ".h5ad")

# Open cell census
census = cellxgene_census.open_soma()

# Fetch desired data and put into an anndata object
adata = cellxgene_census.get_anndata(
    census=census,
    organism=organism,
    obs_value_filter=f"dataset_id == '{dataset_id}'",
)
print(adata)

# Save anndata object
print("Save adata")
adata.write(path)
