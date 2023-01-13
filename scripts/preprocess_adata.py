import polars as pl
import zarr
import numpy as np
import anndata as ad
from rosa.preprocessing import (
    add_gene_biotype,
    add_gene_embeddings,
    add_train_indicators,
    clean_cells_genes,
    calculate_cell_embeddings_pca,
    normalize_expression,
)


BASE_PT = "/Users/nsofroniew/Documents/data/multiomics/enformer"

GENE_INTERVALS_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.bed"
REDUCED_EMBEDDING_PT = (
    BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_reduced_embedding.zarr"
)
TSS_EMBEDDING_PT = BASE_PT + "/Homo_sapiens.GRCh38.genes.enformer_tss_embedding.zarr"
RAW_ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features.h5ad"
EMBEDS_ADATA_PT = "/Users/nsofroniew/Documents/data/multiomics/cell_census/tabula_sapiens_by_features_with_embeds_new.h5ad"

# Read gene intervals
seqs = pl.read_csv(GENE_INTERVALS_PT, sep="\t", has_header=False).to_pandas()
# Set index to ensmbl id which is in column 5
seqs = seqs.set_index("column_5")

# Load gene embeddings
embeds = np.asarray(zarr.open(TSS_EMBEDDING_PT))

# Load anndata raw anndata file
adata = ad.read_h5ad(RAW_ADATA_PT)
# Set index to be feature id which is ensmbl id
# adata.var.set_index("feature_id")

# # Add gene biotypes
# adata = add_gene_biotype(adata)
# Add embeddings
adata = add_gene_embeddings(adata, seqs, embeds)
# Add train indicators
adata = add_train_indicators(adata, fraction=0.7, seed=42)
# Clean cells and genes
adata = clean_cells_genes(adata)
# Normalize expression
adata = normalize_expression(adata)
# Calculate and add cell embeddings using only traning data
adata = calculate_cell_embeddings_pca(adata)

# Write preprocessed anndata file
print("Saving anndata file")
adata.write_h5ad(EMBEDS_ADATA_PT)
