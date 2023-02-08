from typing import Dict, Optional
from pathlib import Path

import polars as pl
import zarr
import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from tqdm import tqdm
from omegaconf import OmegaConf


from ..utils.config import BulkDataConfig, PathConfig, PreProcessingConfig, RosaConfig, EmbeddingPathsConfig


def _add_gene_embeddings(
    adata: ad.AnnData, seqs: pd.DataFrame, embeds: np.ndarray
) -> ad.AnnData:
    """Add gene embeddings to an anndata object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with genes indexed by ENSMBL id.
    seqs: DataFrame
        Information about gene embeddings indexed by ENSMBL id.
    embeds: Array
        Gene embeddings. Must be same length as seqs.

    Returns
    -------
    adata : AnnData
        AnnData object with gene embedding for each gene. Genes in
        the original AnnData object without an embedding were dropped,
        and genes with an embedding that weren't in the original AnnData
        object have been ignored.
    """
    # Set index
    if "feature_id" in adata.var.columns:
        adata.var.set_index("feature_id", inplace=True)

    print(f"{len(seqs)} genes for embedding")
    print(f"{embeds.shape[0]} gene embeddings loaded")

    assert embeds.shape[0] == len(seqs)

    print(f"{embeds.shape[1]} length for each gene embedding")
    print(f"Expression data for {adata.n_vars} genes")

    # Identify matching genes
    matching_genes = list(set(adata.var.index) & set(seqs.index))
    print(f"Matches for {len(matching_genes)} genes")

    # Create new anndata object with matching genes only
    adata = adata[:, adata.var.index.isin(seqs.index)]

    # Identify reordering between embeddings and expression genes
    reorder_indices = [seqs.index.get_loc(ind) for ind in adata.var.index]

    # Add correctly ordered embedding to varm
    adata.varm["embedding"] = embeds[reorder_indices, :]

    # Add embedding metadata to var
    seqs = seqs[seqs.index.isin(adata.var.index)]
    adata.var = pd.concat([adata.var, seqs], axis=1)

    return adata


def add_gene_embeddings(adata: ad.AnnData, config: EmbeddingPathsConfig) -> ad.AnnData:
    # Read gene intervals
    seqs = pl.read_csv(config.gene_intervals, sep="\t", has_header=False).to_pandas()
    # Set index to ensmbl id which is in column 5
    seqs = seqs.set_index("column_5")

    # Load gene embeddings
    embeds = np.asarray(zarr.open(config.embeddings))

    adata = _add_gene_embeddings(adata, seqs, embeds)
    return adata


def add_train_indicators(
    adata: ad.AnnData,
    fraction: float = 0.7,
    seed: Optional[int] = None,
    train_key: str = "train",
) -> ad.AnnData:
    # Set train indicator
    if seed is None:
        np.random.seed(42)
    else:
        np.random.seed(seed)
    adata.var[train_key] = np.random.rand(adata.n_vars) <= fraction
    adata.obs[train_key] = np.random.rand(adata.n_obs) <= fraction
    return adata


def add_gene_biotype(adata: ad.AnnData) -> ad.AnnData:
    print("Adding gene biotypes")
    # Retrieve gene symbols
    annot = sc.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", "external_gene_name", "gene_biotype"],
        use_cache=True,
    ).set_index("ensembl_gene_id")

    print(f"Annotations for {len(annot)} genes")
    # Keep only matching genes
    annot = annot[annot.index.isin(adata.var.index)]
    adata = adata[:, adata.var.index.isin(annot.index)]
    adata.var = pd.concat([adata.var, annot], axis=1)
    return adata


def normalize_expression(
    adata: ad.AnnData, log1p: bool = True, target_sum: Optional[int] = 10_000
) -> ad.AnnData:
    # Store counts in seperate layer
    adata.layers["counts"] = adata.X.copy()
    if target_sum is not None:
        # Normalize by library size
        print(f"Relative library size {adata.X.sum(axis=1).mean() / target_sum}")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        adata.layers["normalized_counts"] = adata.X.copy()

    if log1p:
        # Log1p transform expression
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()
    return adata


def clean_cells_genes(adata: ad.AnnData) -> ad.AnnData:
    sc.pp.filter_cells(adata, min_genes=1)

    # # Drop cell/ tissue types with less than 50 cells
    # drop_cells = adata.obs["count"] < 50
    # print(f"Dropping {drop_cells.sum()} cell/tissue types")
    # adata = adata[np.logical_not(drop_cells), :]

    # Drop genes for which no expression recorded
    drop_genes = adata.X.sum(axis=0) == 0
    print(f"Dropping {drop_genes.sum()} genes")
    adata = adata[:, np.logical_not(drop_genes)]

    # Drop non-protein coding genes
    drop_genes = adata.var["gene_biotype"] != "protein_coding"
    print(f"Dropping {drop_genes.sum()} genes")
    adata = adata[:, np.logical_not(drop_genes)]
    return adata


def calculate_cell_embeddings_pca(adata: ad.AnnData, n_pcs: int = 512) -> ad.AnnData:
    # consider only training cells and genes
    train_cells = adata.obs["train"]
    train_genes = adata.var["train"]
    adata_train = adata[train_cells, train_genes]

    # fit pca on training data
    pca = PCA()
    pca.fit(adata_train.X)

    # compute scores for all cells
    pca_expression = pca.transform(adata[:, train_genes].X)

    # add cell embeddings to obsm
    adata.obsm["embedding"] = pca_expression[:, :n_pcs]
    adata.uns["obs_embedding_pca"] = {
        "explained_variance": np.cumsum(pca.explained_variance_ratio_)[n_pcs]
    }
    return adata


def calculate_gene_embeddings_pca(adata: ad.AnnData, n_pcs: int = 512) -> ad.AnnData:
    # consider only training cells and genes
    # train_genes = adata.var["train"]
    # adata_train = adata[:, train_genes]

    embedding = adata.varm["embedding"]
    # fit pca on training data
    pca = PCA()
    pca.fit(embedding)
    pca_embedding = pca.transform(embedding)

    # add cell embeddings to obsm
    adata.varm["embedding_pca"] = pca_embedding[:, :n_pcs]
    adata.uns["var_embedding_pca"] = {
        "explained_variance": np.cumsum(pca.explained_variance_ratio_)[n_pcs]
    }
    return adata


def add_dendrogram_and_hvgs(adata: ad.AnnData) -> ad.AnnData:
    # Add highly variable genes
    if "log1p" in adata.uns:
        adata.uns["log1p"]["base"] = None  # needed to deal with error?
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # Add dendrogram
    sc.tl.dendrogram(adata, groupby="label", use_rep="X")
    return adata


def average_expression_per_feature(adata: ad.AnnData, feature_name: str) -> ad.AnnData:
    features = adata.obs[feature_name].value_counts()
    n_features = len(features)
    n_var = len(adata.var)
    features_by_var = np.zeros((n_features, n_var), dtype=np.int64)

    obs = features.to_frame(name="count")
    obs.reset_index(level=0, inplace=True, names=feature_name)

    for index, gv in tqdm(enumerate(features.index)):
        values = adata.X[adata.obs[feature_name] == gv].sum(axis=0)
        np.add.at(features_by_var, [index], values.tolist())

    with np.errstate(divide="ignore", invalid="ignore"):
        features_by_var = np.divide(features_by_var, np.expand_dims(features, axis=1))
        features_by_var[np.isnan(features_by_var)] = 0

    return ad.AnnData(X=features_by_var, obs=obs, var=adata.var)


def add_marker_genes(
    adata: ad.AnnData, differential_expression: pd.DataFrame
) -> ad.AnnData:
    # Add differentially expressed genes in test dataset
    test_genes = np.logical_not(adata.var["train"])
    marker_genes_dict = {}  # type: Dict[str, str]
    cats = adata.obs.label.cat.categories
    for i, c in enumerate(cats):
        cid = "{} vs Rest".format(c)
        label_df = differential_expression.loc[
            differential_expression.comparison == cid
        ]
        label_df = label_df[label_df["lfc_mean"] > 0]
        label_df = label_df[label_df["bayes_factor"] > 3]
        label_df = label_df[label_df["non_zeros_proportion1"] > 0.1]

        # Restrict to genes that are in test dataset
        label_df = label_df[label_df.index.isin(adata[:, test_genes].var.index)]

        # Restrict to genes that havn't already been used if possible
        used_genes = set(marker_genes_dict.values())
        if len(used_genes.union(set(label_df.index))) > len(label_df.index):
            label_df = label_df[~label_df.index.isin(used_genes)]

        # Get best marker gene
        marker_genes_dict[c] = label_df["lfc_mean"].idxmax()

    # Add marker genes to adata
    adata.obs["marker_gene"] = adata.obs["label"].map(marker_genes_dict)
    adata.obs["marker_feature_name"] = adata.var.loc[adata.obs["marker_gene"]][
        "feature_name"
    ].values
    return adata


def bin_expression(
    adata: ad.AnnData,
    n_bins: int,
    input_layer: Optional[str] = None,
    output_edges: str = "bin_edges",
    output_layer: str = "binned",
) -> ad.AnnData:
    binned_X = []
    bin_edges = []
    if input_layer is None:
        expression = adata.X
    else:
        expression = adata.layers[input_layer]

    for obs in expression:
        obs = np.asarray(obs)
        non_zero_ids = obs.nonzero()
        non_zero_obs = obs[non_zero_ids]
        bins = np.quantile(non_zero_obs, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = np.digitize(non_zero_obs, bins)
        binned_obs = np.zeros_like(obs, dtype=np.int64)
        binned_obs[non_zero_ids] = non_zero_digits
        binned_X.append(binned_obs)
        bin_edges.append(np.concatenate([[0], bins]))

    adata.layers[output_layer] = np.stack(binned_X)
    adata.obsm[output_edges] = np.stack(bin_edges)

    return adata


def reconstruct_expression(
    adata: ad.AnnData,
    input_layer: Optional[str] = "binned",
    input_edges: str = "bin_edges",
    output_layer: str = "reconstructed",
) -> ad.AnnData:
    if input_layer is None:
        binned_data = adata.X
    else:
        binned_data = adata.layers[input_layer]

    reconstructed_X = []
    for binned_obs, bins in zip(binned_data, adata.obsm[input_edges]):
        bin_sizes = np.diff(bins[1:])
        cumulative_sum = np.cumsum(bin_sizes)
        bin_centers = cumulative_sum - bin_sizes / 2
        # Add zero bin so zero values get mapped to zero, add max value so number of bins correct
        bin_centers = np.concatenate([[0], bin_centers, [bins[-1]]])
        # ensure binned_obs in valid range
        binned_obs_valid = np.clip(
            np.round(binned_obs), 0, len(bin_centers) - 1
        ).astype(int)
        # reconstruct expression
        reconstructed_X.append(bin_centers[binned_obs_valid])

    adata.layers[output_layer] = np.stack(reconstructed_X)
    return adata


def bulk_data(adata: ad.AnnData, config: BulkDataConfig) -> ad.AnnData:
    # Note that genes with no samples will be dropped
    padata = dc.get_pseudobulk(
        adata,
        sample_col=config.sample_col,
        groups_col=config.label_col,
        layer=None,
        min_prop=0,
        min_cells=0,
        min_counts=0,
        min_smpls=0,
    )
    padata.var = adata.var.loc[padata.var.index]
    padata.obs['is_primary_data'] = padata.obs['is_primary_data'].astype(str)
    return padata


def create_io_paths(config: PathConfig) -> tuple[Path, Path]:
    input_path = Path(config.base) / (config.dataset + '.h5ad')
    output_path = Path(config.base) / (config.dataset + '_' + config.preprocessed + '.h5ad')
    return (input_path, output_path)


def preprocessing_pipeline(adata: ad.AnnData, config: PreProcessingConfig) -> ad.AnnData:
    # If necessary perform bulking
    if config.bulk_data is not None:
        adata = bulk_data(adata, config.bulk_data)

    # Add gene biotype information
    adata = add_gene_biotype(adata)

    # Attach preprocessing metadata
    adata.uns['preprocessing'] = OmegaConf.to_container(config)

    return adata


def preprocess(config: RosaConfig) -> None:
    input_path, output_path = create_io_paths(config.paths)
    
    if output_path.exists():
        print(f'Output path {output_path} exists, loading data')

        adata = ad.read_h5ad(output_path)
        print(adata)

        if adata.uns.get('preprocessing', None) == OmegaConf.to_container(config.preprocessing):
            print(f'Data is preprocessed')
            return

    print(f'Trying to load input data at {input_path}')
    adata = ad.read_h5ad(input_path)
    print(adata)

    # Add gene embeddings
    adata = add_gene_embeddings(adata, config.paths.gene_embeddings)

    # Preprocess
    adata = preprocessing_pipeline(adata, config.preprocessing)
    print(adata)

    # Save anndata object
    adata.write_h5ad(output_path)


###########################################################
# Add train indicators
# adata = add_train_indicators(adata, fraction=0.7, seed=42)

# Clean cells and genes
# adata = clean_cells_genes(adata)
# # Normalize expression
# adata = normalize_expression(adata)
# # Bin expression
# adata = bin_expression(adata, n_bins=128)

# # # Calculate and add cell embeddings using only traning data
# adata = calculate_cell_embeddings_pca(adata, 64)


# # Add dendrogram, hvgs, and marker genes
# adata = add_dendrogram_and_hvgs(adata)
# de_df = pd.read_csv(TABULA_SAPIENS_BY_CELL_TYPE_SCVI_DE, index_col=0)
# adata = add_marker_genes(adata, differential_expression=de_df)
###########################################################