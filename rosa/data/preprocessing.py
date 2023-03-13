from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import zarr
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..utils.config import (
    BulkDataConfig,
    CellEmbeddingsConfig,
    ExpressionTransformConfig,
    FilterConfig,
    GeneEmbeddingsConfig,
    MarkerGeneConfig,
    PathConfig,
    PreProcessingConfig,
    RosaConfig,
    SplitConfig,
)


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


def add_gene_embeddings(adata: ad.AnnData, config: GeneEmbeddingsConfig) -> ad.AnnData:
    # Read gene intervals
    seqs = pl.read_csv(config.gene_intervals, sep="\t", has_header=False).to_pandas()
    # Set index to ensmbl id which is in column 5
    seqs = seqs.set_index("column_5")

    # Load gene embeddings
    embeds = np.asarray(zarr.open(config.path))

    adata = _add_gene_embeddings(adata, seqs, embeds)

    # calculate gene embedding pcas
    if config.pcs is not None:
        adata = calculate_gene_embeddings_pca(adata, config.pcs)
    return adata


def add_indicators(
    adata: ad.AnnData,
    config: SplitConfig,
) -> ad.AnnData:
    np.random.seed(config.seed)
    adata.var[config.key] = np.random.rand(adata.n_vars) <= config.fraction
    adata.obs[config.key] = np.random.rand(adata.n_obs) <= config.fraction
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
    adata: ad.AnnData, log1p: bool = True, target_sum: Optional[int] = 100_000
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


def filter_cells_and_genes(adata: ad.AnnData, config: FilterConfig) -> ad.AnnData:
    # Cells must express at least one gene
    sc.pp.filter_cells(adata, min_genes=1)

    # Drop genes for which no expression recorded
    drop_genes = adata.X.sum(axis=0) == 0
    print(f"Dropping {drop_genes.sum()} genes")
    adata = adata[:, np.logical_not(drop_genes)]

    # Drop non-protein coding genes
    if config.coding_only == True:
        drop_genes = adata.var["gene_biotype"] != "protein_coding"
        print(f"Dropping {drop_genes.sum()} genes")
        adata = adata[:, np.logical_not(drop_genes)]
    return adata


def add_cell_embeddings(adata: ad.AnnData, config: CellEmbeddingsConfig) -> ad.AnnData:
    # consider only training cells and genes
    if config.key is not None:
        train_cells = adata.obs[config.key]
        train_genes = adata.var[config.key]
        adata_split = adata[train_cells, train_genes]
    else:
        adata_split = adata

    # fit pca on training data
    pca = PCA()
    pca.fit(adata_split.X)

    # compute scores for all cells
    pca_expression = pca.transform(adata[:, train_genes].X)

    # add cell embeddings to obsm
    n_pcs = config.pcs
    n_pcs = min(n_pcs, pca_expression.shape[1] - 1)
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


def add_rank_genes_groups_markers(
    adata: ad.AnnData, config: MarkerGeneConfig
) -> ad.AnnData:
    markers = {}  # type: Dict[str, str]
    expression_thresh = config.mean_expression_threshold
    mean_expression = np.squeeze(np.array(adata.X.mean(axis=0)))
    
    test_genes = adata.var[
        np.logical_and(
            np.logical_not(adata.var["train"]), mean_expression > np.quantile(mean_expression, expression_thresh)
        )
    ].index
    for c in adata.uns["rank_genes_groups"]["names"].dtype.names:
        genes = adata.uns["rank_genes_groups"]["names"][c]
        logfoldchanges = adata.uns["rank_genes_groups"]["logfoldchanges"][c]
        scores = adata.uns["rank_genes_groups"]["scores"][c]
        thresh = np.quantile(scores, config.score_quantile)
        keep = np.logical_not(np.isin(genes, list(markers.values())))
        keep = np.logical_and(keep, scores > thresh)
        keep = np.logical_and(keep, np.isin(genes, test_genes))
        genes = genes[keep]
        logfoldchanges = logfoldchanges[keep]
        idx = logfoldchanges.argmax()
        markers[c] = genes[idx]
    adata.obs["marker_gene"] = adata.obs[config.label_col].map(markers)
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
    # padata = average_expression_per_feature(adata, config.label_col)
    if config.sample_col != 'single_cell':
        padata = dc.get_pseudobulk(
            adata,
            sample_col=config.sample_col,
            groups_col=config.label_col,
            layer=None,
            min_prop=0,
            min_cells=config.min_cells,
            min_counts=0,
            min_smpls=0,
        )
        padata.var = adata.var.loc[padata.var.index]
        padata.obs["is_primary_data"] = padata.obs["is_primary_data"].astype(str)
        padata.obs["label"] = padata.obs[config.label_col].astype("category")
        padata.obs["sample"] = padata.obs[config.sample_col].astype("category")
        padata.uns = adata.uns
    else:
        padata = adata
        padata.obs["label"] = adata.obs[config.label_col].astype("category")
    return padata


def create_io_paths(config: PathConfig) -> tuple[Path, Path]:
    input_path = Path(config.base) / (config.dataset + ".h5ad")
    if config.preprocessed is not None:
        output_path = Path(config.base) / (
            config.dataset + "_" + config.preprocessed + ".h5ad"
        )
    else:
        output_path = input_path
    return (input_path, output_path)


def rank_genes_groups(
    adata: ad.AnnData, adata_sc: ad.AnnData, config: MarkerGeneConfig
) -> ad.AnnData:
    if config.total_counts:
        sc.pp.normalize_total(adata_sc, target_sum=config.total_counts)
    if config.log1p:
        sc.pp.log1p(adata_sc)
    keep_indices = np.isin(
        adata_sc.var.index, adata.var[np.logical_not(adata.var["train"])].index
    )
    adata_sc = adata_sc[:, keep_indices]
    sc.tl.rank_genes_groups(adata_sc, config.label_col, method="t-test")
    adata.uns["rank_genes_groups"] = adata_sc.uns["rank_genes_groups"]
    return adata


def transform_expression(
    adata: ad.AnnData, config: ExpressionTransformConfig
) -> ad.AnnData:
    # Normalize expression
    if config.log1p is not None and config.total_counts is not None:
        adata = normalize_expression(
            adata, target_sum=config.total_counts, log1p=config.log1p
        )
    # Bin expression
    if config.n_bins is not None:
        adata = bin_expression(adata, n_bins=config.n_bins)
    return adata


def preprocessing_pipeline(
    adata_sc: ad.AnnData, config: PreProcessingConfig
) -> ad.AnnData:
    # If necessary perform bulking
    if config.bulk_data is not None:
        print("Bulk data")
        adata = bulk_data(adata_sc, config.bulk_data)
    else:
        adata = adata_sc

    # Add gene embeddings
    print("Add gene embeddings")
    adata = add_gene_embeddings(adata, config.gene_embeddings)

    # Add gene biotype information
    print("Add gene biotypes")
    adata = add_gene_biotype(adata)

    # Filter cells and genes
    if config.filter is not None:
        print("Filter cells and genes")
        adata = filter_cells_and_genes(adata, config.filter)

    # Add training indicators
    if config.split is not None:
        print("Add splits")
        adata = add_indicators(adata, config.split)

    # Normalize and bin expression
    if config.expression_transform is not None:
        print("Normalize expression")
        adata = transform_expression(adata, config.expression_transform)

    # Calculate and add cell embeddings using only traning data
    if config.cell_embeddings is not None:
        print("Add cell embeddings")
        adata = add_cell_embeddings(adata, config.cell_embeddings)

    # Add marker genes
    if config.markers is not None:
        print("Add rank genes groups")
        adata = rank_genes_groups(adata, adata_sc, config.markers)
        print("Add hvgs and marker genes")
        # adata = add_dendrogram_and_hvgs(adata)
        adata = add_rank_genes_groups_markers(adata, config.markers)

    # Attach preprocessing metadata
    print("Add preprocessing config")
    adata.uns["preprocessing"] = OmegaConf.to_container(config)

    return adata


def preprocess(config: RosaConfig) -> None:
    input_path, output_path = create_io_paths(config.paths)

    if output_path.exists():
        print(f"Output path {output_path} exists, loading data")

        adata = ad.read_h5ad(output_path)
        print(adata)

        if adata.uns.get("preprocessing", None) == OmegaConf.to_container(
            config.preprocessing
        ):
            print(f"Data is preprocessed")
            return

    print(f"Trying to load input data at {input_path}")
    adata = ad.read_h5ad(input_path)
    print(adata)

    # Preprocess
    adata = preprocessing_pipeline(adata, config.preprocessing)
    print(adata)

    # Save anndata object
    adata.write_h5ad(output_path)
    print(f"Data saved to {output_path}")
