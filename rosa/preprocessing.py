import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from tqdm import tqdm


def add_gene_embeddings(adata, seqs, embeds):
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


def add_train_indicators(adata, fraction=0.7, seed=None):
    # Set train indicator
    if seed is None:
        np.random.seed(42)
    else:
        np.random.seed(seed)
    adata.var["train"] = np.random.rand(adata.n_vars) <= fraction
    adata.obs["train"] = np.random.rand(adata.n_obs) <= fraction
    return adata


def add_gene_biotype(adata):
    print("Adding gene biotypes")
    # Retrieve gene symbols
    annot = sc.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", "external_gene_name", "gene_biotype"],
        use_cache=True,
    ).set_index("ensembl_gene_id")

    # Keep only matching genes
    annot = annot[annot.index.isin(adata.var.index)]
    adata = adata[:, adata.var.index.isin(annot.index)]
    adata.var = pd.concat([adata.var, annot], axis=1)
    return adata


def normalize_expression(adata, method="Log1p", target_sum=1e5):
    # Store counts in seperate layer
    adata.layers["counts"] = adata.X.copy()

    if method == "Log1p":
        # Normalize by library size
        print(f"Relative library size {adata.X.sum(axis=1).mean() / target_sum}")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # Log1p transform expression
        sc.pp.log1p(adata)
    elif method == "Pearson":
        # Pearson normalize
        adata.X = np.ceil(adata.X)
        sc.pp.filter_genes(adata, min_cells=1)
        sc.experimental.pp.normalize_pearson_residuals(adata)
        adata.X[adata.X < 0] = 0
    else:
        raise ValueError(f"Method {method} not recognized")

    return adata


def clean_cells_genes(adata):
    # Drop cell/ tissue types with less than 50 cells
    drop_cells = adata.obs["count"] < 50
    print(f"Dropping {drop_cells.sum()} cell/tissue types")
    adata = adata[np.logical_not(drop_cells), :]

    # Drop genes for which no expression recorded
    drop_genes = adata.X.sum(axis=0) == 0
    print(f"Dropping {drop_genes.sum()} genes")
    adata = adata[:, np.logical_not(drop_genes)]

    # Drop non-protein coding genes
    drop_genes = adata.var["gene_biotype"] != "protein_coding"
    print(f"Dropping {drop_genes.sum()} genes")
    adata = adata[:, np.logical_not(drop_genes)]
    return adata


def calculate_cell_embeddings_pca(adata):
    # consider only training cells and genes
    train_cells = adata.obs["train"]
    train_genes = adata.var["train"]
    adata_train = adata[train_cells, train_genes]

    # fit pca on training data
    pca = PCA()
    pca.fit(adata_train.X)

    # compute scores for all cells
    full_cell_embeddings = pca.transform(adata[:, train_genes].X)

    # add cell embeddings to obsm
    adata.obsm["embedding"] = full_cell_embeddings
    return adata


def add_dendrogram_and_hvgs(adata):
    # Add highly variable genes
    adata.uns["log1p"]["base"] = None  # needed to deal with error?
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # Add dendrogram
    sc.tl.dendrogram(adata, groupby="label", use_rep="X")
    return adata


def average_expression_per_feature(adata, feature_name):
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


def add_marker_genes(adata, differential_expression):
    # Add differentially expressed genes in test dataset
    test_genes = np.logical_not(adata.var["train"])
    marker_genes_dict = {}
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
