import warnings

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import kstest, spearmanr

# warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)


def score_predictions(adata, output_layer="predicted", target_layer="measured"):

    # Identify cells and genes not trained on (when possible)
    test_genes = np.logical_not(adata.var["train"])
    test_cells = np.logical_not(adata.obs["train"])
    adata_test = adata[test_cells, test_genes]
    sc.tl.dendrogram(adata_test, groupby="label", use_rep="X")

    # Extract measured and predicted expression
    X_meas = adata_test.layers[target_layer]
    X_pred = adata_test.layers[output_layer]

    # Compute and store results
    results = {}
    results["mse"] = ((X_pred - X_meas) ** 2).mean()
    results["mse_across_cells"] = [
        ((a - b) ** 2).mean() for a, b in zip(X_pred.T, X_meas.T)
    ]
    results["mse_across_genes"] = [
        ((a - b) ** 2).mean() for a, b in zip(X_pred, X_meas)
    ]

    results["spearmanr"] = spearmanr(X_pred.ravel(), X_meas.ravel()).correlation
    results["spearmanr_across_cells"] = [
        spearmanr(a, b).correlation for a, b in zip(X_pred.T, X_meas.T)
    ]
    results["spearmanr_across_genes"] = [
        spearmanr(a, b).correlation for a, b in zip(X_pred, X_meas)
    ]
    results["spearmanr_across_cells_mean"] = np.nanmean(
        results["spearmanr_across_cells"]
    )
    results["spearmanr_across_genes_mean"] = np.nanmean(
        results["spearmanr_across_genes"]
    )

    ks = kstest(X_pred.ravel(), X_meas.ravel())
    results["ks_statistic"] = ks.statistic
    results["ks_pvalue"] = ks.pvalue

    total_expression = np.expm1(X_meas).sum(axis=1)
    total_expression_pred = np.expm1(X_pred).sum(axis=1)
    results["total_expression_captured"] = (
        total_expression_pred.mean() / total_expression.mean() * 100
    )

    print(
        f"""
        mean spearmanr across genes {results['spearmanr_across_genes_mean']:.3f}
        mean spearmanr across cells {results['spearmanr_across_cells_mean']:.3f}
        mean square error {results['mse']:.3f}
        ks-statistic on total expression {results['ks_statistic']:.3f}
        mean percent total expression captured per cell {results['total_expression_captured']:.3f}
        """
    )
    return adata_test, results
