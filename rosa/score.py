import warnings
from scipy.stats import pearsonr, kstest, ConstantInputWarning
import numpy as np
import scanpy as sc
import anndata as ad

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)


def score_predictions(adata):

    # Identify cells and genes not trained on (when possible)
    test_genes = np.logical_not(adata.var["train"])
    test_cells = np.logical_not(adata.obs["train"])
    adata_test = adata[test_cells, test_genes]
    sc.tl.dendrogram(adata_test, groupby="label", use_rep="X")

    # Extract measured and predicted expression
    X_meas = adata_test.X
    X_pred = adata_test.layers["prediction"]

    # Compute and store results
    results = {}
    results["mse"] = ((X_pred - X_meas) ** 2).mean()
    results["mse_across_cells"] = [
        ((a - b) ** 2).mean() for a, b in zip(X_pred.T, X_meas.T)
    ]
    results["mse_across_genes"] = [
        ((a - b) ** 2).mean() for a, b in zip(X_pred, X_meas)
    ]

    results["pearsonr"] = pearsonr(X_pred.ravel(), X_meas.ravel()).statistic
    results["pearsonr_across_cells"] = [
        pearsonr(a, b).statistic for a, b in zip(X_pred.T, X_meas.T)
    ]
    results["pearsonr_across_genes"] = [
        pearsonr(a, b).statistic for a, b in zip(X_pred, X_meas)
    ]
    results["pearsonr_across_cells_mean"] = np.nanmean(results["pearsonr_across_cells"])
    results["pearsonr_across_genes_mean"] = np.nanmean(results["pearsonr_across_genes"])

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
        mean pearsonr across genes {results['pearsonr_across_genes_mean']:.3f}
        mean pearsonr across cells {results['pearsonr_across_cells_mean']:.3f}
        mean square error {results['mse']:.3f}
        mean percent total expression captured per cell {results['total_expression_captured']:.3f}
        """
    )
    return adata_test, results
