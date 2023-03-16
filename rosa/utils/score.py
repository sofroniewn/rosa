import warnings

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import kstest, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)


def score_predictions(adata, output_layer="predicted", target_layer="measured", nbins=None):
    # Extract measured and predicted expression
    X_meas = adata.layers[target_layer]
    X_pred = adata.layers[output_layer]

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

    if nbins is not None:
        results["accuracy_score"] = accuracy_score(X_meas.ravel(), X_pred.ravel())
        precision, recall, fscore, support = precision_recall_fscore_support(X_meas.ravel(), X_pred.ravel(), labels=list(range(nbins)), average='macro')
        results["precision"] = precision
        results["recall"] = recall
        results["fscore"] = fscore
        print(
            f"""
            accuracy {results['accuracy_score']:.3f}
            precision {results['precision']:.3f}
            recall {results['recall']:.3f}
            fscore {results['fscore']:.3f}
            """
        )        

    return results
