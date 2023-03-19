import torch
from torchmetrics.functional import spearman_corrcoef
from torchmetrics.functional.classification import (
    multiclass_accuracy, multiclass_confusion_matrix, multiclass_f1_score,
    multiclass_precision, multiclass_recall)


def score_predictions(predicted, target, nbins):
    # Compute and store results
    results = {}

    results["spearman_obs"] = (
        spearman_corrcoef(predicted.T.float(), target.T.float()).detach().numpy()
    )
    results["spearman_var"] = (
        spearman_corrcoef(predicted.float(), target.float()).detach().numpy()
    )

    results["spearman_obs_mean"] = results["spearman_obs"].mean()
    results["spearman_var_mean"] = results["spearman_var"].mean()

    print(
        f"""
        mean spearman across cells {results['spearman_obs_mean']:.3f}
        mean spearman across genes {results['spearman_var_mean']:.3f}
        """
    )

    results["confusion_matrix"] = (
        multiclass_confusion_matrix(predicted.ravel(), target.ravel(), nbins)
        .detach()
        .numpy()
    )

    return results
