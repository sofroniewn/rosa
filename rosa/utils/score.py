from torchmetrics.functional import spearman_corrcoef
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix,
)


def score_predictions(predicted, target, nbins):
    # Compute and store results
    results = {}

    results["spearman_obs"] = spearman_corrcoef(predicted.T.float(), target.T.float())
    results["spearman_var"] = spearman_corrcoef(predicted.float(), target.float())

    results["spearman_obs_mean"] = results["spearman_obs"].mean()
    results["spearman_var_mean"] = results["spearman_var"].mean()

    results["confusion_matrix"] = multiclass_confusion_matrix(
        predicted.ravel(), target.ravel(), nbins
    )
    return results


def merge_images(img_A, img_B):
    rgb = img_A.unsqueeze(-1).repeat(1, 1, 3)
    rgb[:, :, 1] = img_B
    return rgb
