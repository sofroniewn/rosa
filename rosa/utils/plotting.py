import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def plot_marker_gene_heatmap(
    adata,
    marker_genes,
    output_layer: str = "predicted",
    target_layer="target",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={"wspace": 0})
    max_expression_val = adata.uns['nbins']
    sc.pl.matrixplot(
        adata,
        marker_genes,
        groupby="label",
        gene_symbols="feature_name",
        layer=target_layer,
        vmin=0,
        vmax=max_expression_val,
        ax=ax1,
        show=False,
        title=target_layer,
        dendrogram=True,
    )
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    if type(marker_genes) is dict:
        fig.axes[-1].remove()

    sc.pl.matrixplot(
        adata,
        marker_genes,
        groupby="label",
        gene_symbols="feature_name",
        layer=output_layer,
        vmin=0,
        vmax=max_expression_val,
        ax=ax2,
        show=False,
        title=output_layer,
        dendrogram=True,
    )
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    if type(marker_genes) is dict:
        fig.axes[-1].remove()
    fig.axes[-1].set_yticklabels([])


def plot_expression_and_correlation(
    results
):
    _, axs = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"wspace": 0.2})

    cm = results['confusion_matrix']
    total = cm.sum()
    nbins = cm.shape[0]

    bins = np.arange(nbins)
    ylim = [0, 1 / nbins * 2.5]
    xlim = [-0.5, nbins-0.5]
    axs[0, 0].bar(bins, cm.sum(axis=1) / total) # target
    axs[0, 0].bar(bins, cm.sum(axis=0) / total, alpha=0.5) # predicted
    axs[0, 0].set_ylim(ylim)
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_xlabel("expression")

    # Subplot with confusion matrix
    axs[0, 1].imshow(cm / cm.sum(axis=1)[:, np.newaxis])
    axs[0, 1].set_xlim(xlim)
    axs[0, 1].set_ylim(xlim)
    axs[0, 1].set_aspect("equal", adjustable="box")
    axs[0, 1].set_xlabel("expression predicted")
    axs[0, 1].set_ylabel("expression target")

    # Subplot with correlation across genes
    axs[1, 0].hist(
        results["spearman_obs"], bins=np.linspace(0, 1, 50), density=True
    )
    axs[1, 0].set_xlabel("spearmanr across genes (each data point is a cell)")
    axs[1, 0].set_xlim([0, 1])

    # Subplot with correlation across genes
    axs[1, 1].hist(
        results["spearman_var"], bins=np.linspace(0, 1, 100), density=True
    )
    axs[1, 1].set_xlabel("spearmanr across cells (each data point is a gene)")
    axs[1, 1].set_xlim([0, 1])