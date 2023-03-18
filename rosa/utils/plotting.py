import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.metrics import confusion_matrix


def plot_marker_gene_heatmap(
    adata,
    marker_genes,
    output_layer: str = "predicted",
    target_layer="measured",
    max_expression_val: int = 6,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={"wspace": 0})
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
    adata,
    results,
    output_layer="predicted",
    target_layer="measured",
    max_expression_val: int = 6,
    nbins: int = None,
):
    _, axs = plt.subplots(3, 2, figsize=(14, 13), gridspec_kw={"wspace": 0.2})

    if target_layer is None:
        X_meas = adata.X
    else:
        X_meas = adata.layers[target_layer]
    X_pred = adata.layers[output_layer]

    # Subplot with expression histograms
    if nbins is None:
        bins = np.linspace(0, max_expression_val, 200)
        ylim = [0, 1]
        xlim = [0, max_expression_val]
    else:
        bins = np.arange(nbins + 1) - 0.5
        ylim = [0, 1 / nbins * 2.5]
        xlim = [-0.5, nbins-0.5]
    axs[0, 0].hist(X_meas.flatten(), bins=bins, density=True)
    axs[0, 0].hist(X_pred.flatten(), bins=bins, density=True, alpha=0.5)
    axs[0, 0].set_ylim(ylim)
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_xlabel("expression")

    if nbins is None:
        # Subplot with expression scatter
        axs[0, 1].plot(X_meas.flatten(), X_pred.flatten(), ".", alpha=0.01)
        axs[0, 1].plot(
            xlim,
            xlim,
            "k",
            linewidth="2",
            linestyle="--",
        )
        axs[0, 1].set_xlim(xlim)
        axs[0, 1].set_ylim(xlim)
        axs[0, 1].set_aspect("equal", adjustable="box")
        axs[0, 1].set_xlabel("expression measured")
        axs[0, 1].set_ylabel("expression predicted")
    else:
        # Subplot with confusion matrix
        cm = confusion_matrix(X_meas.flatten(), X_pred.flatten(), labels=list(range(nbins)), normalize='true')
        axs[0, 1].imshow(cm)
        axs[0, 1].set_xlim(xlim)
        axs[0, 1].set_ylim(xlim)
        axs[0, 1].set_aspect("equal", adjustable="box")
        axs[0, 1].set_xlabel("expression predicted")
        axs[0, 1].set_ylabel("expression measured")

    # Subplot with correlation across genes
    axs[1, 0].hist(
        results["spearmanr_across_genes"], bins=np.linspace(0, 1, 50), density=True
    )
    axs[1, 0].set_xlabel("spearmanr across genes (each data point is a cell)")
    axs[1, 0].set_xlim([0, 1])

    # Subplot with correlation across genes
    axs[1, 1].hist(
        results["spearmanr_across_cells"], bins=np.linspace(0, 1, 100), density=True
    )
    axs[1, 1].set_xlabel("spearmanr across cells (each data point is a gene)")
    axs[1, 1].set_xlim([0, 1])

    # Subplot with correlation vs mean expression for genes
    axs[2, 1].plot(
        X_meas.mean(axis=0), results["spearmanr_across_cells"], ".", alpha=0.1
    )
    axs[2, 1].set_xlabel("mean expression across cells (each data point is a gene)")
    axs[2, 1].set_ylabel("spearmanr across cells")
    axs[2, 1].set_xlim(xlim)

    axs[2, 0].set_visible(False)
