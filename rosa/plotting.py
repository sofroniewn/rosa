import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc


def plot_marker_gene_heatmap(adata, marker_genes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={"wspace": 0})
    sc.pl.matrixplot(
        adata,
        marker_genes,
        groupby="label",
        gene_symbols="feature_name",
        layer=None,
        vmin=0,
        vmax=6,
        ax=ax1,
        show=False,
        title="measured",
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
        layer="prediction",
        vmin=0,
        vmax=6,
        ax=ax2,
        show=False,
        title="predicted",
        dendrogram=True,
    )
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    fig.axes[-1].remove()
    if type(marker_genes) is dict:
        fig.axes[-1].remove()
    fig.axes[-1].set_yticklabels([])


def plot_expression_and_correlation(adata, results):
    _, axs = plt.subplots(3, 2, figsize=(14, 13), gridspec_kw={"wspace": 0.2})

    max_expression_val = 6
    # Subplot with expression histograms
    bins = np.linspace(0, max_expression_val, 1000)
    axs[0, 0].hist(adata.X.flatten(), bins=bins, density=True)
    axs[0, 0].hist(
        adata.layers["prediction"].flatten(), bins=bins, density=True, alpha=0.5
    )
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, max_expression_val])
    axs[0, 0].set_xlabel("expression")

    # Subplot with expression scatter
    axs[0, 1].plot(
        adata.X.flatten(), adata.layers["prediction"].flatten(), ".", alpha=0.01
    )
    axs[0, 1].plot(
        [0, max_expression_val],
        [0, max_expression_val],
        "k",
        linewidth="2",
        linestyle="--",
    )
    axs[0, 1].set_xlim([0, max_expression_val])
    axs[0, 1].set_ylim([0, max_expression_val])
    axs[0, 1].set_aspect("equal", adjustable="box")
    axs[0, 1].set_xlabel("expression actual")
    axs[0, 1].set_ylabel("expression predicted")

    # Subplot with correlation across genes
    axs[1, 0].hist(
        results["pearsonr_across_genes"], bins=np.linspace(0, 1, 50), density=True
    )
    axs[1, 0].set_xlabel("pearsonr across genes (each data point is a cell)")
    axs[1, 0].set_xlim([0, 1])

    # Subplot with correlation across genes
    axs[1, 1].hist(
        results["pearsonr_across_cells"], bins=np.linspace(0, 1, 100), density=True
    )
    axs[1, 1].set_xlabel("pearsonr across cells (each data point is a gene)")
    axs[1, 1].set_xlim([0, 1])

    # Subplot with correlation vs mean expression for genes
    axs[2, 1].plot(
        adata.X.mean(axis=0), results["pearsonr_across_cells"], ".", alpha=0.1
    )
    axs[2, 1].set_xlabel("mean expression across cells (each data point is a gene)")
    axs[2, 1].set_ylabel("pearsonr across cells")
    axs[2, 1].set_xlim([0, max_expression_val])

    axs[2, 0].set_visible(False)
