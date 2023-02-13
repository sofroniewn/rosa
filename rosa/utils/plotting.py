import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def plot_marker_gene_heatmap(adata, marker_genes, output_layer: str = "prediction"):
    max_expression_val = 6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 12), gridspec_kw={"wspace": 0})
    sc.pl.matrixplot(
        adata,
        marker_genes,
        groupby="label",
        gene_symbols="feature_name",
        layer=None,
        vmin=0,
        vmax=max_expression_val,
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
    adata, results, output_layer="prediction", target_layer=None
):
    _, axs = plt.subplots(3, 2, figsize=(14, 13), gridspec_kw={"wspace": 0.2})

    if target_layer is None:
        X_meas = adata.X
    else:
        X_meas = adata.layers[target_layer]
    X_pred = adata.layers[output_layer]

    max_expression_val = 6
    # Subplot with expression histograms
    bins = np.linspace(0, max_expression_val, 200)
    axs[0, 0].hist(X_meas.flatten(), bins=bins, density=True)
    axs[0, 0].hist(X_pred.flatten(), bins=bins, density=True, alpha=0.5)
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xlim([0, max_expression_val])
    axs[0, 0].set_xlabel("expression")

    # Subplot with expression scatter
    axs[0, 1].plot(X_meas.flatten(), X_pred.flatten(), ".", alpha=0.01)
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
        adata.X.mean(axis=0), results["spearmanr_across_cells"], ".", alpha=0.1
    )
    axs[2, 1].set_xlabel("mean expression across cells (each data point is a gene)")
    axs[2, 1].set_ylabel("spearmanr across cells")
    axs[2, 1].set_xlim([0, max_expression_val])

    axs[2, 0].set_visible(False)
