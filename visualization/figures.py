import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import visualization.parsers as par
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def median_dms_performance_scatterplot(model_perform):
    """Visualize predictor performance on datasets with median improvement by using alanine scanning (AS) data.

    Parameters
    ----------
    model_perform: pd.DataFrame
        Predictor performance while using AS data or not on each combined dataset with other dataset information.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    # Get datasets with median performance while using alanine scanning data or not.
    med_dmsas = par.pick_sort_med_rmse_diff(model_perform)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(20, 6),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.05},
    )
    par.bar_median_data_count(med_dmsas, axs[0])
    par.scatter_median_data_rmse(med_dmsas, axs[1])
    # Draw lines at the bottom of the scatter plot to indicate target proteins.
    par.underline_median_data_protein(med_dmsas, axs[1])
    plt.legend(fontsize=11, loc=1)
    return fig


def prediction_heatmap(data):
    """Visualize experimental and predicted results in a three-panel heat map.

    Parameters
    ----------
    data: pd.DataFrame
        Mutants data with experimental and predicted scores and other mutational information. Normally, should
        contain data from only one dataset.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    aa_order = [x for x in "MLIVASTNQDEKRHWYFPGC"]
    matrices = [
        par.create_heatmap_score_matrix(data, col, aa_order)
        for col in ["ob_score", "pred_score_ala", "pred_score_noala"]
    ]
    titles = [
        "Experiment result",
        "Prediction with AS data",
        "Prediction without AS data",
    ]

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(18, len(matrices[0]) / 4 + 1.5),
        constrained_layout=True,
        sharey=True,
    )
    fig.set_constrained_layout_pads(h_pad=0, hspace=0)
    im = None
    for i in range(3):
        im = par.create_score_heatmap_on_axes(axs[i], matrices[i].iloc[:, 2:])
        par.dot_score_heatmap_wt([axs[i]], matrices[i], aa_order)
        axs[i].set_ylim(len(matrices[0]) - 0.5, -0.5)
        axs[i].set_xticks(np.arange(len(aa_order)))
        axs[i].set_xticklabels(aa_order, fontsize=15)
        axs[i].set_title(titles[i], fontsize=25)
    axs[0].set_yticks(np.arange(len(matrices[0]["position"])))
    axs[0].set_yticklabels(matrices[0]["position"].astype(int), fontsize=15)
    plt.colorbar(im, ax=axs, orientation="horizontal", shrink=0.2, aspect=1)
    return fig


def mutation_type_bias_barplot(work_data):
    """Visualize relative RMSE change while using alanine scanning data on each mutation type in a bar plot.

    Parameters
    ----------
    work_data: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    mt_comp = par.calc_aa_type_rmse(work_data, "aa2")
    mt_comp["ratio"] = (
        (mt_comp["noala_rmse"] - mt_comp["ala_rmse"]) / mt_comp["noala_rmse"] * 100
    )
    fig = plt.figure(figsize=(14, 2.5))
    sns.barplot(x="aa", y="ratio", hue="aa_property", data=mt_comp, dodge=False)
    plt.xticks(ticks=np.arange(20), label=mt_comp["aa"])
    plt.xlabel("Mutation type")
    plt.ylabel("Relative RMSE change (%)", fontsize=12)
    plt.xlim(-1, 20)
    plt.ylim(0, plt.ylim()[1] * 1.25)
    plt.legend(loc=1, ncol=5)
    return fig


def wildtype_bias_barplot(work_data):
    """Visualize relative RMSE change while using alanine scanning data on each wildtype in a bar plot.

    Parameters
    ----------
    work_data: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    wt_comp = par.calc_aa_type_rmse(work_data, "aa1")
    wt_comp["ratio"] = (
        (wt_comp["noala_rmse"] - wt_comp["ala_rmse"]) / wt_comp["noala_rmse"] * 100
    )
    fig = plt.figure(figsize=(14, 2.5))
    g = sns.barplot(x="aa", y="ratio", hue="aa_property", data=wt_comp, dodge=False)
    plt.plot([-2, 19], [0, 0], "--", c="k", linewidth=1, zorder=-10)
    plt.xticks(ticks=np.arange(19), label=wt_comp["aa"])
    plt.xlabel("Wildtype")
    plt.ylabel("Relative RMSE change (%)", fontsize=12)
    plt.xlim(-2, 19)
    plt.ylim(plt.ylim()[0], plt.ylim()[1] * 1.05)
    g.legend_.remove()
    return fig


def assay_error_barplot(pred_result, dms_as_pair, assay_info):
    """Make bar plot showing RMSE on shared residues of two combined datasets while using AS data.

    Parameters
    ----------
    pred_result: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.
    dms_as_pair: list
        A list with two DMS+AS combined dataset IDs which are based on the same DMS data and will be compared.
    assay_info: dict
        The index are the dataset IDs and values are the assay information which will become the figure legend.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    share_mut = par.get_dmsas_shared_mutants(pred_result, dms_as_pair)
    pos_rmse = par.get_dmsas_pair_position_rmse(share_mut, dms_as_pair)
    pos_rmse["Assay"] = pos_rmse["dmsa_id"].map(assay_info)
    fig = plt.figure(figsize=(len(pos_rmse) / 4 + 2, 4))
    sns.barplot(data=pos_rmse, x="u_pos", y="rmse", hue="Assay")
    plt.xlabel("Position")
    plt.ylabel("RMSE")
    plt.ylim(0, plt.ylim()[1] * 1.08)
    plt.xticks(
        ticks=np.arange(len(pos_rmse) / 2),
        labels=pos_rmse["u_pos"].sort_values().unique().astype(int),
    )
    plt.legend(loc="lower left", title="AS assay")
    return fig


def plot_roc_curve_on_axes(ax, labels, *data_pairs):
    """Illustrate the accuracy of the predictor against its specificity.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    labels: list
        List of labels for each pair of data.
    data_pairs: tuple
        Variable number of lists with two elements for each. The first one is if it is positive and the
        second one is predicted scores.
    """
    # Calculate parameters for ROC plot.
    fpr = []
    tpr = []
    roc_auc = []
    for pair in data_pairs:
        fpr_i, tpr_i, threshold = roc_curve(pair[0], pair[1])
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        roc_auc.append(np.round(auc(fpr_i, tpr_i), 3))

    # Make ROC plot.
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], lw=2, label=f"{labels[i]} (area = {roc_auc[i]})")
    ax.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--", label="Baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    return


def protein_structure_bias_heatmap(structure_data, mut_prediction, colour_range):
    """Visualize relative RMSE change while using alanine scanning data on each protein regions in a heat map.

    The heat map shows the relative RMSE change for using alanine scanning data on each DMS dataset (in columns)
    for each structural regions (in rows), including alpha helix, beta sheet, random coil, turn, buried region
    and surface region.

    Parameters
    ----------
    structure_data: pd.DataFrame
        Non-preprocessed protein secondary structure, accessibility and surface exposure values with residue
        information.
    mut_prediction: pd.DataFrame
        Mutants data with experimental and predicted scores and other mutational information.
    colour_range: tuple
        Tuple with two elements indicating the range for the colour bar.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure created.
    """
    secs_prediction = pd.merge(
        structure_data[["dms_id", "position", "dssp_sec_str"]]
        .dropna()
        .drop_duplicates(),
        mut_prediction,
        how="inner",
        on=["dms_id", "position"],
        validate="1:m",
    )
    secs_perf = par.calc_structure_type_rmse_change(secs_prediction, "dssp_sec_str")
    # Situations for surface_data: i) 'is_surface' is true for surface residues; ii) 'is_surface' is False but
    # 'accessibility' is not NA for buried residues; False and NA for residues without such information.
    # Accessibility data for TP53 are not available for all the substitution type on the same residue for
    # unknown reason and are removed.
    surf_prediction = pd.merge(
        structure_data[["dms_id", "position", "accessibility", "is_surface"]]
        .dropna()
        .drop_duplicates(),
        mut_prediction,
        how="inner",
        on=["dms_id", "position"],
        validate="1:m",
    )
    surf_perf = par.calc_structure_type_rmse_change(surf_prediction, "is_surface")
    str_perf = pd.concat([secs_perf, surf_perf], axis=1, sort=True)

    fig, axs = plt.subplots(
        2, 1, figsize=(19, 3), gridspec_kw={"height_ratios": [6, 3]}
    )
    im = par.spineless_piyg_heatmap(
        axs[0],
        str_perf[["H", "E", ".", "T"]].T,
        str_perf.index,
        ["α helix", "β sheet", "Coil", "Turn"],
        colour_range,
        False,
    )
    im = par.spineless_piyg_heatmap(
        axs[1],
        str_perf[["False", "True"]].T,
        str_perf.index,
        ["Buried", "Surface"],
        colour_range,
        True,
    )
    cbaxes = fig.add_axes([0.87, 0.17, 0.008, 0.67])
    cbar = plt.colorbar(im, ax=axs, shrink=0.5, cax=cbaxes)
    cbar.ax.set_ylabel("Relative RMSE change (%)", fontsize=12, rotation=270)
    cbar.ax.yaxis.set_label_coords(5, 0.5)
    return fig
