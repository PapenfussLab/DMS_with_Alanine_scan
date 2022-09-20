"""Analyze prediction performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr


def subgroup_spearmanr(data, level, x_col, y_col):
    """Calculate value Spearman's correlation in each subgroup of input data.

    Parameters
    ----------
    data: pd.DataFrame
    level: str
        Column in data indicating subgroups.
    x_col: str
        Column in data indicating values for correlation computation.
    y_col: str
        Column in data indicating values for correlation computation.

    Returns
    -------
    group_perf: pd.DataFrame
        DataFrame with the size and correlation value for each subgroup.
    """
    group_perf = dict()
    for group, df in data.groupby(level):
        row = dict()
        row["rho"] = spearmanr(df[x_col], df[y_col])[0]
        row["size"] = len(df)
        group_perf[group] = row
    group_perf = pd.DataFrame(group_perf).T
    return group_perf


def plot_roc_curve_on_axes(ax, label, y_true, y_score):
    """Illustrate the accuracy of classification result against its specificity.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to plot on.
    label: str
        Label for plot legend.
    y_true: pd.Series
        True binary labels.
    y_score: pd.Series
        Target scores for the given labels.
    """
    # Calculate parameters for ROC plot.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = np.round(auc(fpr, tpr), 3)

    # Make ROC plot.
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    return
