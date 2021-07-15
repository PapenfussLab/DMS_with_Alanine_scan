from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matrix import create_score_matrix
from sklearn.metrics import mean_squared_error


AA_PROPERTY = {
    "W": "Aromatic",
    "F": "Aromatic",
    "Y": "Aromatic",
    "P": "Aliphatic",
    "M": "Aliphatic",
    "I": "Aliphatic",
    "L": "Aliphatic",
    "V": "Aliphatic",
    "A": "Aliphatic",
    "G": "Aliphatic",
    "C": "Polar uncharged",
    "S": "Polar uncharged",
    "T": "Polar uncharged",
    "Q": "Polar uncharged",
    "N": "Polar uncharged",
    "D": "Neg. charged",
    "E": "Neg. charged",
    "H": "Pos. charged",
    "R": "Pos. charged",
    "K": "Pos. charged",
}


class TwoSlopeNorm(colors.Normalize):
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
                                              vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])

        Notes
        -----
        This class is copied directly from later version of matplotlib source code.
        """

        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError("vmin, vcenter, and vmax must be in " "ascending order")
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError("vmin, vcenter, and vmax must be in " "ascending order")

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0]),
            mask=np.ma.getmask(result),
        )
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result


def pick_sort_med_rmse_diff(model_perform):
    """Pick and sort combined datasets with median improvement by using alanine scanning (AS).

    The DMS+AS combined dataset targeting each DMS dataset are selected, and the one with (closest to) median
    RMSE change by using alanine scanning data is picked. The final result is also sorted on median protein
    RMSE change for better visualization.

    Parameters
    ----------
    model_perform: pd.DataFrame
        Predictor performance while using AS data or not on each combined dataset with other dataset information.

    Returns
    -------
    med_dmsas: pd.DataFrame
        Similar as input with only picked datasets and sorted by protein RMSE change.
    """
    # Get datasets with median improvement by using alanine sacnning data.
    med_dmsas = pd.DataFrame()
    for dms, df in model_perform.groupby("dms_id", as_index=False):
        foo = df.copy()
        foo["med_dist"] = (foo["diff_rmse"].median() - foo["diff_rmse"]).abs()
        med_dmsas = med_dmsas.append(
            foo.sort_values("med_dist").iloc[0]
        )  # Closest to median.

    # Sort on protein level for better visualization.
    pro_med_diff = (
        med_dmsas[["diff_rmse", "uniprot_id"]]
        .groupby("uniprot_id")
        .median()["diff_rmse"]
    )
    med_dmsas["pro_med_diff"] = med_dmsas["uniprot_id"].map(pro_med_diff)
    med_dmsas.sort_values(["pro_med_diff", "diff_rmse"], inplace=True, ascending=False)
    return med_dmsas


def bar_median_data_count(med_dmsas, ax):
    """Make bar plot showing the size of picked datasets.

    Parameters
    ----------
    med_dmsas: pd.DataFrame
        Predictor performance while using AS data or not for datasets with median RMSE change.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    """
    ax.bar(np.arange(len(med_dmsas)), med_dmsas["size"])
    ax.set_ylabel("Num. of entries", fontsize=12)
    ax.set_yticks([0, 250, 500])
    ax.set_xticks([])
    for spine_name in ax.spines:
        if spine_name != "left":
            ax.spines[spine_name].set_visible(False)
    ax.set_xlim(-1, len(med_dmsas))
    return


def scatter_median_data_rmse(med_dmsas, ax):
    """Make scatter plot showing the prediction RMSE while using AS data or not for picked datasets.

    Parameters
    ----------
    med_dmsas: pd.DataFrame
        Predictor performance while using AS data or not for datasets with median RMSE change.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    """
    ax.scatter(
        np.arange(len(med_dmsas)),
        med_dmsas["noala_rmse"],
        label="No AS",
        s=25,
        marker="D",
        c=[(0.2, 0.4, 1)],
    )
    ax.scatter(
        np.arange(len(med_dmsas)),
        med_dmsas["ala_rmse"],
        s=50,
        label="With AS",
        c=[(1, 0.3, 0.3)],
    )
    ax.set_xticks(ticks=np.arange(len(med_dmsas)))
    ax.set_xticklabels(
        labels=med_dmsas["dms_name"], rotation=45, fontsize=12, ha="right"
    )
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_xlabel("DMS dataset", fontsize=12)
    # Matching lines.
    for i in range(len(med_dmsas)):
        ax.plot([i, i], [0, 0.77], "--", c=(0.9, 0.9, 0.9), zorder=0)
    ax.set_xlim(-1, len(med_dmsas))
    return


def underline_median_data_protein(med_dmsas, ax):
    """Draw lines at the bottom of scatter plot to group picked datasets with target proteins.

    Parameters
    ----------
    med_dmsas: pd.DataFrame
        Predictor performance while using AS data or not for datasets with median RMSE change.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    """
    length = []
    for uniprot_id, df in med_dmsas.groupby("uniprot_id", sort=False):
        length.append(len(df))
    start = -0.25
    for ext in length:
        ax.plot([start, start + ext - 0.5], [0, 0], linewidth=3)
        start += ext
    return


def create_heatmap_score_matrix(data, score_col, aa_order):
    """Create a matrix with experimental or predicted scores for making prediction heat map.

    Parameters
    ----------
    data: pd.DataFrame
        Mutants data with experimental and predicted scores and other mutational information. Normally, should
        contain data from only one dataset.
    score_col: str
        The column name of scores to be transformed into the score matrix.
    aa_order: list
        The amino acid list indicating the column order of the score matrix

    Returns
    -------
    matrix: pd.DataFrame
        A matrix with experimental or predicted scores for each mutation type (in columns) on each position
        (in rows).
    """
    foo = data[["pos_id", "u_pos", "dms_id", "aa1", "aa2"] + [score_col]].rename(
        columns={score_col: "score", "u_pos": "position"}
    )
    matrix = create_score_matrix(foo, handle_na=None, handle_wt=1)
    matrix = matrix.sort_values("position")[
        ["aa1", "position"] + [x + "_score" for x in aa_order]
    ]
    return matrix


def create_score_heatmap_on_axes(ax, data, vcenter=1, vmin=0, vmax=1.5, cmap="bwr"):
    """Create heat map on given axes.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    data: pd.DataFrame or np.array
        2 dimensional numeric data used to generate the heat map.
    vcenter: int or float, optional (default=1)
        The center value of the color bar.
    vmin: int or float, optional (default=0)
        The minimum value of the color bar.
    vmax: int or float, optional (default=1.5)
        The maximum value of the color bar.
    cmap: str, optional (default='bwr')
        Colormap used for heatmap.

    Return
    ------
    im: matplotlib.image.AxesImage
        The created image class. Can be used for creating color bar.
    """
    cmap = plt.cm.get_cmap(cmap, None)
    cmap.set_bad("gray")
    norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    im = ax.imshow(X=data, cmap=cmap, norm=norm)
    return im


def dot_score_heatmap_wt(axs, score_data, x_ticks):
    """Illustrate the wild type in the heat map with dots.

    Parameters
    ----------
    axs: list
        List of axes of the heat map.
    score_data: pd.DataFrame
        Data to generate the heat map.
    x_ticks: list or tuple
        The amino acid list or tuple which is the x ticks of the heat map.
    """
    x_value = [x_ticks.index(aa) for aa in score_data["aa1"]]
    y_value = np.arange(len(score_data))
    for i in range(len(axs)):
        axs[i].scatter(x_value, y_value, color=(0.5, 0.5, 0.5))
    return


def calc_aa_type_rmse(work_data, aa_type):
    """Calculate prediction RMSE while using AS data or not for each wildtype or mutation type amino acid.

    Parameters
    ----------
    work_data: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.
    aa_type: str
        Group mutants by wildtype or mutation type amino acids. Normally should be 'aa1' or 'aa2'.

    Returns
    -------
    aa_perf: pd.DataFrame
        The prediction RMSE while using AS data or not for each amino acid type, with amino acid property.
    """
    aa_perf = pd.DataFrame()
    for aa, df in work_data.groupby(aa_type, as_index=False):
        accuracy = pd.Series()
        accuracy["noala_rmse"] = np.sqrt(
            mean_squared_error(df["ob_score"], df["pred_score_noala"])
        )
        accuracy["ala_rmse"] = np.sqrt(
            mean_squared_error(df["ob_score"], df["pred_score_ala"])
        )
        accuracy["aa"] = aa
        aa_perf = aa_perf.append(accuracy, ignore_index=True)
    aa_perf["aa_property"] = aa_perf["aa"].map(AA_PROPERTY)
    aa_perf.sort_values(["aa_property", "aa"], inplace=True)
    return aa_perf


def get_dmsas_shared_mutants(pred_result, dms_as_pair):
    """Get the shared mutants and their predicted scores between the given combined datasets.

    Parameters
    ----------
    pred_result: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.
    dms_as_pair: list
        A list with two DMS+AS combined dataset IDs which are based on the same DMS data and will be compared.

    Returns
    -------
    share_mut: pd.DataFrame
        Contains the predicted scores while using alanine scanning data for the two combined datasets with
        UniProt residue position and real DMS scores.
    """
    work_data = pred_result[pred_result["dmsa_id"].isin(dms_as_pair)].copy()
    work_data["mutant"] = work_data["u_pos"].astype(str) + "_" + work_data["aa2"]
    share_mut = work_data.pivot(
        index="mutant", columns="dmsa_id", values="pred_score_ala"
    ).dropna()
    share_mut = share_mut.merge(
        work_data[["ob_score", "u_pos", "mutant"]].drop_duplicates(),
        how="left",
        left_index=True,
        right_on="mutant",
        validate="1:1",
    )
    return share_mut


def get_dmsas_pair_position_rmse(share_mut, dms_as_pair):
    """Calculate prediction RMSE while using AS data or not for each wildtype or mutation type amino acid.

    Parameters
    ----------
    share_mut: pd.DataFrame
        Contains the predicted scores while using alanine scanning data for the two combined datasets with
        UniProt residue position and real DMS scores.
    dms_as_pair: list
        A list with two DMS+AS combined dataset IDs which are based on the same DMS data and will be compared.

    Returns
    -------
    pos_rmse: pd.DataFrame
        RMSE while using alanine scanning data for shared residues in the given combined datasets. The result
        is sorted by UniProt residue positions.
    """
    pos_rmse = pd.DataFrame()
    for pos, df in share_mut.groupby("u_pos"):
        row = pd.Series()
        row["u_pos"] = pos
        for assay in dms_as_pair:
            row[assay] = np.sqrt(mean_squared_error(df[assay], df["ob_score"]))
        pos_rmse = pos_rmse.append(row, ignore_index=True)
    pos_rmse = pos_rmse.melt(
        id_vars="u_pos", value_vars=dms_as_pair, value_name="rmse", var_name="dmsa_id"
    )
    pos_rmse.sort_values("u_pos", inplace=True)
    return pos_rmse


def calc_structure_type_rmse_change(data, str_col):
    """Calculate relative RMSE change by using AS data for each types in the given structural feature.

    Parameters
    ----------
    data: pd.DataFrame
        Predicted results while using AS data or not with other mutant information.
    str_col: str
        Name of the structural feature used to group the mutants.

    Returns
    -------
    str_perf: pd.DataFrame
        The relative RMSE change by using AS data for each types (in rows) and DMS datasets (in columns).
    """
    str_perf = pd.DataFrame()
    for dms, dms_df in data.groupby("dms_name"):
        accuracy = pd.Series(name=dms)
        for str_type, str_df in dms_df.groupby(str_col):
            with_as = np.sqrt(
                mean_squared_error(str_df["ob_score"], str_df["pred_score_ala"])
            )
            without_as = np.sqrt(
                mean_squared_error(str_df["ob_score"], str_df["pred_score_noala"])
            )
            accuracy[str(str_type)] = (without_as - with_as) / without_as * 100
        str_perf = str_perf.append(accuracy, sort=False)
    return str_perf


def spineless_piyg_heatmap(
    ax, data, xticklabels, yticklabels, colour_range, show_x_ticks
):
    """Create heat map using PiYG color map with no spine and cells separated with minor ticks.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes to work on.
    data: pd.DataFrame
    xticklabels: list or list like variables
    yticklabels: list or list like variables
    colour_range: tuple
        Tuple with two elements indicating the range for the colour bar.
    show_x_ticks: bool
        If show x ticks or not.

    Return
    ------
    im: matplotlib.image.AxesImage
        The created image class. Can be used for creating color bar.
    """
    cmap = plt.cm.get_cmap("PiYG")
    cmap.set_bad("gray")
    norm = TwoSlopeNorm(0, colour_range[0], colour_range[1])
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect=0.5)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    if show_x_ticks:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=14)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=True)
    ax.set_yticklabels([""] + yticklabels, fontsize=14)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im
