import numpy as np
import pandas as pd


AMINO_ACIDS = (
    "A",
    "V",
    "I",
    "L",
    "M",
    "F",
    "Y",
    "W",
    "S",
    "T",
    "N",
    "Q",
    "C",
    "G",
    "P",
    "R",
    "H",
    "K",
    "D",
    "E",
)


def _create_wt_matrix(work_data, wt_score):
    """Create a wild type matrix in the form of dms_data.

    Parameters
    ----------
    work_data: pd.DataFrame
    wt_score: float
        Score to be assigned to wild type.

    Returns
    -------
    wt_matrix: pd.DataFrame
        The columns are pos_id, aa2 and score. pos_id are the position indices. aa2
        are the wild types of corresponding pos_id which is used as fake mutation
        types. 'score' are all set according to wt_score.
    """
    wt_matrix = work_data[["pos_id", "aa1"]].drop_duplicates()
    wt_matrix.columns = ["pos_id", "aa2"]  # Set wild type as fake mutation type.
    wt_matrix["score"] = wt_score
    return wt_matrix


def _handle_wild_type(work_data, handle_wt):
    """Create a mixed DataFrame with original dms_data and handled wild type data.

    Parameters
    ----------
    work_data: pd.DataFrame
    handle_wt: int, float, None or 'data'
        Way to handle wild type. If it is int or float, wild type score will be assigned
        by this value. If it is None, wild type score will be assigned with NA which could
        possibly be further changed by imputing process. If it is 'data', wild type score
        will be assigned by synonymous mutations in work_data where 'aa1' and 'aa2' are
        the same. We note that, if this value is not 'data' and work_data does contain
        synonymous mutations. an error will be raised.

    Returns
    -------
    mixed_data: pd.DataFrame
        The columns are pos_id, aa2 and score.
    """
    if handle_wt == "data":
        mixed_data = work_data.copy()
    elif np.any(work_data["aa1"] == work_data["aa2"]):
        raise ValueError("Input data contains synonymous mutation!")
    elif handle_wt is None:
        mixed_data = work_data.copy()
    elif isinstance(handle_wt, (int, float)):  # If the type is int or float.
        mixed_data = work_data.append(
            _create_wt_matrix(work_data, handle_wt), sort=False
        )
    else:
        raise AttributeError("Invalid value for handle_wt.")
    return mixed_data


def _handle_na_values(score_matrix, handle_na):
    """Handle NA values in the score matrix.

    Parameters
    ----------
    score_matrix: pd.DataFrame
        A pd.DataFrame whose rows are distinct positions with all the input amino acids as
        wild or mutation type and columns are the variant effect scores of those amino acids.
    handle_na: {'drop', 'row_mean', None}
        Way to handle missing values. If it is 'drop', the rows with NA values will be dropped.
        If it is 'row_mean', missing values will be imputed by positional mean value. If it is
        None, the missing values will keep unchanged.

    Returns
    -------
    score_matrix: pd.DataFrame
        score_matrix with NA values handled.
    """
    if handle_na == "drop":
        score_matrix = score_matrix.dropna()
    elif handle_na == "row_mean":
        # Can only fill na by column.
        score_matrix = score_matrix.T.fillna(score_matrix.mean(axis=1), axis=0).T
    elif handle_na is not None:
        raise AttributeError("Invalid value for handle_nan.")
    return score_matrix


def create_score_matrix(
    dms_data, picked_aa=AMINO_ACIDS, handle_na="drop", handle_wt=1.0
):
    """Create the score_matrix.

    Return a score_matrix whose rows are distinct positions with all the input amino acids as
    wild or mutation type and columns are the variant effect scores of those amino acids as well
    as some extra information which are: dms_id, position and aa1. The index is the pos_id.

    Parameters
    ----------
    dms_data: pd.DataFrame

    picked_aa: list or tuple, optional (default=AMINO_ACIDS)

    handle_na: {'drop', 'row_mean', None}, optional (default='drop')
        Way to handle missing values. If it is 'drop', the rows with NA values will be dropped.
        If it is 'row_mean', missing values will be imputed by positional mean value. If it is
        None, the missing values will keep unchanged.

    handle_wt: int, float, None or 'data', optional (default=1.0)
        Way to handle wild type. If it is int or float, wild type score will be assigned
        by this value. If it is None, wild type score will be assigned with NA which could
        possibly be further changed by imputing process. If it is 'data', wild type score
        will be assigned by synonymous mutations in work_data where 'aa1' and 'aa2' are
        the same. We note that, if this value is not 'data' and work_data does contain
        synonymous mutations. an error will be raised.

    Returns
    -------
    score_matrix: pd.DataFrame
    """
    work_data = dms_data[dms_data["aa2"].isin(picked_aa)]
    mixed_data = _handle_wild_type(work_data, handle_wt)
    score_matrix = mixed_data.pivot(index="pos_id", columns="aa2", values="score")

    # Reset column order and names.
    score_matrix = score_matrix.reindex(
        columns=list(picked_aa)
    )  # Set np.nan if column is missing.
    score_matrix.columns = [x + "_score" for x in score_matrix.columns]

    score_matrix = _handle_na_values(score_matrix, handle_na)

    # Add information columns to score matrix and change the order of columns
    info_col = work_data.loc[
        :, ["pos_id", "dms_id", "position", "aa1"]
    ].drop_duplicates()
    info_col.index = info_col["pos_id"]
    score_matrix = pd.merge(
        info_col.loc[:, ["dms_id", "position", "aa1"]],
        score_matrix,
        left_index=True,
        right_index=True,
        how="right",
        validate="1:1",
    )
    return score_matrix
