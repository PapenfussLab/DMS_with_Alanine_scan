import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


TESTED_PROTEINS = [
    "P38398",
    "P0CG63",
    "P06654",
    "P63279",
    "P63165",
    "P35520",
    "P04386",
    "P60484",
    "P05067",
    "Q9BYF1",
    "Q9GZX7",
    "P61073",
    "P02724",
    "P05412",
    "P04049",
    "P01112",
    "P04637",
    "P40238",
]


def calculate_score_feature_correlation(score_feature, features, score_col, set_col):
    """Calculate the spearman correlation between scores and features for each dataset.

    Parameters
    ----------
    score_feature: pd.DataFrame
        Contains score and feature values with dataset identifier.
    features: list
        Features used during training and testing in the same order of the feature importance data.
    score_col: str
        The name of column contains score values. Normally should be 'score' (DMS score) or 'AS_score'
        (alanine scanning score).
    set_col: str
        The name of column contains dataset identifier. Normally should be 'dms_name' (DMS name) or
        'Ascan_id' (alanine scanning ID).

    Returns
    -------
    feat_corr: pd.DataFrame
        Correlation between scores and features (in columns) for each dataset (in rows). Columns are
        ordered by median correlation coefficient.
    """
    feat_corr = pd.DataFrame()
    for dataset, df in score_feature.groupby(set_col):
        row = pd.Series()
        row.name = dataset
        for feat in features:
            row[feat] = spearmanr(df[score_col], df[feat])[0]
        feat_corr = feat_corr.append(row)
    sort_feat = feat_corr.median().sort_values().index
    feat_corr = feat_corr[sort_feat]
    return feat_corr


def _read_protein_result(data_dir, uniprot_id):
    """Read the prediction result for individual protein.

    Parameters
    ----------
    data_dir: str
        Indicate the directory where the prediction results are.

    Returns
    -------
    protein_result: pd.DataFrame
        Predicted results while using alanine scanning (AS) data or not with other mutant information.
    """
    ala_result = pd.read_csv(
        f"{data_dir}{uniprot_id}_with_ala_0_test_prediction.csv", index_col=0
    )
    noala_result = pd.read_csv(
        f"{data_dir}{uniprot_id}_nothing_0_test_prediction.csv", index_col=0
    )
    protein_result = pd.merge(
        ala_result,
        noala_result,
        on=["dmsa_id", "position", "aa2", "ob_score"],
        validate="1:1",
        suffixes=["_ala", "_noala"],
    )[["dmsa_id", "position", "aa2", "ob_score", "pred_score_ala", "pred_score_noala"]]
    return protein_result


def combine_prediction_result(data_dir, target_proteins=None):
    """Read and combine prediction results.

    Parameters
    ----------
    data_dir: str
        Indicate the directory where the prediction results are.
    target_proteins: list or None, optional (default=None)
        The proteins' UniProt ID to be read. None for all tested proteins.

    Returns
    -------
    result: pd.DataFrame
        Predicted results while using alanine scanning (AS) data or not with other mutant information.
    """
    result = pd.DataFrame()
    if target_proteins is None:
        target_proteins = TESTED_PROTEINS
    for uniprot_id in target_proteins:
        protein_result = _read_protein_result(data_dir, uniprot_id)
        result = result.append(protein_result)
    return result


def calculate_model_performance(pred_result):
    """Calculate the predictor performance for each DMS+AS combined dataset for using AS data or not.

    Parameters
    ----------
    pred_result: pd.DataFrame
        Output of combine_prediction_result.

    Returns
    -------
    model_perform: pd.DataFrame
        Predictor performance while using AS data or not on each combined dataset with other dataset
        information.
    """
    model_perform = pd.DataFrame()
    for dmsa, df in pred_result.groupby("dmsa_id", as_index=False):
        accuracy = pd.Series()
        accuracy["noala_rmse"] = np.sqrt(
            mean_squared_error(df["ob_score"], df["pred_score_noala"])
        )
        accuracy["ala_rmse"] = np.sqrt(
            mean_squared_error(df["ob_score"], df["pred_score_ala"])
        )
        accuracy["diff_rmse"] = accuracy["noala_rmse"] - accuracy["ala_rmse"]
        accuracy["size"] = len(df)
        accuracy = accuracy.append(
            df[
                [
                    "uniprot_id",
                    "protein_name",
                    "dms_id",
                    "dms_name",
                    "Ascan_id",
                    "dmsa_id",
                ]
            ].iloc[0]
        )
        model_perform = model_perform.append(accuracy, ignore_index=True)
    return model_perform


def read_feature_importance(data_dir, features, model, target_proteins=None):
    """Read and combine feature importance for given model (predictor).

    Parameters
    ----------
    data_dir: str
        Indicate the directory where the feature importance data are.
    features: list
        Features used during training and testing in the same order of the feature importance data.
    model: 'nothing' or 'with_Ala'
        It's part of the file name which indicates whether the predictor is using alanine scanning data
        or not.
    target_proteins: list or None, optional (default=None)
        The proteins' UniProt ID to be read. None for all tested proteins.

    Returns
    -------
    importance: pd.DataFrame
        Importance for each feature (in columns) and tested protein (in rows).
    """
    importance = pd.DataFrame()
    if target_proteins is None:
        target_proteins = TESTED_PROTEINS
    for unip in target_proteins:
        with open(f"{data_dir}{unip}_{model}_0_feature_param.pickle", "rb") as file:
            array = pickle.load(file)
        pro_imp = pd.Series(data=array, index=features)
        pro_imp.name = unip
        importance = importance.append(pro_imp)
    return importance
