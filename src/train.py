"""Helper functions used while model training."""

import pandas as pd


def add_training_weight(input_data):
    """Add training weight for protein variant data.

    Variants were inversely weighted by the number of measurements available (number of rows) in
    the input_data.

    Parameters
    ----------
    input_data: pd.DataFrame
        Protein variant data with columns: 'uniprot_id' (UniProtKB ID), 'u_pos' (UniProt residue
        position) and 'aa2' (variant type).

    Returns
    -------
    weighted_data: pd.DataFrame
        Protein variant data with an extra column 'weight' indicating training weight.
    """
    weight = input_data.groupby(["uniprot_id", "u_pos", "aa2"])["score"].count()
    weight = 1 / weight
    weight.name = "weight"
    weighted_data = pd.merge(
        input_data,
        weight,
        left_on=["uniprot_id", "u_pos", "aa2"],
        right_index=True,
        how="outer",
        validate="m:1",
    ).reset_index(drop=True)
    return weighted_data


def refit_matrix_score(train_data, test_data):
    """Calculate substitution matrix score in DeMaSk from training data and fit them to modelling data.

    Parameters
    ----------
    train_data: pd.DataFrame
        Training data which are used to calculate substitution matrix score.
    test_data: pd.DataFrame
        Testing data.

    Returns
    -------
    train_data_refit: pd.DataFrame
        Training data with recalculated matrix feature values.
    test_data_refit: pd.DataFrame
        Testing data with recalculated matrix feature values.
    """
    # Aviod duplications because of multiple alanine scanning.
    pure_dms = train_data.groupby(["dms_id", "position", "aa2"]).first()
    matrix_map = pure_dms.groupby("sub_type")["score"].mean(numeric_only=True)
    train_data_refit = train_data.copy()
    train_data_refit["matrix"] = train_data_refit["sub_type"].map(matrix_map)
    test_data_refit = test_data.copy()
    test_data_refit["matrix"] = test_data_refit["sub_type"].map(matrix_map)
    return train_data_refit, test_data_refit
