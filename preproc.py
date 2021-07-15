import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def _impute_missing_value(data, how):
    """Impute the missing values in the input data.

    Parameters
    ----------
    data: pd.DataFrame
        Data with missing values.
    how: str, 'mean' or 'most_frequent'
        Way to impute the missing values, by either mean or most frequent value.

    Returns
    -------
    impute_result: pd.DataFrame
    """
    columns = data.columns
    impute_method = SimpleImputer(strategy=how)
    impute_result = pd.DataFrame(impute_method.fit_transform(data))
    impute_result.columns = columns
    return impute_result


def _encode_categorical_feature(data, encoder_file=None):
    """Transform categorical features into one hot sparse matrix.

    The function will change the input columns to a one hot sparse DataFrame whose columns
    are the possible values of those features and 1.0 will be assigned to the rows with
    corresponding value otherwise being 0.

    Parameters
    ----------
    data: pd.DataFrame
        Data whose columns are all categorical features to be transformed.
    encoder_file: str or None, optional (default=None)
        If a str is passed, it should be the filename of an OneHotEncoder and the function will use it.
        Otherwise the function will create and use a new OneHotEncoder object.

    Returns
    -------
    encode_result: pd.DataFrame

    Example
    -------
    >>> _encode_categorical_feature(pd.DataFrame({'one':['a', 'b'], 'two': ['A', 'B']}))
           one_a  one_b  two_A  two_B
    0    1.0    0.0    1.0    0.0
    1    0.0    1.0    0.0    1.0
    """
    if encoder_file:  # Encode with existing encoder.
        with open(encoder_file, "rb") as file:
            encoder = pickle.load(file)
        encode_result = pd.DataFrame(encoder.fit_transform(data))
        encode_result.columns = encoder.get_feature_names(data.columns)

    else:  # encoder_dir is None, the encoder will be disgarded.
        encoder = OneHotEncoder(sparse=False)
        encode_result = pd.DataFrame(encoder.fit_transform(data))
        encode_result.columns = encoder.get_feature_names(data.columns)

    return encode_result


def impute_encode_features(
    data, categ_feat, numer_feat, retain_col=None, encoder_file=None
):
    """Impute the input features and encode the categorical features with one hot sparse matrix.

    Parameters
    ----------
    data: pd.DataFrame
        Data with columns to be imputed and encoded.
    categ_feat: list
        List of categorical features which will be imputed by most frequent value and encoded
        with one hot encoder.
    numer_feat: list
        List of numerical features which will be imputed by mean value.
    retain_col: list, optional (default=None)
        If it is not None, the according columns in data will be retained in clean_data. This is
        supposed to be used on categorical features, keeping both the encoded and original features.
    encoder_file: str or None, optional (default=None)
        If a str is passed, it should be the filename of an OneHotEncoder and the function will use it.
        Otherwise the function will create and use a new OneHotEncoder object.

    Returns
    -------
    clean_data: pd.DataFrame
    list:
        A list of numerical and encoded categorical feature names.
    """
    work_data = data.copy().reset_index(drop=True)
    work_data[numer_feat] = _impute_missing_value(work_data[numer_feat], "mean")
    work_data[categ_feat] = _impute_missing_value(
        work_data[categ_feat], "most_frequent"
    )
    encode_result = _encode_categorical_feature(work_data[categ_feat], encoder_file)
    clean_data = pd.concat([work_data.drop(categ_feat, axis=1), encode_result], axis=1)
    clean_data.index = data.index
    if retain_col:
        clean_data[retain_col] = data[retain_col]
    return clean_data
