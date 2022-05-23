import numpy as np
import pandas as pd
import GPyOpt
import pickle
import time
from sklearn.model_selection import cross_val_score


def _create_hp_dict(value_list, search_space):
    """Return a dictionary with hyper-parameter names as keys and values from the value_list.

    Parameters
    ----------
    search_space: list
        A list of dict indicating the range of hyper-parameters to be tuned. The keys of dict
        should include name, type (discrete or continuous) and domain (tuple of possible values
        or range). Details can be found in GPyOpt.methods.BayesianOptimization for the 'domain'
        parameter. This code also need another key, dtype, which indicates the type of tuned
        hyper-parameter.
        Example: [{'name':'hp1', 'type':'discrete', 'domain':(0., 0.5, 1.), 'dtype':float}]

    value_list: list, np.array or pd.Series
        A list-like variable contains the values for each hyper-parameter to be set to the
        estimator later. The order of the hyper-parameters are the same as self.search_space.

    Returns
    -------
    hp_dict: dict
    """
    hp_dict = dict()
    for i in range(len(search_space)):
        name = search_space[i]["name"]
        dtype = search_space[i]["dtype"]
        if dtype == int:
            # Force the data type back to int which may have been changed by GPyOpt.
            hp_dict[name] = int(value_list[i])
        else:
            hp_dict[name] = value_list[i]
    return hp_dict


def _create_evaluation_function(search_space, estimator, trainx, trainy, cv_kwargs):
    """Return a function which evaluates the model trained by given hyper-parameter by cross-validation.

    Parameters
    ----------
    search_space: list
        A list of dict indicating the range of hyper-parameters to be tuned. The keys of dict
        should include name, type (discrete or continuous) and domain (tuple of possible values
        or range). Details can be found in GPyOpt.methods.BayesianOptimization for the 'domain'
        parameter. This code also need another key, dtype, which indicates the type of tuned
        hyper-parameter.
        Example: [{'name':'hp1', 'type':'discrete', 'domain':(0., 0.5, 1.), 'dtype':float}]

    estimator: estimator object
        The estimator to be fitted.

    trainx: pd.DataFrame, list, np.array or None
        Feature matrix of training data.

    trainy: pd.Series, list, np.array or None
        Experimental score values for training data.

    cv_kwargs: dict
        A dictionary indicating the parameter values to be put into sklearn.model_selection_cross_val_score.
        Keys usually include: cv, scoring, group, n_jobs and fit_params, which can be used to identify
        sample_weight.

    Returns
    -------
    evaluate_hyperparameters: function
        A function to calculate cross-validation evaluate the model trained using given training data and
        model information.
        This function takes a two dimensional array whose first dimension represents the input iteration
        batch. And the second dimension shows the selected hyper-parameter values in the same order as
        search_space.
        It returns a 2D array saving the validation y values (each as an array) trained by current input
        hyper-parameters for each iteration.
    """

    def evaluate_hyperparameters(hyperparam):
        """Evaluate the model trained by input hyper-parameters.

        Parameters
        ----------
        hyperparam: np.array
            It is a two dimensional array whose first dimension represents the input iteration batch.
            And the second dimension represents the selected hyper-parameter values in the same order as
            search_space.

        Returns
        -------
        scores: np.array
            It is a 2D array saving the validation y values (each as an array) trained by current
            input hyper-parameters for each iteration.
        """
        scores = []
        for hp in hyperparam:
            hp_dict = _create_hp_dict(hp, search_space)
            estimator.set_params(**hp_dict)
            # Fit and calculate the mean score with cross-validation using the input hyper-parameters.
            mean_cv_score = cross_val_score(
                estimator, trainx, trainy, **cv_kwargs
            ).mean()
            scores.append([mean_cv_score])
        return np.array(scores)

    return evaluate_hyperparameters


def _tune_by_gpopt(
    search_space,
    eval_hp_func,
    num_iterations,
    num_cores,
    if_maximize,
    output_header=None,
    random_seed=None,
    max_time=np.inf,
    eps=1e-8,
):
    """Find the best hyper-parameters within given range using GPyOpt Bayesian optimization.

    Parameters
    ----------
    search_space: list
        A list of dict indicating the range of hyper-parameters to be tuned. The keys of dict
        should include name, type (discrete or continuous) and domain (tuple of possible values
        or range). Details can be found in GPyOpt.methods.BayesianOptimization for the 'domain'
        parameter. This code also need another key, dtype, which indicates the type of tuned
        hyper-parameter.
        Example: [{'name':'hp1', 'type':'discrete', 'domain':(0., 0.5, 1.), 'dtype':float}]

    eval_hp_func: function
        A function to calculate cross-validation prediction error of the model trained using given training
        data and model information. It is the output of _create_evaluation_function.

    num_iterations: int
        Exploration horizon, or number of acquisitions.

    num_cores: int
        Number of cores used to evaluate the model.

    if_maximize: bool
        If true, the model is trying to maximize the value of evaluation output. The metric for evaluation
        is determined by 'scoring' in cv_kwargs in eval_kwargs.
        Comment: GPyOpt handle this parameter using a trick. Its default setting is to minimize the observation
        function as if it's measured by prediction error. While trying to maximize it instead, the package
        split get the opposite value of the function output it still try to minimize it in essence.
        Unfortunately, the reporting files are recording these flipped value which makes it quite hard
        to interpret the maximized results.

    output_header: str, optional(default=None)
        If it is not None, it indicates the directory and file name prefix for the reporting files.

    random_seed: None or int, optional(default=None)
        Seed of random state which will be put into np.random.seed. It will determine the random initial
        searching points for Bayesian optimization.

    max_time: int or np.inf, optional(default=np.inf)
        Maximum exploration horizon in seconds.

    eps: float, optional(default=1e-8)
        Minimum distance between two consecutive hyper-parameters to stop.

    Returns
    -------
    tuned_hp: dict
        A dict whose keys are the hyper-parameters tuned and values are the optimized values.
    """
    np.random.seed(random_seed)
    # An initial 5 point running will be done.
    gpyopt_bo = GPyOpt.methods.BayesianOptimization(
        f=eval_hp_func,
        domain=search_space,
        num_cores=num_cores,
        maximize=if_maximize,
    )
    rf, ef = None, None
    if output_header is not None:
        rf = output_header + "report.txt"
        ef = output_header + "evaluation.txt"
    # Extra running.
    gpyopt_bo.run_optimization(
        max_iter=num_iterations,
        report_file=rf,
        evaluations_file=ef,
        max_time=max_time,
        eps=eps,
    )
    if output_header is not None:
        gpyopt_bo.plot_convergence(filename=output_header + "converg.png")
    tuned_hp = _create_hp_dict(gpyopt_bo.x_opt, search_space)
    return tuned_hp


def fit_best_estimator(
    search_space,
    estimator,
    trainx,
    trainy,
    cv_kwargs,
    bo_kwargs,
):
    """Fit training data to the estimator with best hyper-parameter found by Bayesian optimization tuning.

    Parameters
    ----------
    search_space: list
        A list of dict indicating the range of hyper-parameters to be tuned. The keys of dict
        should include name, type (discrete or continuous) and domain (tuple of possible values
        or range). Details can be found in GPyOpt.methods.BayesianOptimization for the 'domain'
        parameter. This code also need another key, dtype, which indicates the type of tuned
        hyper-parameter.
        Example: [{'name':'hp1', 'type':'discrete', 'domain':(0., 0.5, 1.), 'dtype':float}]

    estimator: estimator object
        The estimator to be fitted.

    trainx: pd.DataFrame, list, np.array or None
        Feature matrix of training data.

    trainy: pd.Series, list, np.array or None
        Experimental score values for training data.

    cv_kwargs: dict
        A dictionary indicating the parameter values to be put into sklearn.model_selection.cross_val_score.
        Keys usually include: cv, scoring, group, n_jobs and fit_params, which can be used to identify
        sample_weight.

    bo_kwargs: dict
        Keyword arguments to be used for Bayesian optimization tuning. It is a dict with output_header,
        num_cores, if_maximize, num_iterations, random_seed, max_time, etc., as keys. Details can be found
        in corresponding tuning functions.

    Returns
    -------
    best_estimator: estimator object
        The estimator with tuned hyper-parameters and fit with all trianing data.
    """
    eval_hp_func = _create_evaluation_function(
        search_space, estimator, trainx, trainy, cv_kwargs
    )
    tuned_hp = _tune_by_gpopt(search_space, eval_hp_func, **bo_kwargs)
    best_estimator = estimator.set_params(**tuned_hp).fit(
        trainx, trainy, cv_kwargs["fit_params"]["sample_weight"]
    )
    return best_estimator


def save_predictor(predictor, output_header):
    """Save the predictor itself.

    Parameters
    ----------
    predictor: estimator object
    output_header: str
        It indicates the directory and file name prefix of the output file.
    """
    with open(f"{output_header}models.pickle", "wb") as file:
        pickle.dump(predictor, file)
    return


def save_feature_importance(predictor, output_header):
    """Save the feature_importances_ values of the predictor.

    Parameters
    ----------
    predictor: estimator object
    output_header: str
        It indicates the directory and file name prefix of the output file.
    """
    with open(f"{output_header}feature_param.pickle", "wb") as file:
        pickle.dump(predictor.feature_importances_, file)
    return


def save_compared_prediction(
    predictor,
    mut_data,
    features,
    y_col_name,
    output_header,
    if_train=False,
    info_col=["dmsa_id", "position", "aa2"],
):
    """Save a DataFrame of predicted and observed scores and some other information for each mutant.

    Parameters
    ----------
    predictor: estimator object

    mut_data: pd.DataFrame
        The input mutants for prediction which should contain all the features and observed scores.

    features: list
        The name of the columns whose values will be used as features.

    y_col_name: str
        The column name of DMS scores.

    output_header: str
        It indicates the directory and file name prefix of the output file.

    if_train: bool, optional(default=False)
        If these input mutants were used as training data.

    info_col: list, optional(default=["dms_id", "position", "aa2"])
        Information columns of mutant to be recorded.
    """
    # Index of observed scores are maintained for concatenation.
    result = mut_data[y_col_name].to_frame(name="ob_score")
    result["pred_score"] = predictor.predict(mut_data[features])
    result["if_train"] = if_train
    mut_info = mut_data[info_col]
    mut_pred = pd.concat([mut_info, result], axis=1, sort=False)
    mut_pred.to_csv(f"{output_header}prediction.csv")
    return


def save_tuned_hyperparameters(predictor, search_space, output_header):
    """Save the names of tuned hyper-parameter and their optimized values.

    Parameters
    ----------
    predictor: estimator object

    search_space: list
        A list of dict indicating the range of hyper-parameters to be tuned. The keys of dict
        should include name, type (discrete or continuous) and domain (tuple of possible values
        or range). Details can be found in GPyOpt.methods.BayesianOptimization for the 'domain'
        parameter. This code also need another key, dtype, which indicates the type of tuned
        hyper-parameter.
        Example: [{'name':'hp1', 'type':'discrete', 'domain':(0., 0.5, 1.), 'dtype':float}]

    output_header: str
        It indicates the directory and file name prefix of the output file.
    """
    tuned_params = [x["name"] for x in search_space]
    tuned_df = pd.DataFrame(predictor.get_params(), index=[0])[tuned_params]
    tuned_df.to_csv(f"{output_header}tuned_result.csv")
    return


def monitor_process(directory, log_text, indention):
    """Monitor process by writing log files.

    Parameters
    ----------
    directory: str
        The directory of the log file.
        Example: './log/'

    log_text: str
        Context of the log.

    indention: int
        Indention of the log message.
    """
    present = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with open(f"{directory}log.txt", "a+") as file:
        file.write(f"{' ' * 2 * indention}{log_text} at: {present} (UTC)\n")
    return


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
    pure_dms = train_data.groupby(['dms_id', 'position', 'aa2']).first()
    matrix_map = pure_dms.groupby('sub_type')['score'].mean()
    train_data_refit = train_data.copy()
    train_data_refit['matrix'] = train_data_refit['sub_type'].map(matrix_map)
    test_data_refit = test_data.copy()
    test_data_refit['matrix'] = test_data_refit['sub_type'].map(matrix_map)
    return train_data_refit, test_data_refit
