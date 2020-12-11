import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import sys

sys.path.append("/data/gpfs/projects/punim0860/imputation/project_modules")
import models as models


train_info = ["P38398", 0, True]
test_unip = train_info[0]
test_rep = train_info[1]
test_with_ala = train_info[2]

# Files reading.
dms_ascan = pd.read_csv(
    "/data/gpfs/projects/punim0860/imputation/data/dms_ascan_20200918.csv", index_col=0
)
with open(
    "/data/gpfs/projects/punim0860/imputation/data/feature_dictionary.txt", "r"
) as file:
    feature = eval(file.read())["Envision"]

# Data preprocessing.
dms_ascan = dms_ascan[dms_ascan["uniprot_id"] != "P51681"]  # Remove CCR5 data.
unava_data = dms_ascan[dms_ascan["Ascan_id"].isna()]  # Mutants without AS data.
dms_ascan = dms_ascan[~dms_ascan["Ascan_id"].isna()].reset_index(drop=True)
unava_data["AS_score"] = dms_ascan["AS_score"].mean()
# Set training weight.
unip_weight = (
    dms_ascan[["uniprot_id", "dms_id"]].groupby("uniprot_id").nunique()["dms_id"]
)
unip_weight = 1 / unip_weight
dms_ascan["weight"] = dms_ascan["uniprot_id"].map(unip_weight)
# Separate training and testing data.
train_data = dms_ascan[dms_ascan["uniprot_id"] != test_unip]
test_data = dms_ascan[dms_ascan["uniprot_id"] == test_unip]
unava_test = unava_data[unava_data["uniprot_id"] == test_unip]

# Parameters setting.
search_space = [
    {
        "name": "n_estimators",
        "type": "discrete",
        "domain": np.arange(1, 501),
        "dtype": int,
    },
    {
        "name": "max_depth",
        "type": "discrete",
        "domain": np.arange(1, 101),
        "dtype": int,
    },
    {
        "name": "min_samples_split",
        "type": "discrete",
        "domain": np.arange(2, 201),
        "dtype": int,
    },
    {
        "name": "min_samples_leaf",
        "type": "discrete",
        "domain": np.arange(1, 101),
        "dtype": int,
    },
    {
        "name": "min_weight_fraction_leaf",
        "type": "continuous",
        "domain": (0, 0.5),
        "dtype": float,
    },
    {
        "name": "max_leaf_nodes",
        "type": "discrete",
        "domain": np.arange(2, 101),
        "dtype": int,
    },
]
# np.random.seed(0)
estimator = RandomForestRegressor(n_jobs=1)

# Parameters preprocessing.
if test_with_ala:
    model_feature = feature + ["AS_score", "Ascan_score_avail"]
    model_name = "with_Ala"
else:
    model_feature = feature.copy()
    model_name = "nothing"
output_header = f"./log/{test_unip}_{model_name}_{test_rep}_"
bo_kwargs = {
    "num_iterations": 1000,
    "num_cores": 8,
    "if_maximize": True,
    "output_header": output_header,
}
cv_kwargs = {"scoring": "neg_mean_squared_error", "cv": GroupKFold(5), "n_jobs": 1}
cv_kwargs["groups"] = train_data["uniprot_id"]
cv_kwargs["fit_params"] = {"sample_weight": train_data["weight"]}

# Training.
models.monitor_process("./log/", f"{output_header[6:-1]} starts.", 0)
predictor = models.fit_best_estimator(
    search_space,
    estimator,
    train_data[model_feature],
    train_data["score"],
    cv_kwargs,
    bo_kwargs,
)
models.save_feature_importance(predictor, output_header)
models.save_tuned_hyperparameters(predictor, search_space, output_header)
models.save_compared_prediction(
    predictor, test_data, model_feature, "score", output_header + "test_"
)

# Evaluation on mutants without AS data.
models.save_compared_prediction(
    predictor, unava_test, model_feature, "score", output_header + "unava_"
)
models.monitor_process("./log/", f"{output_header[6:-1]} ends.", 0)
