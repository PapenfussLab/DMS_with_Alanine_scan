"""Testing code."""

import preproc as preproc
import train as train
import analysis as ana
import tune as tune
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
import unittest


class TestPreprocessing(unittest.TestCase):
    def test_01_impute_missing_value_1(self):
        categ_data = pd.DataFrame(
            {"1": ["a", "b", np.nan, "b", np.nan], "2": ["B", np.nan, "B", "A", "B"]}
        )
        result = preproc.impute_missing_value(categ_data, ["1", "2"], None)
        expect = pd.DataFrame(
            {"1": ["a", "b", "b", "b", "b"], "2": ["B", "B", "B", "A", "B"]}
        )
        assert_frame_equal(result, expect)

    def test_02_impute_missing_value_2(self):
        numer_data = pd.DataFrame(
            {"a": [1, 2, np.nan, 3, np.nan], "b": [2.3, np.nan, 0.4, 1.4, 2.3]}
        )
        result = preproc.impute_missing_value(numer_data, None, ["b", "a"])
        expect = pd.DataFrame(
            {"a": [1.0, 2.0, 2.0, 3.0, 2.0], "b": [2.3, 1.6, 0.4, 1.4, 2.3]}
        )
        assert_frame_equal(result, expect)

    def test_03_impute_missing_value_3(self):
        mix_data = pd.DataFrame(
            {
                "1": ["a", "b", np.nan, "a", np.nan],
                "a": [1, 2, np.nan, 3, np.nan],
                "b": [2.3, np.nan, 0.4, 1.4, 2.3],
                "2": ["B", np.nan, "B", "A", "B"],
            },
            index=[1, 2, 3, 4, 0],
        )
        result = preproc.impute_missing_value(mix_data, ["2", "1"], ["a", "b"])
        expect = pd.DataFrame(
            {
                "1": ["a", "b", "a", "a", "a"],
                "a": [1.0, 2.0, 2.0, 3.0, 2.0],
                "b": [2.3, 1.6, 0.4, 1.4, 2.3],
                "2": ["B", "B", "B", "A", "B"],
            },
            index=[1, 2, 3, 4, 0],
        )
        assert_frame_equal(result, expect)

    def test_04_encode_categorical_feature_1(self):
        categ_data = pd.DataFrame(
            {"1": ["a", "b", "b", "b", "b"], "2": ["B", "B", "B", "A", "B"]},
            index=[1, 2, 3, 4, 0],
        )
        result, encoded_col = preproc.encode_categorical_feature(categ_data, ["1", "2"])
        expect = pd.DataFrame(
            {
                "1_a": [1.0, 0.0, 0.0, 0.0, 0.0],
                "1_b": [0.0, 1.0, 1.0, 1.0, 1.0],
                "2_A": [0.0, 0.0, 0.0, 1.0, 0.0],
                "2_B": [1.0, 1.0, 1.0, 0.0, 1.0],
            },
            index=[1, 2, 3, 4, 0],
        )
        assert_frame_equal(result, expect)
        self.assertEqual(encoded_col, ["1_a", "1_b", "2_A", "2_B"])

    def test_05_encode_categorical_feature_2(self):
        mix_data = pd.DataFrame(
            {
                "1": ["a", "b", "b", "b", "b"],
                "a": [1, 2, 2, 3, 2],
                "b": [2.3, 1.6, 0.4, 1.4, 2.3],
                "2": ["B", "B", "B", "A", "B"],
            },
            index=[1, 2, 3, 4, 0],
        )
        result, encoded_col = preproc.encode_categorical_feature(
            mix_data, ["2", "1"], ["1", "2"]
        )
        expect = pd.DataFrame(
            {
                "a": [1, 2, 2, 3, 2],
                "b": [2.3, 1.6, 0.4, 1.4, 2.3],
                "2_A": [0.0, 0.0, 0.0, 1.0, 0.0],
                "2_B": [1.0, 1.0, 1.0, 0.0, 1.0],
                "1_a": [1.0, 0.0, 0.0, 0.0, 0.0],
                "1_b": [0.0, 1.0, 1.0, 1.0, 1.0],
                "1": ["a", "b", "b", "b", "b"],
                "2": ["B", "B", "B", "A", "B"],
            },
            index=[1, 2, 3, 4, 0],
        )
        assert_frame_equal(result, expect)
        self.assertEqual(encoded_col, ["2_A", "2_B", "1_a", "1_b"])


class TestTraining(unittest.TestCase):
    def test_01_add_training_weight(self):
        model_data = pd.DataFrame(
            {
                "uniprot_id": ["a", "b", "a", "a", "a"],
                "u_pos": [1, 2, 2, 3, 1],
                "aa2": ["A", "C", "C", "A", "A"],
                "score": [0.5, 0.4, 0.7, 0.8, 1.0],
            },
            index=[1, 7, 3, 4, 0],
        )
        result = train.add_training_weight(model_data)
        expect = pd.DataFrame(
            {
                "uniprot_id": ["a", "a", "b", "a", "a"],
                "u_pos": [1, 1, 2, 2, 3],
                "aa2": ["A", "A", "C", "C", "A"],
                "score": [0.5, 1.0, 0.4, 0.7, 0.8],
                "weight": [0.5, 0.5, 1.0, 1.0, 1.0],
            },
            index=[0, 1, 2, 3, 4],
        )
        assert_frame_equal(result, expect)

    def test_02_refit_matrix_score(self):
        train_data = pd.DataFrame(
            {
                "dms_id": ["dms1"] * 4 + ["dms2"],
                "position": [2, 2, 3, 4, 2],
                "Ascan_id": ["a", "b", "a", "a", "a"],
                "aa2": ["C", "C", "C", "A", "C"],
                "sub_type": ["AC", "AC", "DC", "AC", "AC"],
                "score": [0.5, 0.5, 0.7, 0.9, 1.0],
            },
            index=[1, 7, 3, 4, 0],
        )
        test_data = pd.DataFrame(
            {"sub_type": ["AC", "DC", "AK", "AC"], "matrix": [1, 2, 3, 4]},
            index=[8, 2, 3, 1],
        )
        tr_result, te_result = train.refit_matrix_score(train_data, test_data)
        tr_expect = pd.DataFrame(
            {
                "dms_id": ["dms1"] * 4 + ["dms2"],
                "position": [2, 2, 3, 4, 2],
                "Ascan_id": ["a", "b", "a", "a", "a"],
                "aa2": ["C", "C", "C", "A", "C"],
                "sub_type": ["AC", "AC", "DC", "AC", "AC"],
                "score": [0.5, 0.5, 0.7, 0.9, 1.0],
                "matrix": [0.8, 0.8, 0.7, 0.8, 0.8],
            },
            index=[1, 7, 3, 4, 0],
        )
        assert_frame_equal(tr_result, tr_expect)
        te_expect = pd.DataFrame(
            {"sub_type": ["AC", "DC", "AK", "AC"], "matrix": [0.8, 0.7, np.nan, 0.8]},
            index=[8, 2, 3, 1],
        )
        assert_frame_equal(te_result, te_expect)


class TestAnalysis(unittest.TestCase):
    def test_01_subgroup_spearmanr(self):
        input_data = pd.DataFrame(
            {
                "model_group": ["group1"] * 5 + ["group2"] * 3,
                "score": [1, 2, 3, 4, 5, -5, 3, 5],
                "pred_score": [1, 2, 3, 4, -2, -6, 2, 6],
            }
        )
        result = ana.subgroup_spearmanr(
            input_data, "model_group", "score", "pred_score"
        )
        expect = pd.DataFrame(
            {
                "rho": [0.0, 1.0],
                "size": [5.0, 3.0],
            },
            index=["group1", "group2"],
        )
        assert_frame_equal(result, expect)


class TestTuning(unittest.TestCase):
    def setUp(self):
        self.estimator = DummyRegressor("constant")
        self.search_space = [
            {
                "name": "constant",
                "type": "discrete",
                "domain": np.arange(8),
                "dtype": int,
            }
        ]
        n_sample = 10
        n_feature = 5
        self.trainx = np.random.rand(n_sample, n_feature)
        self.trainy = [5] * n_sample

    def test_01__create_hp_dict_1(self):
        search_space_01 = [
            {
                "name": "constant",
                "type": "discrete",
                "domain": np.arange(0.1, 7.1),
                "dtype": int,
            }
        ]
        result = tune._create_hp_dict([1.1], search_space_01)
        expect = {"constant": 1}
        self.assertEqual(result, expect)

    def test_02__create_hp_dict_2(self):
        search_space_01 = [
            {
                "name": "constant",
                "type": "discrete",
                "domain": np.arange(0.1, 7.1),
                "dtype": float,
            }
        ]
        result = tune._create_hp_dict([3.1], search_space_01)
        expect = {"constant": 3.1}
        self.assertEqual(result, expect)

    def test_03__create_evaluation_function_1(self):
        func = tune._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error", "cv": 2},
        )
        result = func([[5]])
        self.assertEqual(result, 0.0)

    def test_04__create_evaluation_function_2(self):
        func = tune._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error", "cv": 2},
        )
        result = func([[3]])
        self.assertEqual(result, -4.0)

    def test_05__tune_by_gpyopt(self):
        eval_hp_func = tune._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error", "cv": 3},
        )
        result = tune._tune_by_gpopt(
            self.search_space, eval_hp_func, 6, 1, True, random_seed=0
        )
        expect = {"constant": 5}
        self.assertEqual(result, expect)

    def test_06_fit_best_estimator(self):
        result = tune.fit_best_estimator(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {
                "scoring": "neg_mean_squared_error",
                "cv": 3,
                "fit_params": {"sample_weight": None},
            },
            {
                "num_iterations": 6,
                "num_cores": 1,
                "if_maximize": True,
                "random_seed": 0,
            },
        )
        self.assertEqual(result.get_params()["constant"], 5.0)
