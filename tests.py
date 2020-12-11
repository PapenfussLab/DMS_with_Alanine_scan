import models as models
import numpy as np
import tempfile
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
import unittest


class TestFitting(unittest.TestCase):
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

    def test_01__create_hp_dict(self):
        search_space_01 = [
            {
                "name": "constant",
                "type": "discrete",
                "domain": np.arange(0.1, 7.1),
                "dtype": int,
            }
        ]
        result = models._create_hp_dict([1.1], search_space_01)
        expect = {"constant": 1}
        self.assertEqual(result, expect)

        search_space_01 = [
            {
                "name": "constant",
                "type": "discrete",
                "domain": np.arange(0.1, 7.1),
                "dtype": float,
            }
        ]
        result = models._create_hp_dict([3.1], search_space_01)
        expect = {"constant": 3.1}
        self.assertEqual(result, expect)

    def test_02__create_evaluation_function(self):
        func = models._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error", "cv": 2},
        )
        result = func([[4]])
        self.assertEqual(result, -1.0)

    def test_03__tune_by_bayesian_optimization(self):
        eval_hp_func = models._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error"},
        )
        result = models._tune_by_bayesian_optimization(
            self.search_space, eval_hp_func, 6, 1, True
        )
        expect = {"constant": 5}
        self.assertEqual(result, expect)

    def test_04_fit_best_estimator(self):
        result = models.fit_best_estimator(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {
                "scoring": "neg_mean_squared_error",
                "fit_params": {"sample_weight": None},
            },
            {"num_iterations": 6, "num_cores": 1, "if_maximize": True},
        )
        self.assertEqual(result.get_params()["constant"], 5.0)


class TestProcessing(unittest.TestCase):
    def setUp(self):
        self.dms_data = pd.DataFrame(
            {
                "dms_id": ["dms"] * 8,
                "pos_id": [
                    "dms_1",
                    "dms_1",
                    "dms_2",
                    "dms_2",
                    "dms_3",
                    "dms_4",
                    "dms_4",
                    "dms_4",
                ],
                "position": [1, 1, 2, 2, 3, 4, 4, 4],
                "aa1": ["A", "A", "C", "C", "A", "E", "E", "E"],
                "aa2": ["C", "D", "A", "D", "D", "A", "C", "D"],
                "score": [0.9, 0.8, 0.85, 0.7, 0.6, 0.65, 0.75, 0.95],
            },
            index=[8, 1, 2, 3, 4, 5, 9, 7],
        )
        self.feature = ["position"]
        self.y_col_name = "score"
        self.tr_te_indices = [
            [8, 1, 7, 4, 5, 2],
            [9, 3],
        ]
        self.train_data = pd.DataFrame(
            {
                "dms_id": ["dms"] * 6,
                "pos_id": ["dms_1", "dms_1", "dms_4", "dms_3", "dms_4", "dms_2"],
                "position": [1, 1, 4, 3, 4, 2],
                "aa1": ["A", "A", "E", "A", "E", "C"],
                "aa2": ["C", "D", "D", "D", "A", "A"],
                "score": [0.9, 0.8, 0.95, 0.6, 0.65, 0.85],
            },
            index=[8, 1, 7, 4, 5, 2],
        )
        self.test_data = pd.DataFrame(
            {
                "dms_id": ["dms"] * 2,
                "pos_id": ["dms_4", "dms_2"],
                "position": [4, 2],
                "aa1": ["E", "C"],
                "aa2": ["C", "D"],
                "score": [0.75, 0.7],
            },
            index=[9, 3],
        )
        self.estimator = DummyRegressor("constant", constant=1).fit(
            self.train_data[self.feature], self.train_data[self.y_col_name]
        )

    def test_01__split_tr_te(self):
        train_data, test_data = models.split_tr_te(self.dms_data, self.tr_te_indices)
        assert_frame_equal(train_data, self.train_data)
        assert_frame_equal(test_data, self.test_data)

    def test_02_save_compared_prediction(self):
        tempdir = tempfile.TemporaryDirectory(dir="./")
        output_header = tempdir.name + "/"
        models.save_compared_prediction(
            self.estimator, self.test_data, self.feature, self.y_col_name, output_header
        )
        result = pd.read_csv(output_header + "prediction.csv", index_col=0)
        expect = pd.DataFrame(
            {
                "dms_id": ["dms"] * 2,
                "position": [4, 2],
                "aa2": ["C", "D"],
                "ob_score": [0.75, 0.7],
                "pred_score": [1, 1],
                "if_train": [False] * 2,
            },
            index=[9, 3],
        )
        assert_frame_equal(result, expect)
        tempdir.cleanup()

    def test_03_save_tuned_hyperparameters(self):
        tempdir = tempfile.TemporaryDirectory(dir="./")
        output_header = tempdir.name + "/"
        models.save_tuned_hyperparameters(
            self.estimator, [{"name": "constant"}, {"name": "strategy"}], output_header
        )
        result = pd.read_csv(output_header + "tuned_result.csv", index_col=0)
        expect = pd.DataFrame({"constant": 1, "strategy": "constant"}, index=[0])
        assert_frame_equal(result, expect)
        tempdir.cleanup()
