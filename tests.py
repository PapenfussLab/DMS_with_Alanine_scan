import preproc as preproc
import models as models
import matrix as mat
import analysis as ana
import visualization.parsers as vpar
import numpy as np
import tempfile
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
import unittest


class TestPreprocessing(unittest.TestCase):
    def test_01_impute_missing_value(self):
        categ_data = pd.DataFrame(
            {"1": ["a", "b", np.nan, "b", np.nan], "2": ["B", np.nan, "B", "A", "B"]}
        )
        result = preproc._impute_missing_value(categ_data, "most_frequent")
        expect = pd.DataFrame(
            {"1": ["a", "b", "b", "b", "b"], "2": ["B", "B", "B", "A", "B"]}
        )
        assert_frame_equal(result, expect)

        numer_data = pd.DataFrame(
            {"a": [1, 2, np.nan, 3, np.nan], "b": [2.3, np.nan, 0.4, 1.4, 2.3]}
        )
        result = preproc._impute_missing_value(numer_data, "mean")
        expect = pd.DataFrame(
            {"a": [1.0, 2.0, 2.0, 3.0, 2.0], "b": [2.3, 1.6, 0.4, 1.4, 2.3]}
        )
        assert_frame_equal(result, expect)

    def test_02_encode_categorical_feature(self):
        categ_data = pd.DataFrame(
            {"1": ["a", "b", "b", "b", "b"], "2": ["B", "B", "B", "A", "B"]},
            index=[1, 2, 3, 4, 0],
        )
        result = preproc._encode_categorical_feature(categ_data)
        expect = pd.DataFrame(
            {
                "1_a": [1.0, 0.0, 0.0, 0.0, 0.0],
                "1_b": [0.0, 1.0, 1.0, 1.0, 1.0],
                "2_A": [0.0, 0.0, 0.0, 1.0, 0.0],
                "2_B": [1.0, 1.0, 1.0, 0.0, 1.0],
            },
            index=[0, 1, 2, 3, 4],
        )
        assert_frame_equal(result, expect)

    def test_03_preproces_data_for_envision(self):
        dms_data = pd.DataFrame(
            {
                "1": ["a", "b", np.nan, "b", np.nan],
                "a": [1, 2, np.nan, 3, np.nan],
                "b": [2.3, np.nan, 0.4, 1.4, 2.3],
                "2": ["B", np.nan, "B", "A", "B"],
            },
            index=[1, 2, 3, 4, 0],
        )
        result = preproc.impute_encode_features(dms_data, ["2", "1"], ["b", "a"])
        expect = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, 3.0, 2.0],
                "b": [2.3, 1.6, 0.4, 1.4, 2.3],
                "2_A": [0.0, 0.0, 0.0, 1.0, 0.0],
                "2_B": [1.0, 1.0, 1.0, 0.0, 1.0],
                "1_a": [1.0, 0.0, 0.0, 0.0, 0.0],
                "1_b": [0.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=[1, 2, 3, 4, 0],
        )
        assert_frame_equal(result, expect)


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

    def test_03__tune_by_gpyopt(self):
        eval_hp_func = models._create_evaluation_function(
            self.search_space,
            self.estimator,
            self.trainx,
            self.trainy,
            {"scoring": "neg_mean_squared_error"},
        )
        result = models._tune_by_gpopt(self.search_space, eval_hp_func, 6, 1, True)
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


class TestPostProcessing(unittest.TestCase):
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

    def test_01_save_compared_prediction(self):
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

    def test_02_save_tuned_hyperparameters(self):
        tempdir = tempfile.TemporaryDirectory(dir="./")
        output_header = tempdir.name + "/"
        models.save_tuned_hyperparameters(
            self.estimator, [{"name": "constant"}, {"name": "strategy"}], output_header
        )
        result = pd.read_csv(output_header + "tuned_result.csv", index_col=0)
        expect = pd.DataFrame({"constant": 1, "strategy": "constant"}, index=[0])
        assert_frame_equal(result, expect)
        tempdir.cleanup()


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.test_dms_data = pd.DataFrame(
            {
                "dms_id": ["test"] * 8,
                "pos_id": [
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_3",
                    "test_4",
                    "test_4",
                    "test_4",
                ],
                "position": [1, 1, 2, 2, 3, 4, 4, 4],
                "aa1": ["A", "A", "C", "C", "A", "E", "E", "E"],
                "aa2": ["C", "D", "A", "D", "D", "A", "C", "D"],
                "score": [0.9, 0.8, 0.85, 0.7, 0.6, 0.65, 0.75, 0.95],
            },
            index=[
                "test_1_C",
                "test_1_D",
                "test_2_A",
                "test_2_D",
                "test_3_D",
                "test_4_A",
                "test_4_C",
                "test_4_D",
            ],
        )

    def test_01_create_wt_matrix(self):
        result = mat._create_wt_matrix(self.test_dms_data, 0.0)
        expect = pd.DataFrame(
            {
                "pos_id": ["test_1", "test_2", "test_3", "test_4"],
                "aa2": ["A", "C", "A", "E"],
                "score": [0.0] * 4,
            },
            index=["test_1_C", "test_2_A", "test_3_D", "test_4_A"],
        )
        assert_frame_equal(result, expect)

    def test_02_create_score_matrix(self):
        result = mat.create_score_matrix(self.test_dms_data, ["A", "C", "D"])
        expect = pd.DataFrame(
            {
                "dms_id": ["test"] * 3,
                "position": [1, 2, 4],
                "aa1": ["A", "C", "E"],
                "A_score": [1.0, 0.85, 0.65],
                "C_score": [0.9, 1.0, 0.75],
                "D_score": [0.8, 0.7, 0.95],
            },
            index=["test_1", "test_2", "test_4"],
        )
        expect.index.name = "pos_id"
        assert_frame_equal(result, expect)

    def test_03_create_score_matrix_check(self):
        wt_dms_data = pd.DataFrame(
            {
                "dms_id": ["test"] * 8,
                "pos_id": [
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_3",
                    "test_4",
                    "test_4",
                    "test_4",
                ],
                "position": [1, 1, 2, 2, 3, 4, 4, 4],
                "aa1": ["A", "A", "C", "C", "A", "E", "E", "E"],
                "aa2": ["C", "D", "C", "D", "D", "A", "C", "D"],
                "score": [0.9, 0.8, 0.85, 0.7, 0.6, 0.65, 0.75, 0.95],
            }
        )
        self.assertRaises(
            ValueError, mat.create_score_matrix, wt_dms_data, ["A", "C", "D"]
        )

    def test_04_create_score_matrix_keep_nan(self):
        result = mat.create_score_matrix(self.test_dms_data, ["A", "C", "E"], None)
        expect = pd.DataFrame(
            {
                "dms_id": ["test"] * 3,
                "position": [1, 2, 4],
                "aa1": ["A", "C", "E"],
                "A_score": [1.0, 0.85, 0.65],
                "C_score": [0.9, 1.0, 0.75],
                "E_score": [np.nan, np.nan, 1.0],
            },
            index=["test_1", "test_2", "test_4"],
        )
        expect.index.name = "pos_id"
        assert_frame_equal(result, expect)

    def test_05_create_score_matrix_row_mean(self):
        result = mat.create_score_matrix(
            self.test_dms_data, ["A", "C", "E"], "row_mean", None
        )
        expect = pd.DataFrame(
            {
                "dms_id": ["test"] * 3,
                "position": [1, 2, 4],
                "aa1": ["A", "C", "E"],
                "A_score": [0.9, 0.85, 0.65],
                "C_score": [0.9, 0.85, 0.75],
                "E_score": [0.9, 0.85, 0.7],
            },
            index=["test_1", "test_2", "test_4"],
        )
        expect.index.name = "pos_id"
        assert_frame_equal(result, expect)


class TestAnalasis(unittest.TestCase):
    def test_01_calculate_score_feature_correlation(self):
        score_feat = pd.DataFrame(
            {
                "score": [1, 2, 3, 4, 5, 6],
                "dms_id": ["dms1"] * 3 + ["dms2"] * 3,
                "feat1": [1, 2, 3, 4, 5, 6],
                "feat2": [6, 5, 4, 3, 2, 1],
                "feat3": [3, 2, 1, 4, 5, 6],
            }
        )
        result = ana.calculate_score_feature_correlation(
            score_feat, ["feat1", "feat2", "feat3"], "score", "dms_id"
        )
        expect = pd.DataFrame(
            data={"feat2": [-1.0, -1.0], "feat3": [-1.0, 1.0], "feat1": [1.0, 1.0]},
            index=["dms1", "dms2"],
        )
        assert_frame_equal(result, expect)

    def test_02_calculate_model_performance(self):
        pred_result = pd.DataFrame(
            {
                "dmsa_id": ["dmsa1"] * 5 + ["dmsa2"] * 2,
                "ob_score": [1, 2, 3, 4, -5, 3, 5],
                "pred_score_ala": [1, 2, 3, 4, -6, 2, 6],
                "pred_score_noala": [2, 1, 4, 0, 5, 3, 5],
                "uniprot_id": ["unip1"] * 5 + ["unip2"] * 2,
                "protein_name": ["prot1"] * 5 + ["prot2"] * 2,
                "dms_id": ["dms1"] * 5 + ["dms2"] * 2,
                "dms_name": ["DMS1"] * 5 + ["DMS2"] * 2,
                "Ascan_id": ["as1"] * 5 + ["as2"] * 2,
            }
        )
        result = ana.calculate_model_performance(pred_result)
        expect = pd.DataFrame(
            {
                "noala_rmse": [np.sqrt(23.8), 0.0],
                "ala_rmse": [np.sqrt(0.2), 1.0],
                "diff_rmse": [np.sqrt(23.8) - np.sqrt(0.2), -1.0],
                "size": [5.0, 2.0],
                "uniprot_id": ["unip1", "unip2"],
                "protein_name": ["prot1", "prot2"],
                "dms_id": ["dms1", "dms2"],
                "dms_name": ["DMS1", "DMS2"],
                "Ascan_id": ["as1", "as2"],
                "dmsa_id": ["dmsa1", "dmsa2"],
            },
            index=[0, 1],
        )
        expect = expect[expect.columns.sort_values()]
        assert_frame_equal(result, expect)


class TestFiguresParsers(unittest.TestCase):
    def test_01_pick_sort_med_rmse_diff(self):
        model_perform = pd.DataFrame(
            {
                "dms_id": ["dms_1"] * 3 + ["dms_2"] * 6 + ["dms_3"],
                "uniprot_id": ["unip_1"] * 3 + ["unip_2"] * 7,
                "diff_rmse": [0.5, 0.3, 0.2, 0.1, 0.1, 0.4, 0.5, 0.8, 0.6, 0.35],
            }
        )
        result = vpar.pick_sort_med_rmse_diff(model_perform)
        expect = pd.DataFrame(
            {
                "diff_rmse": [0.4, 0.35, 0.3],
                "dms_id": ["dms_2", "dms_3", "dms_1"],
                "med_dist": [0.05, 0, 0],
                "uniprot_id": ["unip_2", "unip_2", "unip_1"],
                "pro_med_diff": [0.375, 0.375, 0.3],
            },
            index=[5, 9, 1],
        )
        assert_frame_equal(result, expect)

    def test_02_create_heatmap_score_matrix(self):
        order = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
        score = pd.DataFrame(
            {"aa2": order * 2, "score": np.random.rand(40)}, index=np.arange(40)
        )
        score.loc[0, "score"] = 1
        score.loc[22, "score"] = 1
        score.loc[38, "score"] = np.nan
        score_part = score.loc[[i for i in range(40) if i not in [0, 22, 38]]]
        data = pd.DataFrame(
            {
                "pos_id": ["dms_2"] * 19 + ["dms_1"] * 18,
                "u_pos": [2] * 19 + [1] * 18,
                "dms_id": ["dms"] * 37,
                "aa1": ["A"] * 19 + ["D"] * 18,
                "aa2": score_part["aa2"],
                "score_": score_part["score"],
            }
        )

        result = vpar.create_heatmap_score_matrix(data, "score_", order)
        expect = pd.DataFrame(
            {"aa1": ["D", "A"], "position": [1, 2]},
            index=pd.Series(["dms_1", "dms_2"], name="pos_id"),
        )
        for i, aa in enumerate(order):
            expect[aa + "_score"] = [score.loc[20 + i, "score"], score.loc[i, "score"]]
        assert_frame_equal(result, expect)

    def test_03_calc_aa_type_rmse(self):
        data = pd.DataFrame(
            {
                "some_aa": ["H"] * 5 + ["W"] * 3,
                "ob_score": [0, 1, 2, 3, 2, 1, 4, 0],
                "pred_score_ala": [0, 1, 2, 2, 0, 1, 4, 0],
                "pred_score_noala": [-1, 3, 1, 1, 2, 1, 4, 0],
            }
        )
        result = vpar.calc_aa_type_rmse(data, "some_aa")
        expect = pd.DataFrame(
            {
                "aa": ["W", "H"],
                "ala_rmse": [0.0, 1.0],
                "noala_rmse": [0.0, np.sqrt(2)],
                "aa_property": ["Aromatic", "Pos. charged"],
            },
            index=[1, 0],
        )
        assert_frame_equal(result, expect)

    def test_04_get_dmsas_shared_mutants(self):
        pred_result = pd.DataFrame(
            {
                "dmsa_id": ["dmsa1"] * 5 + ["dmsa2"] * 2 + ["dmsa3"],
                "ob_score": [1, 2, 3, 4, -5, 2, -5, 2],
                "pred_score_ala": [1, 2, 3, 4, -6, 2, 6, 3],
                "u_pos": [1, 2, 3, 4, 4, 2, 4, 4],
                "aa2": ["A", "C", "C", "A", "D", "C", "D", "D"],
            }
        )
        result = vpar.get_dmsas_shared_mutants(pred_result, ["dmsa1", "dmsa2"])
        expect = pd.DataFrame(
            {
                "dmsa1": [2.0, -6.0],
                "dmsa2": [2.0, 6.0],
                "ob_score": [2, -5],
                "u_pos": [2, 4],
                "mutant": ["2_C", "4_D"],
            },
            index=[1, 4],
        )
        assert_frame_equal(result, expect)

    def test_05_get_dmsas_pair_position_rmse(self):
        shared_mut = pd.DataFrame(
            {
                "dmsa1": [2.0, 1, -6.0],
                "dmsa2": [2.0, 0, -4.0],
                "ob_score": [2, 1, -5],
                "u_pos": [2, 2, 4],
                "mutant": ["2_C", "2_A", "4_D"],
            },
            index=[1, 2, 4],
        )
        result = vpar.get_dmsas_pair_position_rmse(shared_mut, ["dmsa1", "dmsa2"])
        expect = pd.DataFrame(
            {
                "u_pos": [2.0, 2.0, 4.0, 4.0],
                "dmsa_id": ["dmsa1", "dmsa2", "dmsa1", "dmsa2"],
                "rmse": [0.0, np.sqrt(0.5), 1.0, 1.0],
            },
            index=[0, 2, 1, 3],
        )
        assert_frame_equal(result, expect)

    def test_06_calc_structure_type_rmse_change(self):
        data = pd.DataFrame(
            {
                "dms_name": ["DMS1"] * 8 + ["DMS2"] * 2,
                "some_str": ["str1"] * 5 + ["str2"] * 3 + ["str1"] * 2,
                "ob_score": [0, 1, 2, 3, 2, 1, 4, 0, 1, 1],
                "pred_score_noala": [0, 1, 2, 2, 0, 1, 4, 3, 0, 2],
                "pred_score_ala": [-1, 3, 1, 1, 2, 1, 4, 3, 0, 4],
            }
        )
        result = vpar.calc_structure_type_rmse_change(data, "some_str")
        expect = pd.DataFrame(
            {
                "str1": [(1.0 - np.sqrt(2)) * 100, (1.0 - np.sqrt(5)) * 100],
                "str2": [0, np.nan],
            },
            index=["DMS1", "DMS2"],
        )
        assert_frame_equal(result, expect)
