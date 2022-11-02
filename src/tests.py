"""Testing code."""

import mavedb_tool as mtool
import preproc as preproc
import train as train
import analysis as ana
import tune as tune
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
import unittest


class TestMaveDBTool(unittest.TestCase):
    def test_01__clean_mavedb_scores(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "p.Cys39Ala",
                    "p.Met1Cys",
                    "p.Cys2Ala",
                    "p.Met1Cys",
                    "p.Arg23His",
                    "p.Cys39Ala",
                ],
                "score": [3, 1, 19, 7, np.nan, np.nan, 1.2],
            },
            index=[7, 6, 5, 4, 3, 2, 1],
        )
        result = mtool.clean_mavedb_scores(ori_data)
        expect = pd.DataFrame(
            {
                "hgvs_pro": ["p.Cys2Ala", "p.Cys39Ala", "p.Met1Cys"],
                "score": [7, 1.1, 11],
            },
            index=[0, 1, 2],
        )
        assert_frame_equal(result, expect)

    def test_02__is_single_hgvs_pro_variant_1(self):
        result = mtool._is_single_hgvs_pro_variant("p.Cys28delinsTrpVal")
        expect = False
        self.assertEqual(result, expect)

    def test_03__is_single_hgvs_pro_variant_2(self):
        result = mtool._is_single_hgvs_pro_variant("p.[Cys2Ala;Asp5Glu]")
        expect = False
        self.assertEqual(result, expect)

    def test_04__is_single_hgvs_pro_variant_3(self):
        result = mtool._is_single_hgvs_pro_variant("p.Ter337Ala")
        expect = False
        self.assertEqual(result, expect)

    def test_05__is_single_hgvs_pro_variant_4(self):
        result = mtool._is_single_hgvs_pro_variant("p.Asn234Thrfs")
        expect = False
        self.assertEqual(result, expect)

    def test_06__is_single_hgvs_pro_variant_5(self):
        result = mtool._is_single_hgvs_pro_variant("p.Met1?")
        expect = False
        self.assertEqual(result, expect)

    def test_07__is_single_hgvs_pro_variant_6(self):
        result = mtool._is_single_hgvs_pro_variant("p.Arg37Met")
        expect = True
        self.assertEqual(result, expect)

    def test_08__annotate_singe_hgvs_pro_variant_1(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.Lys119*")
        expect = tuple(["np.nan", "np.nan", "np.nan", "nonsense"])
        self.assertEqual(results, expect)

    def test_09__annotate_singe_hgvs_pro_variant_2(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.His2537Ter")
        expect = tuple(["np.nan", "np.nan", "np.nan", "nonsense"])
        self.assertEqual(results, expect)

    def test_10__annotate_singe_hgvs_pro_variant_3(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.Asn2del")
        expect = tuple(["np.nan", "np.nan", "np.nan", "deletion"])
        self.assertEqual(results, expect)

    def test_11__annotate_singe_hgvs_pro_variant_4(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.Met1=")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_12__annotate_singe_hgvs_pro_variant_5(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.=")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_13__annotate_singe_hgvs_pro_variant_6(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.(=)")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_14__annotate_singe_hgvs_pro_variant_7(self):
        results = mtool._annotate_singe_hgvs_pro_variant("_wt")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_15__annotate_singe_hgvs_pro_variant_8(self):
        results = mtool._annotate_singe_hgvs_pro_variant("_sy")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_16__annotate_singe_hgvs_pro_variant_9(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.Val14Val")
        expect = tuple(["np.nan", "np.nan", "np.nan", "synonymous"])
        self.assertEqual(results, expect)

    def test_17__annotate_singe_hgvs_pro_variant_10(self):
        results = mtool._annotate_singe_hgvs_pro_variant("p.Arg38Pro")
        expect = tuple(["R", "P", 38, "missense"])
        self.assertEqual(results, expect)

    def test_18_annotate_mavedb_sav_data(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Tyr",
                    "p.Leu24*",
                    "p.Ter337Ala",
                    "p.Phe55Ser",
                    "p.[Cys3Try;Gln22Glu;Lys32Ile]",
                    "p.Asn234Thrfs",
                    "p.Thr25Thr",
                    "_wt",
                ]
            },
            index=[7, 4, 8, 29, 31, 0, 74, 55],
        )
        result = mtool.annotate_mavedb_sav_data(ori_data)
        expect = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Tyr",
                    "p.Leu24*",
                    "p.Ter337Ala",
                    "p.Phe55Ser",
                    "p.[Cys3Try;Gln22Glu;Lys32Ile]",
                    "p.Asn234Thrfs",
                    "p.Thr25Thr",
                    "_wt",
                ],
                "position": [
                    1,
                    "np.nan",
                    "np.nan",
                    55,
                    "np.nan",
                    "np.nan",
                    "np.nan",
                    "np.nan",
                ],
                "aa1": [
                    "M",
                    "np.nan",
                    "np.nan",
                    "F",
                    "np.nan",
                    "np.nan",
                    "np.nan",
                    "np.nan",
                ],
                "aa2": [
                    "Y",
                    "np.nan",
                    "np.nan",
                    "S",
                    "np.nan",
                    "np.nan",
                    "np.nan",
                    "np.nan",
                ],
                "mut_type": [
                    "missense",
                    "nonsense",
                    "other",
                    "missense",
                    "other",
                    "other",
                    "synonymous",
                    "synonymous",
                ],
            },
            index=[7, 4, 8, 29, 31, 0, 74, 55],
        )
        assert_frame_equal(result, expect)

    def test_19_get_dms_wiltype_like_score_1(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": ["p.Met1Cys", "p.Cys39Ala", "p.Cys2Ala", "p.Arg23His"],
                "mut_type": ["missense", "missense", "missense", "missense"],
                "score": [1, 2, 3, 4],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = np.nan
        np.testing.assert_equal(result, expect)

    def test_20_get_dms_wiltype_like_score_2(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "p.Cys39Ala",
                    "p.=",
                    "p.Cys2Ala",
                    "p.Arg23His",
                ],
                "mut_type": [
                    "missense",
                    "missense",
                    "synonymous",
                    "missense",
                    "missense",
                ],
                "score": [1, 2, 100, 3, 4],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = 100
        self.assertEqual(result, expect)

    def test_21_get_dms_wiltype_like_score_3(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "_sy",
                    "p.Cys39Ala",
                    "p.Cys2Ala",
                    "_wt",
                    "p.Arg23His",
                ],
                "mut_type": [
                    "missense",
                    "synonymous",
                    "missense",
                    "missense",
                    "synonymous",
                    "missense",
                ],
                "score": [1, 100, 2, 3, 101, 4],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = 101
        self.assertEqual(result, expect)

    def test_22_get_dms_wiltype_like_score_4(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "p.=",
                    "p.Cys39Ala",
                    "p.Cys2Ala",
                    "p.(=)",
                    "p.Arg23His",
                ],
                "mut_type": [
                    "missense",
                    "synonymous",
                    "missense",
                    "missense",
                    "synonymous",
                    "missense",
                ],
                "score": [1, 100, 2, 3, 101, 4],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = 100
        self.assertEqual(result, expect)

    def test_23_get_dms_wiltype_like_score_5(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "p.Cys39Ala",
                    "p.Cys39Cys",
                    "p.Cys2Ala",
                    "p.Arg23His",
                    "p.Met1=",
                ],
                "mut_type": [
                    "missense",
                    "missense",
                    "synonymous",
                    "missense",
                    "missense",
                    "synonymous",
                ],
                "score": [1, 2, 102, 3, 4, 100],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = 101
        self.assertEqual(result, expect)

    def test_24_get_dms_wiltype_like_score_6(self):
        ori_data = pd.DataFrame(
            {
                "hgvs_pro": [
                    "p.Met1Cys",
                    "p.Cys39Ala",
                    "_wt",
                    "p.Cys39Cys",
                    "p.Cys2Ala",
                    "p.Arg23His",
                    "p.Met1=",
                ],
                "mut_type": [
                    "missense",
                    "missense",
                    "synonymous",
                    "synonymous",
                    "missense",
                    "missense",
                    "synonymous",
                ],
                "score": [1, 2, 103, 102, 3, 4, 100],
            }
        )
        result = mtool.get_dms_wiltype_like_score(ori_data)
        expect = 103
        self.assertEqual(result, expect)


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

    def test_06_normalize_dms_score_1(self):
        data = pd.DataFrame(
            {"score": [7, -5, -1, 2], "id": ["v1", "v3", "v5", "v2"]},
            index=[1, 3, 5, 2],
        )
        result = preproc.normalize_dms_score(data, -1, "negative")
        expect = pd.DataFrame(
            {"score": [3, 0, 1, 1.75], "id": ["v1", "v3", "v5", "v2"]},
            index=[1, 3, 5, 2],
        )
        pd.testing.assert_frame_equal(expect, result)

    def test_07_normalize_dms_score_2(self):
        data = pd.DataFrame(
            {"score": [7, -5, -1, 2], "id": ["v1", "v3", "v5", "v2"]},
            index=[1, 3, 5, 2],
        )
        result = preproc.normalize_dms_score(data, 2, "positive")
        expect = pd.DataFrame(
            {"score": [0, 2.4, 1.6, 1], "id": ["v1", "v3", "v5", "v2"]},
            index=[1, 3, 5, 2],
        )
        pd.testing.assert_frame_equal(expect, result)

    def test_08_normalize_dms_score_3(self):
        data = pd.DataFrame({"score": np.arange(1000)})
        result = preproc.normalize_dms_score(data, 900, "negative")
        expect = pd.DataFrame({"score": (np.arange(1000) - 5) / 895})
        pd.testing.assert_frame_equal(expect, result)

    def test_09_normalize_dms_score_4(self):
        data = pd.DataFrame({"score": np.arange(2000)})
        result = preproc.normalize_dms_score(data, 300, "positive")
        expect = pd.DataFrame({"score": (1989 - np.arange(2000)) / 1689})
        pd.testing.assert_frame_equal(expect, result)


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
