import unittest

from numpy.testing import assert_approx_equal

from dwie_evaluation import load_data, EvaluatorDWIE


class DWIEEvalTest(unittest.TestCase):

    def test_ner_scenario1_b(self):
        print('====TESTING NER SCENARIO 1 b====')
        ner_gold_path = 'tests/data/tests_ner_gold_b.json'
        ner_pred_scenario1_path = 'tests/data/tests_ner_pred_scenario1_b.json'
        gold = load_data(ner_gold_path)
        pred_scenario1 = load_data(ner_pred_scenario1_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario1_tps_p = (1.0 + 1.0 + 1.0)
        ner_soft_scenario1_tps_g = (1.0 + 1.0 + 1.0)
        ner_soft_scenario1_fps = (1.0 + 1.0)
        ner_soft_scenario1_fns = (1.0 + 1.0 + 1.0)
        #
        ner_soft_scenario1_pr = ner_soft_scenario1_tps_p / (ner_soft_scenario1_tps_p + ner_soft_scenario1_fps)
        ner_soft_scenario1_re = ner_soft_scenario1_tps_g / (ner_soft_scenario1_tps_g + ner_soft_scenario1_fns)
        ner_soft_scenario1_f1 = (2 * ner_soft_scenario1_pr * ner_soft_scenario1_re) / \
                                (ner_soft_scenario1_pr + ner_soft_scenario1_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario1_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario1_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario1_f1)

        ner_hard_scenario_tps = 1 + 1 + 1
        ner_hard_scenario_fps = 1 + 1
        ner_hard_scenario_fns = 1 + 1 + 1

        ner_hard_scenario_pr = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fps)
        ner_hard_scenario_re = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fns)
        ner_hard_scenario_f1 = (2 * ner_hard_scenario_pr * ner_hard_scenario_re) / \
                               (ner_hard_scenario_pr + ner_hard_scenario_re)

        assert_approx_equal(dwie_eval.tags_hard.get_pr(), ner_hard_scenario_pr)
        assert_approx_equal(dwie_eval.tags_hard.get_re(), ner_hard_scenario_re)
        assert_approx_equal(dwie_eval.tags_hard.get_f1(), ner_hard_scenario_f1)

        print('====END TESTING NER SCENARIO 1 b====')

    def test_ner_scenario2_b(self):
        print('====TESTING NER SCENARIO 2 b====')
        ner_gold_path = 'tests/data/tests_ner_gold_b.json'
        ner_pred_scenario1_path = 'tests/data/tests_ner_pred_scenario2_b.json'
        gold = load_data(ner_gold_path)
        pred_scenario1 = load_data(ner_pred_scenario1_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario1_tps_p = (1.0 + 1.0 + 1.0)
        ner_soft_scenario1_tps_g = (1.0 + 1.0 + 1.0)
        ner_soft_scenario1_fps = (1.0 + 1.0)
        ner_soft_scenario1_fns = (1.0 + 1.0 + 1.0)
        #
        ner_soft_scenario1_pr = ner_soft_scenario1_tps_p / (ner_soft_scenario1_tps_p + ner_soft_scenario1_fps)
        ner_soft_scenario1_re = ner_soft_scenario1_tps_g / (ner_soft_scenario1_tps_g + ner_soft_scenario1_fns)
        ner_soft_scenario1_f1 = (2 * ner_soft_scenario1_pr * ner_soft_scenario1_re) / \
                                (ner_soft_scenario1_pr + ner_soft_scenario1_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario1_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario1_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario1_f1)

        ner_hard_scenario_tps = 1 + 1 + 1
        ner_hard_scenario_fps = 1 + 1
        ner_hard_scenario_fns = 1 + 1 + 1

        ner_hard_scenario_pr = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fps)
        ner_hard_scenario_re = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fns)
        ner_hard_scenario_f1 = (2 * ner_hard_scenario_pr * ner_hard_scenario_re) / \
                               (ner_hard_scenario_pr + ner_hard_scenario_re)

        assert_approx_equal(dwie_eval.tags_hard.get_pr(), ner_hard_scenario_pr)
        assert_approx_equal(dwie_eval.tags_hard.get_re(), ner_hard_scenario_re)
        assert_approx_equal(dwie_eval.tags_hard.get_f1(), ner_hard_scenario_f1)

        print('====END TESTING NER SCENARIO 2 b====')

    def test_ner_scenario3_b(self):
        print('====TESTING NER SCENARIO 3 b====')
        ner_gold_path = 'tests/data/tests_ner_gold_b.json'
        ner_pred_scenario1_path = 'tests/data/tests_ner_pred_scenario3_b.json'
        gold = load_data(ner_gold_path)
        pred_scenario1 = load_data(ner_pred_scenario1_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario1_tps_p = (1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0)
        ner_soft_scenario1_tps_g = (8 / 9 + 8 / 9 + 8 / 9 + 1.0 + 1.0 + 1.0)
        ner_soft_scenario1_fps = 0.0
        ner_soft_scenario1_fns = (1 / 9 + 1 / 9 + 1 / 9)
        #
        ner_soft_scenario1_pr = ner_soft_scenario1_tps_p / (ner_soft_scenario1_tps_p + ner_soft_scenario1_fps)
        ner_soft_scenario1_re = ner_soft_scenario1_tps_g / (ner_soft_scenario1_tps_g + ner_soft_scenario1_fns)
        ner_soft_scenario1_f1 = (2 * ner_soft_scenario1_pr * ner_soft_scenario1_re) / \
                                (ner_soft_scenario1_pr + ner_soft_scenario1_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario1_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario1_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario1_f1)

        ner_hard_scenario_tps = 1 + 1 + 1
        ner_hard_scenario_fps = 1 + 1 + 1 + 1 + 1 + 1
        ner_hard_scenario_fns = 1 + 1 + 1

        ner_hard_scenario_pr = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fps)
        ner_hard_scenario_re = ner_hard_scenario_tps / (ner_hard_scenario_tps + ner_hard_scenario_fns)
        ner_hard_scenario_f1 = (2 * ner_hard_scenario_pr * ner_hard_scenario_re) / \
                               (ner_hard_scenario_pr + ner_hard_scenario_re)

        assert_approx_equal(dwie_eval.tags_hard.get_pr(), ner_hard_scenario_pr)
        assert_approx_equal(dwie_eval.tags_hard.get_re(), ner_hard_scenario_re)
        assert_approx_equal(dwie_eval.tags_hard.get_f1(), ner_hard_scenario_f1)

        print('====END TESTING NER SCENARIO 3 b====')

    def test_ner_scenario1(self):
        print('====TESTING NER SCENARIO 1====')
        ner_gold_path = 'tests/data/tests_ner_gold.json'
        ner_pred_scenario1_path = 'tests/data/tests_ner_pred_scenario1.json'
        gold = load_data(ner_gold_path)
        pred_scenario1 = load_data(ner_pred_scenario1_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario1_tps_p = (1.0 + 1.0 + 1.0) + (1.0)
        ner_soft_scenario1_tps_g = (1.0 + 1.0 + 1.0) + (1.0)
        ner_soft_scenario1_fps = 0.0
        ner_soft_scenario1_fns = (1.0 + 1.0)
        #
        ner_soft_scenario1_pr = ner_soft_scenario1_tps_p / (ner_soft_scenario1_tps_p + ner_soft_scenario1_fps)
        ner_soft_scenario1_re = ner_soft_scenario1_tps_g / (ner_soft_scenario1_tps_g + ner_soft_scenario1_fns)
        ner_soft_scenario1_f1 = (2 * ner_soft_scenario1_pr * ner_soft_scenario1_re) / \
                                (ner_soft_scenario1_pr + ner_soft_scenario1_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario1_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario1_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario1_f1)

        print('====END TESTING NER SCENARIO 1====')

    def test_ner_scenario2(self):
        print('====TESTING NER SCENARIO 2====')
        ner_gold_path = 'tests/data/tests_ner_gold.json'
        ner_pred_scenario2_path = 'tests/data/tests_ner_pred_scenario2.json'
        gold = load_data(ner_gold_path)
        pred_scenario2 = load_data(ner_pred_scenario2_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario2[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario2_tps_p = (1.0 + 1.0 + 1.0) + (1.0)
        ner_soft_scenario2_tps_g = (1.0 + 1.0 + 1.0) + (1.0)
        ner_soft_scenario2_fps = 0.0
        ner_soft_scenario2_fns = (1.0 + 1.0)
        #
        ner_soft_scenario2_pr = ner_soft_scenario2_tps_p / (ner_soft_scenario2_tps_p + ner_soft_scenario2_fps)
        ner_soft_scenario2_re = ner_soft_scenario2_tps_g / (ner_soft_scenario2_tps_g + ner_soft_scenario2_fns)
        ner_soft_scenario2_f1 = (2 * ner_soft_scenario2_pr * ner_soft_scenario2_re) / \
                                (ner_soft_scenario2_pr + ner_soft_scenario2_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario2_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario2_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario2_f1)

        print('====END TESTING NER SCENARIO 2====')

    def test_ner_scenario3(self):
        print('====TESTING NER SCENARIO 3====')
        ner_gold_path = 'tests/data/tests_ner_gold.json'
        ner_pred_scenario3_path = 'tests/data/tests_ner_pred_scenario3.json'
        gold = load_data(ner_gold_path)
        pred_scenario3 = load_data(ner_pred_scenario3_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario3[identifier], gold[identifier])
        dwie_eval.printInfo()

        ner_soft_scenario3_tps_p = (1.0 + 1.0 + 1.0) + (1.0 + 0.5 + 0.5)
        ner_soft_scenario3_tps_g = (9 / 9 + 8 / 9 + 8 / 9) + (1.0 + 1.0 + 1.0)
        ner_soft_scenario3_fps = (0.0 + 0.5 + 0.5)
        ner_soft_scenario3_fns = (0.0 + 1 / 9 + 1 / 9)
        #
        ner_soft_scenario3_pr = ner_soft_scenario3_tps_p / (ner_soft_scenario3_tps_p + ner_soft_scenario3_fps)
        ner_soft_scenario3_re = ner_soft_scenario3_tps_g / (ner_soft_scenario3_tps_g + ner_soft_scenario3_fns)
        ner_soft_scenario3_f1 = (2 * ner_soft_scenario3_pr * ner_soft_scenario3_re) / \
                                (ner_soft_scenario3_pr + ner_soft_scenario3_re)

        assert_approx_equal(dwie_eval.tags_soft.get_pr(), ner_soft_scenario3_pr)
        assert_approx_equal(dwie_eval.tags_soft.get_re(), ner_soft_scenario3_re)
        assert_approx_equal(dwie_eval.tags_soft.get_f1(), ner_soft_scenario3_f1)

        print('====END TESTING NER SCENARIO 3====')

    def test_relations_scenario1(self):
        print('====TESTING RELATIONS SCENARIO 1====')
        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario1 = 'tests/data/tests_rel_pred_scenario1.json'

        gold = load_data(rel_gold_path)

        pred_scenario1 = load_data(rel_pred_scenario1)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        rel_soft_scenario1_tps_p = 1.0
        rel_soft_scenario1_tps_g = 1.0
        rel_soft_scenario1_fps = 0.0
        rel_soft_scenario1_fns = 1.0
        #
        rel_soft_scenario1_pr = rel_soft_scenario1_tps_p / (rel_soft_scenario1_tps_p + rel_soft_scenario1_fps)
        rel_soft_scenario1_re = rel_soft_scenario1_tps_g / (rel_soft_scenario1_tps_g + rel_soft_scenario1_fns)
        rel_soft_scenario1_f1 = (2 * rel_soft_scenario1_pr * rel_soft_scenario1_re) / \
                                (rel_soft_scenario1_pr + rel_soft_scenario1_re)

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), rel_soft_scenario1_pr)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), rel_soft_scenario1_re)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), rel_soft_scenario1_f1)

        rel_hard_scenario1_tps = 1
        rel_hard_scenario1_fps = 0
        rel_hard_scenario1_fns = 1

        rel_hard_scenario1_pr = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fps)
        rel_hard_scenario1_re = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fns)
        rel_hard_scenario1_f1 = (2 * rel_hard_scenario1_pr * rel_hard_scenario1_re) / \
                                (rel_hard_scenario1_pr + rel_hard_scenario1_re)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), rel_hard_scenario1_pr)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), rel_hard_scenario1_re)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), rel_hard_scenario1_f1)

    def test_relations_scenario2(self):
        print('====TESTING RELATIONS SCENARIO 2====')

        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario2 = 'tests/data/tests_rel_pred_scenario2.json'

        gold = load_data(rel_gold_path)

        pred_scenario2 = load_data(rel_pred_scenario2)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario2[identifier], gold[identifier])
        dwie_eval.printInfo()

        #
        rel_soft_scenario2_tps_p = 1.0
        rel_soft_scenario2_tps_g = 1.0
        rel_soft_scenario2_fps = 0.0
        rel_soft_scenario2_fns = 1.0

        rel_soft_scenario2_pr = rel_soft_scenario2_tps_p / (rel_soft_scenario2_tps_p + rel_soft_scenario2_fps)
        rel_soft_scenario2_re = rel_soft_scenario2_tps_g / (rel_soft_scenario2_tps_g + rel_soft_scenario2_fns)
        rel_soft_scenario2_f1 = (2 * rel_soft_scenario2_pr * rel_soft_scenario2_re) / \
                                (rel_soft_scenario2_pr + rel_soft_scenario2_re)

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), rel_soft_scenario2_pr)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), rel_soft_scenario2_re)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), rel_soft_scenario2_f1)
        #

        rel_hard_scenario1_tps = 1
        rel_hard_scenario1_fps = 0
        rel_hard_scenario1_fns = 1

        rel_hard_scenario1_pr = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fps)
        rel_hard_scenario1_re = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fns)
        rel_hard_scenario1_f1 = (2 * rel_hard_scenario1_pr * rel_hard_scenario1_re) / \
                                (rel_hard_scenario1_pr + rel_hard_scenario1_re)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), rel_hard_scenario1_pr)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), rel_hard_scenario1_re)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), rel_hard_scenario1_f1)

    def test_relations_scenario3(self):
        print('====TESTING RELATIONS SCENARIO 3====')
        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario3 = 'tests/data/tests_rel_pred_scenario3.json'

        gold = load_data(rel_gold_path)

        pred_scenario1 = load_data(rel_pred_scenario3)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        #
        rel_soft_scenario3_tps_p = (1.0 + 2 / 3)
        rel_soft_scenario3_tps_g = ((8 * 8) / (9 * 8) + 1.0)
        rel_soft_scenario3_fps = 2.0 - (1.0 + 2 / 3)
        rel_soft_scenario3_fns = 2.0 - ((8 * 8) / (9 * 8) + 1.0)

        rel_soft_scenario3_pr = rel_soft_scenario3_tps_p / (rel_soft_scenario3_tps_p + rel_soft_scenario3_fps)
        rel_soft_scenario3_re = rel_soft_scenario3_tps_g / (rel_soft_scenario3_tps_g + rel_soft_scenario3_fns)
        rel_soft_scenario3_f1 = (2 * rel_soft_scenario3_pr * rel_soft_scenario3_re) / \
                                (rel_soft_scenario3_pr + rel_soft_scenario3_re)

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), rel_soft_scenario3_pr)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), rel_soft_scenario3_re)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), rel_soft_scenario3_f1)
        #

        rel_hard_scenario1_tps = 0
        rel_hard_scenario1_fps = 2
        rel_hard_scenario1_fns = 2

        rel_hard_scenario1_pr = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fps)
        rel_hard_scenario1_re = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fns)
        if rel_hard_scenario1_pr == 0 and rel_hard_scenario1_re == 0:
            rel_hard_scenario1_f1 = 0
        else:
            rel_hard_scenario1_f1 = (2 * rel_hard_scenario1_pr * rel_hard_scenario1_re) / \
                                    (rel_hard_scenario1_pr + rel_hard_scenario1_re)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), rel_hard_scenario1_pr)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), rel_hard_scenario1_re)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), rel_hard_scenario1_f1)

    def test_relations_scenario3b(self):
        print('====TESTING RELATIONS SCENARIO 3b====')
        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario3b = 'tests/data/tests_rel_pred_scenario3b.json'

        gold = load_data(rel_gold_path)

        pred_scenario1 = load_data(rel_pred_scenario3b)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        #
        rel_soft_scenario3b_tps_p = (1.0 + 1.0 + 2 / 3)
        rel_soft_scenario3b_tps_g = ((7 * 8) / (9 * 8) + 1.0)
        rel_soft_scenario3b_fps = 3.0 - (1.0 + 1.0 + 2 / 3)
        rel_soft_scenario3b_fns = 2.0 - ((7 * 8) / (9 * 8) + 1.0)

        rel_soft_scenario3b_pr = rel_soft_scenario3b_tps_p / (rel_soft_scenario3b_tps_p + rel_soft_scenario3b_fps)
        rel_soft_scenario3b_re = rel_soft_scenario3b_tps_g / (rel_soft_scenario3b_tps_g + rel_soft_scenario3b_fns)
        rel_soft_scenario3b_f1 = (2 * rel_soft_scenario3b_pr * rel_soft_scenario3b_re) / \
                                 (rel_soft_scenario3b_pr + rel_soft_scenario3b_re)

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), rel_soft_scenario3b_pr)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), rel_soft_scenario3b_re)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), rel_soft_scenario3b_f1)
        #

        rel_hard_scenario1_tps = 0
        rel_hard_scenario1_fps = 3
        rel_hard_scenario1_fns = 2

        rel_hard_scenario1_pr = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fps)
        rel_hard_scenario1_re = rel_hard_scenario1_tps / (rel_hard_scenario1_tps + rel_hard_scenario1_fns)
        if rel_hard_scenario1_pr == 0 and rel_hard_scenario1_re == 0:
            rel_hard_scenario1_f1 = 0
        else:
            rel_hard_scenario1_f1 = (2 * rel_hard_scenario1_pr * rel_hard_scenario1_re) / \
                                    (rel_hard_scenario1_pr + rel_hard_scenario1_re)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), rel_hard_scenario1_pr)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), rel_hard_scenario1_re)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), rel_hard_scenario1_f1)

    def test_relations_mention_based_01(self):
        print('====TESTING RELATIONS MENTION BASED SCENARIO 01====')
        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario = 'tests/data/tests_rel_pred_mention_based_01.json'

        gold = load_data(rel_gold_path)

        pred_scenario1 = load_data(rel_pred_scenario)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        #
        rel_soft_scenario_tps_p = (1.0 + 1.0 + 2 / 3)
        rel_soft_scenario_tps_g = ((7 * 8) / (9 * 8) + 1.0)
        rel_soft_scenario_fps = 3.0 - (1.0 + 1.0 + 2 / 3)
        rel_soft_scenario_fns = 2.0 - ((7 * 8) / (9 * 8) + 1.0)

        rel_soft_scenario_pr = rel_soft_scenario_tps_p / (rel_soft_scenario_tps_p + rel_soft_scenario_fps)
        rel_soft_scenario_re = rel_soft_scenario_tps_g / (rel_soft_scenario_tps_g + rel_soft_scenario_fns)
        rel_soft_scenario_f1 = (2 * rel_soft_scenario_pr * rel_soft_scenario_re) / \
                               (rel_soft_scenario_pr + rel_soft_scenario_re)

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), rel_soft_scenario_pr)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), rel_soft_scenario_re)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), rel_soft_scenario_f1)
        #

        rel_mention_scenario_fns = (9 * 8) + (2 * 1) - 3
        rel_mention_scenario_fps = 2
        rel_mention_scenario_tps = 3

        rel_mention_scenario_pr = rel_mention_scenario_tps / (rel_mention_scenario_tps + rel_mention_scenario_fps)
        rel_mention_scenario_re = rel_mention_scenario_tps / (rel_mention_scenario_tps + rel_mention_scenario_fns)
        rel_mention_scenario_f1 = (2 * rel_mention_scenario_pr * rel_mention_scenario_re) / \
                                  (rel_mention_scenario_pr + rel_mention_scenario_re)

        assert_approx_equal(dwie_eval.rels_mention.get_pr(), rel_mention_scenario_pr)
        assert_approx_equal(dwie_eval.rels_mention.get_re(), rel_mention_scenario_re)
        assert_approx_equal(dwie_eval.rels_mention.get_f1(), rel_mention_scenario_f1)

        rel_hard_scenario_tps = 0
        rel_hard_scenario_fps = 3
        rel_hard_scenario_fns = 2

        rel_hard_scenario_pr = rel_hard_scenario_tps / (rel_hard_scenario_tps + rel_hard_scenario_fps)
        rel_hard_scenario_re = rel_hard_scenario_tps / (rel_hard_scenario_tps + rel_hard_scenario_fns)
        if rel_hard_scenario_pr == 0 and rel_hard_scenario_re == 0:
            rel_hard_scenario_f1 = 0
        else:
            rel_hard_scenario_f1 = (2 * rel_hard_scenario_pr * rel_hard_scenario_re) / \
                                   (rel_hard_scenario_pr + rel_hard_scenario_re)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), rel_hard_scenario_pr)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), rel_hard_scenario_re)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), rel_hard_scenario_f1)

    def test_relations_mention_based_02(self):
        print('====TESTING RELATIONS MENTION BASED SCENARIO 02====')
        rel_gold_path = 'tests/data/tests_rel_gold.json'
        rel_pred_scenario = 'tests/data/tests_rel_pred_mention_based_02.json'

        gold = load_data(rel_gold_path)

        pred_scenario1 = load_data(rel_pred_scenario)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario1[identifier], gold[identifier])
        dwie_eval.printInfo()

        #

        assert_approx_equal(dwie_eval.rels_soft.get_pr(), 0.0)
        assert_approx_equal(dwie_eval.rels_soft.get_re(), 0.0)
        assert_approx_equal(dwie_eval.rels_soft.get_f1(), 0.0)
        #

        rel_mention_scenario_fns = (9 * 8) + (2 * 1) - 3
        rel_mention_scenario_fps = 2
        rel_mention_scenario_tps = 3

        rel_mention_scenario_pr = rel_mention_scenario_tps / (rel_mention_scenario_tps + rel_mention_scenario_fps)
        rel_mention_scenario_re = rel_mention_scenario_tps / (rel_mention_scenario_tps + rel_mention_scenario_fns)
        rel_mention_scenario_f1 = (2 * rel_mention_scenario_pr * rel_mention_scenario_re) / \
                                  (rel_mention_scenario_pr + rel_mention_scenario_re)

        assert_approx_equal(dwie_eval.rels_mention.get_pr(), rel_mention_scenario_pr)
        assert_approx_equal(dwie_eval.rels_mention.get_re(), rel_mention_scenario_re)
        assert_approx_equal(dwie_eval.rels_mention.get_f1(), rel_mention_scenario_f1)

        assert_approx_equal(dwie_eval.rels_hard.get_pr(), 0.0)
        assert_approx_equal(dwie_eval.rels_hard.get_re(), 0.0)
        assert_approx_equal(dwie_eval.rels_hard.get_f1(), 0.0)

    def test_readme_example(self):
        from dwie_evaluation import load_json, EvaluatorDWIE

        print('====TEST README EXAMPLE====')
        dwie_eval = EvaluatorDWIE()

        loaded_ground_truth = load_json('tests/data/tests_rel_gold.json', None)
        loaded_predicted = load_json('tests/data/tests_rel_gold.json', None)

        for article_id in loaded_ground_truth.keys():
            dwie_eval.add(loaded_predicted[article_id], loaded_ground_truth[article_id])

        # Coreference Metrics
        print('Coref MUC F1:', dwie_eval.coref_muc.get_f1())
        print('Coref B-Cubed F1:', dwie_eval.coref_bcubed.get_f1())
        print('Coref CEAFe F1:', dwie_eval.coref_ceafe.get_f1())
        print('Coref Avg.:', sum([dwie_eval.coref_muc.get_f1(), dwie_eval.coref_bcubed.get_f1(), \
                                  dwie_eval.coref_ceafe.get_f1()]) / 3)

        # NER Metrics
        print('NER Mention-Level F1:', dwie_eval.tags_mention.get_f1())
        print('NER Hard Entity-Level F1:', dwie_eval.tags_hard.get_f1())
        print('NER Soft Entity-Level F1:', dwie_eval.tags_soft.get_f1())

        # Relation Extraction (RE) Metrics
        print('RE Mention-Level F1:', dwie_eval.rels_mention.get_f1())
        print('RE Hard Entity-Level F1:', dwie_eval.rels_hard.get_f1())
        print('RE Soft Entity-Level F1:', dwie_eval.rels_soft.get_f1())
