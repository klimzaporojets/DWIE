import unittest

from numpy.testing import assert_approx_equal

from dwie_evaluation import load_data, EvaluatorDWIE


class DWIEEvalTest(unittest.TestCase):

    # @unittest.skip('')
    def test_cluster_scenario1(self):
        print('====TESTING CLUSTER SCENARIO 1====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario1.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario1.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 2) * (2 / 2 + 2 / 2)
        re_bcubed_singleton_men = (1 / 3) * (2 / 2 + 2 / 2 + 0 / 1)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = 1 / (1)
        re_ceafe_singleton_ent = 1 / (1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 1====')

    def test_cluster_scenario2(self):
        print('====TESTING CLUSTER SCENARIO 2====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario2.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario2.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 3) * (2 / 2 + 2 / 2 + 1 / 1)
        re_bcubed_singleton_men = (1 / 5) * (2 / 3 + 2 / 3 + 0 / 3 + 1 / 1 + 0 / 1)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 2) / (2 + 3) + (2 * 1) / (1 + 1)) / (1 + 1)
        re_ceafe_singleton_ent = ((2 * 2) / (2 + 3) + (2 * 1) / (1 + 1)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 2====')

    def test_cluster_scenario3_a(self):
        print('====TESTING CLUSTER SCENARIO 3 a====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario3.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario3_a.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 12) * (
                5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 2 / 7 + 2 / 7 + 5 / 7 + 5 / 7 + 5 / 7 + 5 / 7 + 5 / 7)
        re_bcubed_singleton_men = (1 / 12) * (
                5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 2 / 2 + 2 / 2 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 5) / (5 + 5) + (2 * 5) / (5 + 7)) / (1 + 1)
        re_ceafe_singleton_ent = ((2 * 5) / (5 + 5) + (2 * 5) / (5 + 7)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 3 a====')

    def test_cluster_scenario3_b(self):
        print('====TESTING CLUSTER SCENARIO 3 b====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario3.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario3_b.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 12) * (
                5 / 10 + 5 / 10 + 5 / 10 + 5 / 10 + 5 / 10 + 2 / 2 + 2 / 2 + 5 / 10 + 5 / 10 + 5 / 10 + 5 / 10 + 5 / 10)
        re_bcubed_singleton_men = (1 / 12) * (
                5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 2 / 2 + 2 / 2 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 5) / (5 + 10) + (2 * 2) / (2 + 2)) / (1 + 1)
        re_ceafe_singleton_ent = ((2 * 5) / (5 + 10) + (2 * 2) / (2 + 2)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 3 b====')

    def test_cluster_scenario3_c(self):
        print('====TESTING CLUSTER SCENARIO 3 c====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario3.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario3_c.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 12) * (
                5 / 12 + 5 / 12 + 5 / 12 + 5 / 12 + 5 / 12 + 2 / 12 + 2 / 12 + 5 / 12 + 5 / 12 + 5 / 12 + 5 / 12 + 5 / 12)
        re_bcubed_singleton_men = (1 / 12) * (
                5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 2 / 2 + 2 / 2 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5 + 5 / 5)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 5) / (5 + 12)) / (1)
        re_ceafe_singleton_ent = ((2 * 5) / (5 + 12)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 3 c====')

    def test_cluster_scenario3_d(self):
        print('====TESTING CLUSTER SCENARIO 3 d====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario3.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario3_d.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 12) * (
                1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1)
        re_bcubed_singleton_men = (1 / 12) * (
                1 / 5 + 1 / 5 + 1 / 5 + 1 / 5 + 1 / 5 + 1 / 2 + 1 / 2 + 1 / 5 + 1 / 5 + 1 / 5 + 1 / 5 + 1 / 5)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 1) / (1 + 5) + (2 * 1) / (1 + 2) + (2 * 1) / (1 + 5)) / \
                                 (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)
        re_ceafe_singleton_ent = ((2 * 1) / (1 + 5) + (2 * 1) / (1 + 2) + (2 * 1) / (1 + 5)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 3 d====')

    def test_cluster_scenario4(self):
        print('====TESTING CLUSTER SCENARIO 4====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario4.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario4.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        # testing mention-centric b-cubed for singletons
        pr_bcubed_singleton_men = (1 / 7) * (
                0 / 3 + 2 / 3 + 2 / 3 + 1 / 3 + 2 / 3 + 2 / 3 + 0 / 1)
        re_bcubed_singleton_men = (1 / 5) * (
                2 / 2 + 2 / 2 + 1 / 1 + 2 / 2 + 2 / 2)
        f1_bcubed_singleton_men = (2 * pr_bcubed_singleton_men * re_bcubed_singleton_men) / \
                                  (pr_bcubed_singleton_men + re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_pr(), pr_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_re(), re_bcubed_singleton_men)
        assert_approx_equal(dwie_eval.coref_bcubed.get_f1(), f1_bcubed_singleton_men)

        # testing entity-centric ceafe for singletons
        pr_ceafe_singleton_ent = ((2 * 2) / (2 + 3) + (2 * 2) / (2 + 3)) / (1 + 1 + 1)
        re_ceafe_singleton_ent = ((2 * 2) / (2 + 3) + (2 * 2) / (2 + 3)) / (1 + 1 + 1)
        f1_ceafe_singleton_ent = (2 * pr_ceafe_singleton_ent * re_ceafe_singleton_ent) / \
                                 (pr_ceafe_singleton_ent + re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_pr(), pr_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_re(), re_ceafe_singleton_ent)
        assert_approx_equal(dwie_eval.coref_ceafe.get_f1(), f1_ceafe_singleton_ent)

        print('====END TESTING CLUSTER SCENARIO 4====')

    def test_cluster_scenario5(self):
        """ This is for the following paper: https://www.aclweb.org/anthology/P14-2006.pdf """
        print('====TESTING CLUSTER SCENARIO 5====')
        ner_gold_path = 'tests/data/cluster/tests_cluster_gold_scenario5.json'
        ner_pred_scenario_path = 'tests/data/cluster/tests_cluster_pred_scenario5.json'
        gold = load_data(ner_gold_path)
        pred_scenario = load_data(ner_pred_scenario_path)
        dwie_eval = EvaluatorDWIE()
        for identifier in gold.keys():
            dwie_eval.add(pred_scenario[identifier], gold[identifier])
        dwie_eval.printInfo()

        print('====END TESTING CLUSTER SCENARIO 5====')
