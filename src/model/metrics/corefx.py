import json
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_jsonl(filename, tag):
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
        data = [x for x in data if tag is None or tag in x['tags']]
        return {x['id']: x for x in data}


def clusters_to_mentions(cluster_spans):
    flatten_mentions = list()
    for curr_cluster, entity_type in cluster_spans:
        for curr_span in curr_cluster:
            flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


def cluster_to_mentions(cluster_spans, entity_type):
    flatten_mentions = list()
    for curr_span in cluster_spans:
        flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


def mention2cluster(clusters):
    clusters = [tuple(tuple(m) for m in gc) for gc in clusters]
    mention_to_cluster = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster[mention] = cluster
    return mention_to_cluster


class MetricCoref:

    def __init__(self, name, m, verbose=False):
        self.name = name
        self.m = m
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0

    def add(self, pred, gold):
        if self.m == self.ceafe or self.m == self.ceafe_singleton_entities or self.m == self.ceafe_singleton_mentions:
            p_num, p_den, r_num, r_den = self.m(pred, gold)
        else:
            p_num, p_den = self.m(pred, mention2cluster(gold))
            r_num, r_den = self.m(gold, mention2cluster(pred))

        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def get_f1(self):
        precision = self.get_pr()
        recall = self.get_re()
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_pr(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        return precision

    def get_re(self):
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return recall

    def print(self):
        f1 = self.get_f1()

        print('coref\t{}\t{}'.format(self.name, f1))

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def b_cubed_singleton_entities(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                correct += count * count

            numerator += correct / float(len(cluster)) / float(len(cluster))

            denominator += 1
        return numerator, denominator

    @staticmethod
    def b_cubed_singleton_mentions(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4_entity_centric(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        (kzaporoj) - Entity centric (normalizes by the len of the involved clusters)
        """
        return (
                2
                * len([mention for mention in gold_clustering if mention in predicted_clustering])
                / float(len(gold_clustering) + len(predicted_clustering))
        )

    @staticmethod
    def phi4_mention_centric(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        (kzaporoj) - Mention centric (sum of the number of mentions in intersected clusters)
        """
        return (
            len([mention for mention in gold_clustering if mention in predicted_clustering])
        )

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) != 1]  # is this really correct?
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_entity_centric(gold_cluster, cluster)
        # print('pred:', [len(x) for x in clusters])
        # print('gold:', [len(x) for x in gold_clusters])
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)

    @staticmethod
    def ceafe_singleton_entities(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        (kzaporoj) - this is entity-centric version where the cost is based on formula (9) of the paper
        """
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_entity_centric(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)

    @staticmethod
    def ceafe_singleton_mentions(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        (kzaporoj) - this is mention-centric version where the cost is based on formula (8) of the paper
        """
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_mention_centric(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        cluster_mentions = [item for sublist in clusters for item in sublist]
        gold_cluster_mentions = [item for sublist in gold_clusters for item in sublist]
        return similarity, len(cluster_mentions), similarity, len(gold_cluster_mentions)


# version modified by Klim to include singletons
class MetricCorefExternal:

    def __init__(self, task, verbose=False):
        self.task = task
        self.debug = False
        self.iter = 0
        self.clear()

    def clear(self):
        self.coref_muc = MetricCoref('muc', MetricCoref.muc)
        self.coref_bcubed_singleton_men = MetricCoref('bcubed-m', MetricCoref.b_cubed_singleton_mentions)
        self.coref_ceafe_singleton_ent = MetricCoref('ceaf-e', MetricCoref.ceafe_singleton_entities)

    def step(self):
        self.clear()
        self.iter += 1

    def update2(self, output_dict, metadata):
        for idx, (pred, gold) in enumerate(zip(output_dict['pred'], output_dict['gold'])):
            self.coref_muc.add(pred, gold)
            self.coref_bcubed_singleton_men.add(pred, gold)
            self.coref_ceafe_singleton_ent.add(pred, gold)

            if self.debug:
                print('ID', metadata['identifiers'][idx])
                print('pred:', pred)
                print('gold:', gold)
                tokens = metadata['tokens'][idx]
                print('pred:', [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in pred])
                print('gold:', [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in gold])
                print()

    def print(self, dataset_name, details=False):
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter,
                                                                   'muc-ext', self.coref_muc.get_f1()))
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter,
                                                                   'bcubed-m-ext',
                                                                   self.coref_bcubed_singleton_men.get_f1()))
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter,
                                                                   'ceaf-e-ext',
                                                                   self.coref_ceafe_singleton_ent.get_f1()))
        tmp = (self.coref_muc.get_f1() + self.coref_bcubed_singleton_men.get_f1() +
               self.coref_ceafe_singleton_ent.get_f1()) / 3
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter, 'avg-ext', tmp))

    def log(self, tb_logger, dataset_name):
        return
