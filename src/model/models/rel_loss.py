import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics.objective import MetricObjective
from model.metrics.relations import MetricConceptRelationSoftF1, MetricConceptRelationToMentionsF1, MetricSpanRelationF1x
from model.models.misc.misc import get_mask_from_sequence_lengths


def create_mapping(spans, clusters):
    num_batch = len(spans)
    max_spans = max([len(x) for x in spans])
    max_concepts = max([len(x) for x in clusters])

    mapping = torch.zeros(num_batch, max_concepts, max_spans)

    for batch, (myspans, myclusters) in enumerate(zip(spans, clusters)):
        span2index = {}
        for idx, span in enumerate(myspans):
            span2index[span] = idx

        for idx, cluster in enumerate(myclusters):
            for span in cluster:
                if span in span2index:  # in case relation pruner != coref pruner
                    mapping[batch, idx, span2index[span]] = 1.0

    return mapping


def sum_scores(scores, u):
    if scores.dim() != 4:
        raise BaseException('scores is not a 4-dimensional tensor')
    if u.dim() != 3:
        raise BaseException('mapping is not a 3-dimensional tensor')
    if scores.size(0) != u.size(0):
        raise BaseException('batch size doesn\'t match')
    num_batch, num_mentions, num_concepts = u.size()
    v = u.unsqueeze(1).expand(num_batch, num_concepts, num_mentions, num_concepts)
    o = torch.matmul(v, scores)
    p = o.view(o.size()[0:2] + (-1,))
    q = torch.matmul(u, p)
    q = q.view(q.size()[0:2] + o.size()[2:])
    return q


def log1mex(x):
    v1 = torch.log(-torch.expm1(x))
    return v1


def create_square_mask(lengths):
    mask = get_mask_from_sequence_lengths(lengths, lengths.max().item())
    mask = mask.float()
    square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))
    return square_mask


def decode_span_relations(scores, spanss, labels):
    relations = []
    for b, spans in enumerate(spanss):
        length = len(spans)
        rels = []
        for src, dst, rel in torch.nonzero(scores[b, 0:length, 0:length, :] > 0).tolist():
            rels.append((spans[src], labels[rel], spans[dst]))
        relations.append(rels)
    return relations


def gold_cluster_to_span_relations(clusters, relations, labels):
    output = []
    for cs, rels in zip(clusters, relations):
        tmp = []
        for src_cluster_idx, dst_cluster_idx, rel_idx in rels:
            for src in cs[src_cluster_idx]:
                for dst in cs[dst_cluster_idx]:
                    tmp.append((src, labels[rel_idx], dst))
        output.append(tmp)
    return output


def masked_sum(scores, mask):
    mask = mask.float()
    x = scores * mask.unsqueeze(1).unsqueeze(2)
    x = x.sum(dim=-1) * mask.unsqueeze(1)
    return x.sum()


def create_task_relations(name, config, labels):
    if config['type'] == 'binary':
        return LossRelations(name, config, labels)
    elif config['type'] == 'latent-binary':
        return LossRelationsLatent(name, config, labels)
    elif config['type'] == 'binary-x':
        return LossRelationsX(name, config, labels)
    elif config['type'] == 'span-binary':
        return TaskSpanRelations(name, config, labels)
    else:
        raise BaseException('no such relation task:', config['type'])


def create_relation_targets_2(pred_spans, relations, num_relations, span_lengths):
    gold_spans = relations['gold_spans']
    gold_m2i = relations['gold_m2i']
    gold_relations = relations['gold_relations']
    num_concepts = relations['num_concepts']

    num_batch = span_lengths.size(0)
    max_spans = span_lengths.max().item()

    targets = torch.zeros(num_batch, max_spans, max_spans, num_relations)

    for batch, (p_spans, g_spans, m2i, rels, max_clusters) in enumerate(
            zip(pred_spans, gold_spans, gold_m2i, gold_relations, num_concepts)):
        if len(rels) > 0:
            gold2index = {span: idx for span, idx in zip(g_spans, m2i)}
            pred2cluster = torch.LongTensor([gold2index.get(span, max_clusters) for span in p_spans])

            rels = torch.LongTensor(rels)
            cluster_targets = torch.zeros(max_clusters + 1, max_clusters + 1, num_relations)
            cluster_targets[rels[:, 0], rels[:, 1], rels[:, 2]] = torch.ones(rels.size(0))

            dim = (pred2cluster.size(0), pred2cluster.size(0))
            r = pred2cluster.unsqueeze(-1).expand(dim).reshape(-1)
            c = pred2cluster.unsqueeze(-2).expand(dim).reshape(-1)

            indices = torch.arange(pred2cluster.size(0))
            rr = indices.unsqueeze(-1).expand(dim).reshape(-1)
            cc = indices.unsqueeze(-2).expand(dim).reshape(-1)
            targets[batch, rr, cc, :] = cluster_targets[r, c, :]

    return targets.to(span_lengths.device)


def decode_relations_new(targets, lengths, labels):
    relations = []
    for b, length in enumerate(lengths):
        rels = []
        for src, dst, rel in torch.nonzero(targets[b, 0:length, 0:length, :] > 0).tolist():
            rels.append((src, dst, labels[rel]))
        relations.append(rels)
    return relations


class TaskSpanRelations(nn.Module):

    def __init__(self, name, config, labels):
        super(TaskSpanRelations, self).__init__()
        self.name = name
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config.get('normalize', True) else config['weight']
        self.debug = config['debug']
        print('TaskSpanRelations: weight={}'.format(self.weight))

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']

        if self.enabled:
            mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(-1)).sum()

            output['loss'] = obj

            output['span-rel-pred'] = decode_span_relations(mention_scores, pred_spans, self.labels)
            output['span-rel-gold'] = gold_cluster_to_span_relations(relations['gold_clusters2'],
                                                                     relations['gold_relations'], self.labels)

            output['pred'] = [None for x in relations['gold_relations']]
            output['gold'] = [None for x in relations['gold_relations']]
        else:
            output['loss'] = 0  # (trainer skips minibatch if zero)

            output['span-rel-pred'] = [None for x in relations['gold_relations']]
            output['span-rel-gold'] = [None for x in relations['gold_relations']]

            output['pred'] = [None for x in relations['gold_relations']]
            output['gold'] = [None for x in relations['gold_relations']]

        return output['loss'], output

    def create_metrics(self):
        return [MetricSpanRelationF1x(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []


class LossRelationsX(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsX, self).__init__()
        self.name = name
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config.get('normalize', True) else config['weight']
        self.debug = config['debug']
        print('LossRelationsX: weight={} (fix-norm)'.format(self.weight))
        self.evaluate_mentionwise_predictions = config['mentionwise']

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']

        if self.enabled:
            mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(-1)).sum()
        else:
            obj = 0  # (trainer skips minibatch if zero)

        output['loss'] = obj

        if self.enabled:
            mapping = create_mapping(pred_spans, coref['pred']).to(mention_scores.device)
            concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

            # only for debugging
            if mention_targets is not None:
                concept_lengths = [len(x) for x in coref['pred']]
                mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
                output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                    clusters, triples in zip(coref['pred'], mytargets)]

            if predict:
                if mention_scores is None:
                    output['pred'] = [[] for x in coref['pred']]
                else:
                    pred_mentions = (mention_scores > 0).float()
                    pred_concepts = sum_scores(pred_mentions, mapping)
                    pred_concepts = (pred_concepts > 0).float()

                    concept_lengths = [len(x) for x in coref['pred']]
                    predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                    output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                      clusters, triples in zip(coref['pred'], predictions)]

                output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                                  clusters, triples in zip(relations['gold_clusters2'], relations['gold_relations'])]

                if self.evaluate_mentionwise_predictions:
                    output['span-rel-pred'] = decode_span_relations(mention_scores, pred_spans, self.labels)
                    output['span-rel-gold'] = gold_cluster_to_span_relations(relations['gold_clusters2'],
                                                                             relations['gold_relations'], self.labels)
                else:
                    output['span-rel-pred'] = [None for x in relations['gold_relations']]
                    output['span-rel-gold'] = [None for x in relations['gold_relations']]
        else:
            output['pred'] = [None for x in relations['gold_relations']]
            output['gold'] = [None for x in relations['gold_relations']]
            output['span-rel-pred'] = [None for x in relations['gold_relations']]
            output['span-rel-gold'] = [None for x in relations['gold_relations']]

        return output['loss'], output

    def create_metrics(self):
        if self.enabled:
            metrics = [
                MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricConceptRelationToMentionsF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)
            ]
            if self.evaluate_mentionwise_predictions:
                metrics.append(
                    MetricSpanRelationF1x(self.name, self.labels, verbose=self.debug)
                )
            return metrics
        else:
            return []


class LossRelations(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelations, self).__init__()
        self.name = name
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels)
        self.debug = config['debug']

    def forward(self, mention_scores, mention_targets, mention_lengths, mention_mask, mapping, coref, relations,
                predict=False):
        output = {}

        if self.enabled and mention_targets is not None:
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(
                -1)).sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

        if mention_targets is not None:
            concept_lengths = [len(x) for x in coref['pred']]
            mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
            output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for clusters, triples
                                in zip(coref['pred'], mytargets)]

        if predict:
            if mention_scores is None:
                output['pred'] = [[] for x in coref['pred']]
            else:
                pred_mentions = (mention_scores > 0).float()
                pred_concepts = sum_scores(pred_mentions, mapping)
                pred_concepts = (pred_concepts > 0).float()

                concept_lengths = [len(x) for x in coref['pred']]
                predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                  clusters, triples in zip(coref['pred'], predictions)]

            output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              clusters, (_, triples, _) in
                              zip(relations['gold_clusters2'], relations['gold_relations'])]

        return output['loss'], output

    def create_metrics(self):
        return [MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []


class LossRelationsLatent(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsLatent, self).__init__()
        self.name = name
        self.num_relations = len(labels)

        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels)
        self.latent = True
        self.old_implementation = False
        self.debug = config['debug']

    def forward(self, mention_scores, mention_targets, mention_lengths, mention_mask, mapping, coref, relations,
                predict=False):
        output = {}

        if mention_targets is not None:
            concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

            if self.latent:
                if self.old_implementation:
                    # not all concept pairs have mention pairs
                    mask = (sum_scores(torch.ones(mention_scores.size()).cuda(), mapping) > 0).float()

                    mention_logits = F.logsigmoid(-mention_scores)  # [-inf, 0]
                    concept_logits = sum_scores(mention_logits, mapping)

                    if self.debug:
                        tmp = (concept_logits * mask == 0).float() * mask
                        print('tmp:', tmp.sum().item())

                    x = concept_logits - 1e-8

                    loss = concept_targets * log1mex(x)
                    loss += (1 - concept_targets) * concept_logits
                    loss *= mask

                    obj = - self.weight * loss.sum() / self.num_relations
                else:
                    mask = (sum_scores(torch.ones(mention_scores.size()).cuda(), mapping) > 0).float()

                    mention_logits = -torch.log1p(torch.exp(mention_scores.double()))
                    concept_logits = sum_scores(mention_logits, mapping.double())
                    concept_logits = concept_logits + (1.0 - mask) * -10000

                    loss = concept_targets * torch.log(-torch.expm1(concept_logits - 1e-100))
                    loss += (1 - concept_targets) * concept_logits
                    loss *= mask

                    obj = - self.weight * loss.sum() / self.num_relations
            else:
                pos_logits = F.logsigmoid(mention_scores)
                neg_logits = F.logsigmoid(-mention_scores)

                pos_logits2 = sum_scores(pos_logits, mapping)
                neg_logits2 = sum_scores(neg_logits, mapping)
                loss2 = concept_targets * pos_logits2 + (1 - concept_targets) * neg_logits2

                obj = - self.weight * loss2.sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        if mention_targets is not None:
            concept_lengths = [len(x) for x in coref['pred']]
            mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
            output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for clusters, triples
                                in zip(coref['pred'], mytargets)]

        if predict:
            if mention_scores is None:
                output['pred'] = [[] for _ in coref['pred']]
            else:
                pred_mentions = (mention_scores > 0).float()
                pred_concepts = sum_scores(pred_mentions, mapping)
                pred_concepts = (pred_concepts > 0).float()

                concept_lengths = [len(x) for x in coref['pred']]
                predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                  clusters, triples in zip(coref['pred'], predictions)]

            output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              clusters, (_, triples, _) in
                              zip(relations['gold_clusters2'], relations['gold_relations'])]

        return output['loss'], output

    def create_metrics(self):
        return [MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []
