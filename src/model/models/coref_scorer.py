import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.misc.misc import MyGate, overwrite_spans
from model.models.misc.span_pair_scorers import OptFFpairs


def coref_add_scores(coref_scores, filtered_prune_scores):
    scores_left = filtered_prune_scores
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)
    coref_scores = coref_scores + scores_left + scores_right

    # zero-out self references (without this pruner doesn't work)
    eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
    coref_scores = coref_scores * (1.0 - eye)
    return coref_scores


class ModuleCorefScorer(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefScorer, self).__init__()
        self.coref_prop = config['coref_prop']
        self.update_coref_scores = config['update_coref_scores']

        print('ModuleCorefScorer(cp={})'.format(self.coref_prop))

        self.coref_pruner = coref_pruner
        self.coref = OptFFpairs(dim_span, 1, config, span_pair_generator)
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        triangular_mask = filtered_spans['triangular_mask']

        coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        if self.coref_prop > 0:

            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs, update)
                update = self.gate(update, ctxt)

                if self.update_coref_scores:
                    coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                    coref_scores = coref_add_scores(coref_scores, self.coref_pruner(update))

            update_filtered['span_vecs'] = update
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update)

        return update_all, update_filtered, coref_scores


class ModuleCorefBasicScorer(nn.Module):
    """
    Without graph propagation
    """

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefBasicScorer, self).__init__()

        print('ModuleCorefBasicScorer')
        self.scorer = OptFFpairs(dim_span, 1, config, span_pair_generator)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        coref_scores = self.scorer(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        return all_spans, filtered_spans, coref_scores
