import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.misc.misc import MyGate, overwrite_spans
from model.models.misc.span_pair_scorers import create_pair_scorer


class ModuleAttentionProp(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleAttentionProp, self).__init__()
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.att_prop = config['att_prop']

        print('ModuleAttentionProp(ap={})'.format(self.att_prop))

        self.attention = create_pair_scorer(dim_span, 1, config, span_pair_generator)
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        if self.att_prop > 0:

            for _ in range(self.att_prop):
                scores = self.attention(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                probs = F.softmax(scores - (1.0 - square_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs, update)
                update = self.gate(update, ctxt)

            update_filtered['span_vecs'] = update

            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update)

        return update_all, update_filtered
