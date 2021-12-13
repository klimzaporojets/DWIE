import torch
import torch.nn as nn

from model.models.misc.misc import batched_index_select, get_mask_from_sequence_lengths


def indices_to_spans(top_indices, span_lengths, max_span_width):
    b = top_indices // max_span_width
    w = top_indices % max_span_width
    e = b + w
    return [list(zip(b[i, 0:length].tolist(), e[i, 0:length].tolist())) for i, length in
            enumerate(span_lengths.tolist())]


def span_intersection(pred, gold):
    numer = 0
    for p, g in zip(pred, gold):
        numer += len(set(p) & set(g))
    return numer


def create_masks(num_mentions, max_mentions):
    mask = get_mask_from_sequence_lengths(num_mentions, max_mentions).float()
    square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))

    triangular_mask = torch.ones(max_mentions, max_mentions).tril(0).unsqueeze(0).to(num_mentions.device)
    return square_mask, square_mask * triangular_mask


def filter_spans(span_vecs, span_indices):
    tmp = span_vecs.contiguous().view(span_vecs.size(0), -1, span_vecs.size(-1))
    return batched_index_select(tmp, span_indices)


def prune_spans(span_scores, sequence_lengths, sort_after_pruning, prune_ratio=0.2):
    span_lengths = (sequence_lengths * prune_ratio + 1).long()
    span_scores = span_scores.view(span_scores.size(0), -1)
    values, top_indices = torch.topk(span_scores, span_lengths.max().item(), largest=True, sorted=True)
    if sort_after_pruning:
        for b, l in enumerate(span_lengths.tolist()):
            top_indices[b, l:] = span_scores.size(1) - 1
        top_indices, _ = torch.sort(top_indices)
    return top_indices, span_lengths


def create_spans_targets(scores, gold_spans):
    targets = torch.zeros_like(scores)
    max_span_length = scores.size(2)
    for i, spans in enumerate(gold_spans):
        for begin, end in spans:
            if begin is not None and end is not None and end - begin < max_span_length:
                targets[i, begin, end - begin, 0] = 1.0
    return targets


def decode_accepted_spans(scores):
    num_batch = scores.size(0)
    max_span_length = scores.size(2)
    output = [list() for _ in range(num_batch)]
    for batch_idx, span_idx in torch.nonzero((scores.view(num_batch, -1) > 0).float()).tolist():
        begin = span_idx // max_span_length
        length = span_idx % max_span_length
        output[batch_idx].append((begin, begin + length))
    return output


class MentionPruner(nn.Module):

    def __init__(self, dim_span, max_span_length, config):
        super(MentionPruner, self).__init__()
        self.config = config
        self.dim_span = dim_span
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.max_span_length = max_span_length
        self.sort_after_pruning = config['sort_after_pruning']
        self.prune_ratio = config['prune_ratio']
        self.add_pruner_loss = config['add_pruner_loss']
        self.weight = config['weight'] if self.add_pruner_loss else None
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.scorer = nn.Sequential(
            nn.Linear(dim_span, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, 1)
        )

        print('MentionPruner:', self.max_span_length, self.prune_ratio, self.sort_after_pruning, self.add_pruner_loss)
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0

    def create_new(self):
        return MentionPruner(self.dim_span, self.max_span_length, self.config)

    def forward(self, all_spans, gold_spans, sequence_lengths):
        span_vecs = all_spans['span_vecs']
        span_mask = all_spans['span_mask']
        span_begin = all_spans['span_begin']
        span_end = all_spans['span_end']

        prune_scores = self.scorer(span_vecs) - (1.0 - span_mask.unsqueeze(-1)) * 1e4
        span_pruned_indices, span_lengths = prune_spans(prune_scores, sequence_lengths, self.sort_after_pruning,
                                                        prune_ratio=self.prune_ratio)
        pred_spans = indices_to_spans(span_pruned_indices, span_lengths, self.max_span_length)
        square_mask, triangular_mask = create_masks(span_lengths, span_pruned_indices.size(1))
        all_spans['span_scores'] = prune_scores

        self.span_generated += sum([len(x) for x in pred_spans])
        self.span_recall_numer += span_intersection(pred_spans, gold_spans)
        self.span_recall_denom += sum([len(x) for x in gold_spans])

        if self.add_pruner_loss:
            prune_targets = create_spans_targets(prune_scores, gold_spans)
            mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().unsqueeze(-1)
            obj_pruner = (self.loss(prune_scores, prune_targets) * mask).sum() * self.weight
            self.span_loss += obj_pruner.item()
            enabled_spans = decode_accepted_spans(prune_scores)
        else:
            obj_pruner = 0
            enabled_spans = None

        return obj_pruner, all_spans, {
            'prune_indices': span_pruned_indices,
            'span_vecs': filter_spans(span_vecs, span_pruned_indices),
            'span_scores': filter_spans(prune_scores, span_pruned_indices),
            'span_begin': filter_spans(span_begin.view(prune_scores.size()), span_pruned_indices),
            'span_end': filter_spans(span_end.view(prune_scores.size()), span_pruned_indices),
            'span_lengths': span_lengths,
            'square_mask': square_mask,
            'triangular_mask': triangular_mask,
            'spans': pred_spans,
            'enabled_spans': enabled_spans
        }

    def end_epoch(self, dataset_name):
        print('{}-span-generator: {} / {} = {}'.format(dataset_name, self.span_generated, self.span_recall_denom,
                                                       self.span_generated / self.span_recall_denom))
        print('{}-span-recall: {} / {} = {}'.format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                    self.span_recall_numer / self.span_recall_denom))
        print('{}-span-loss: {}'.format(dataset_name, self.span_loss))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0
