import torch
import torch.nn as nn

from model.metrics.f1 import MetricSpanNER
from model.metrics.objective import MetricObjective
from model.models.misc.nnets import FeedForward


def create_all_spans(batch_size, length, width):
    b = torch.arange(length, dtype=torch.long)
    w = torch.arange(width, dtype=torch.long)
    e = b.unsqueeze(-1) + w.unsqueeze(0)
    b = b.unsqueeze(-1).expand_as(e)

    b = b.unsqueeze(0).expand((batch_size,) + b.size())
    e = e.unsqueeze(0).expand((batch_size,) + e.size())
    return b, e


def create_span_targets(ref, instances):
    targets = torch.zeros(ref.size())
    max_span_length = targets.size(2)
    for i, spans in enumerate(instances):
        for begin, end, label in spans:
            if end - begin < max_span_length:
                targets[i, begin, end - begin, label] = 1.0
    return targets


def decode_span_predictions(logits, labels):
    predictions = torch.nonzero(logits > 0)
    preds = [list() for _ in range(logits.size(0))]
    if predictions.size(0) < logits.size(0) * logits.size(1):
        for batch, begin, width, l in predictions.tolist():
            preds[batch].append((begin, begin + width + 1, labels[l]))
    return preds


class TaskNER(nn.Module):

    def __init__(self, name, dim_span, dictionary, config):
        super(TaskNER, self).__init__()
        self.name = name
        self.enabled = config['enabled']
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = dictionary.tolist()
        self.weight = config['weight']
        self.add_pruner_scores = config['add_pruner_scores']
        self.add_pruner_loss = config['add_pruner_loss']

        if config['divide_by_number_of_labels']:
            self.weight /= len(self.labels)

        print('TaskNER: weight=', self.weight, 'add_pruner_scores=', self.add_pruner_scores)

        self.net = FeedForward(dim_span, config['network'])
        self.out = nn.Linear(self.net.dim_output, dictionary.size)

    def forward(self, spans_all, sequence_lengths, gold_tags_indices):
        output = {}

        if self.enabled:
            span_vecs = spans_all['span_vecs']
            span_end = spans_all['span_end']

            logits = self.out(self.net(span_vecs))

            if self.add_pruner_scores:
                logits = logits + spans_all['span_scores']

            mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().unsqueeze(-1)

            span_targets = create_span_targets(logits, gold_tags_indices).to(logits.device)

            obj = self.loss(logits, span_targets) * mask

            if self.add_pruner_loss:
                pruner_target = (span_targets.sum(-1) > 0).float().unsqueeze(-1)
                obj_pruner = self.loss(spans_all['span_scores'], pruner_target) * mask
                obj = obj + obj_pruner

            output['loss'] = obj.sum() * self.weight
            output['pred'] = decode_span_predictions(logits * mask, self.labels)
            output['gold'] = [[(b, e + 1, self.labels[l]) for b, e, l in spans] for spans in gold_tags_indices]
        else:
            output['loss'] = 0  # torch.tensor(0.0).cuda() (trainer skips minibatch if zero)
            num_batch = spans_all['span_vecs'].size(0)
            output['pred'] = [[] for x in range(num_batch)]
            output['gold'] = [[] for x in range(num_batch)]

        return output['loss'], output

    def create_metrics(self):
        return [MetricSpanNER(self.name, labels=self.labels), MetricObjective(self.name)] if self.enabled else []
