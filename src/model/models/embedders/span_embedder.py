import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.misc.misc import batched_index_select
from model.models.misc.nnets import FeedForward


def bucket_values(distances: torch.Tensor,
                  num_identity_buckets: int = 4,
                  num_total_buckets: int = 10) -> torch.Tensor:
    """
    Places the given values (designed for distances) into ``num_total_buckets``semi-logscale
    buckets, with ``num_identity_buckets`` of these capturing single values.

    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

    Parameters
    ----------
    distances : ``torch.Tensor``, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: int, optional (default = 4).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : int, (default = 10)
        The total number of buckets to bucket values into.

    Returns
    -------
    A tensor of the same shape as the input, containing the indices of the buckets
    the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (num_identity_buckets - 1)
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def create_span_embedder(dim_input, max_span_length, config):
    se_type = config['type']

    if se_type == 'endpoint':
        return SpanEndpoint(dim_input, max_span_length, config)
    elif se_type == 'average':
        return SpanAverage(dim_input, max_span_length, config)
    else:
        raise BaseException('no such span extractor:', se_type)


class SpanSelfAttention(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanSelfAttention, self).__init__()
        self.ff = FeedForward(dim_input, config['attention'])
        self.out = nn.Linear(self.ff.dim_output, 1)
        self.dim_output = dim_input

    def forward(self, inputs, b, e, max_width):
        w = torch.arange(max_width).to(b.device)
        indices = b.unsqueeze(-1) + w.unsqueeze(0).unsqueeze(0)
        vectors = batched_index_select(inputs, torch.clamp(indices, max=inputs.size(1) - 1))

        mask = indices <= e.unsqueeze(-1)

        scores = self.out(self.ff(vectors)).squeeze(-1)
        scores = scores - (1.0 - mask.float()) * 1e38
        probs = F.softmax(scores, -1)
        output = torch.matmul(probs.unsqueeze(-2), vectors).squeeze(-2)
        return output


class SpanEndpoint(nn.Module):

    def __init__(self, dim_input, max_span_length, config):
        super(SpanEndpoint, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = 2 * dim_input
        self.span_average = config['average']

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            self.dim_output += config['span_embed']

        if self.span_average:
            self.dim_output += dim_input

        if 'ff_dim' in config:
            self.ff = nn.Sequential(
                nn.Linear(self.dim_output, config['ff_dim']),
                nn.ReLU(),
                nn.Dropout(config['ff_dropout'])
            )
            self.dim_output = config['ff_dim']
        else:
            self.ff = nn.Sequential()

    def forward(self, inputs, b, e, max_width):
        b_vec = batched_index_select(inputs, b)
        e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))

        vecs = [b_vec, e_vec]

        if self.span_embed:
            vecs.append(self.embed(e - b))

        if self.span_average:
            vecs.append(span_average(inputs, b, e, max_width))

        vec = torch.cat(vecs, -1)
        return self.ff(vec)


def span_average(inputs, b, e, max_width):
    w = torch.arange(max_width).to(b.device)
    indices = b.unsqueeze(-1) + w.unsqueeze(0).unsqueeze(0)
    vectors = batched_index_select(inputs, torch.clamp(indices, max=inputs.size(1) - 1))

    mask = (indices <= e.unsqueeze(-1)).float()
    lengths = mask.sum(-1)
    probs = mask / lengths.unsqueeze(-1)
    output = torch.matmul(probs.unsqueeze(-2), vectors).squeeze(-2)
    return output


class SpanAverage(nn.Module):

    def __init__(self, dim_input, max_span_length, config):
        super(SpanAverage, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = dim_input

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            self.dim_output += config['span_embed']

    def forward(self, inputs, b, e, max_width):
        output = span_average(inputs, b, e, max_width)

        if self.span_embed:
            emb = self.embed(e - b)
            return torch.cat((output, emb), -1)
        else:
            return output


def span_distance_tokens(span_begin, span_end):
    span_begin = span_begin.view(span_begin.size(0), -1)
    span_end = span_end.view(span_end.size(0), -1)
    span_dist = torch.relu(span_begin.unsqueeze(-1) - span_end.unsqueeze(-2))
    span_dist = span_dist + span_dist.permute(0, 2, 1)

    return span_dist


def span_distance_ordering(span_begin, span_end):
    span_index = torch.arange(span_begin.size(1)).unsqueeze(0)
    span_index = span_index.expand(span_begin.size()[0:2])
    span_dist = torch.abs(span_index.unsqueeze(-1) - span_index.unsqueeze(-2))
    return span_dist.to(span_begin.device)


class SpanPairs(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanPairs, self).__init__()
        self.num_distance_buckets = config['num_distance_buckets']
        self.dim_distance_embedding = config['dim_distance_embedding']
        self.distance_embeddings = nn.Embedding(self.num_distance_buckets,
                                                self.dim_distance_embedding) if self.dim_distance_embedding > 0 else None
        self.dim_output = dim_input * 2 + self.dim_distance_embedding
        self.span_product = config['span_product']

        if self.span_product:
            self.dim_output += dim_input

        if config['distance_function'] == 'tokens':
            self.distance_function = span_distance_tokens
            self.requires_sorted_spans = False
        elif config['distance_function'] == 'ordering':
            self.distance_function = span_distance_ordering
            self.requires_sorted_spans = True
        elif self.dim_distance_embedding > 0:
            raise BaseException('no such distance function')
        else:
            self.requires_sorted_spans = False

    def forward(self, span_vecs, span_begin, span_end):
        num_batch, num_spans, dim_vector = span_vecs.size()
        left = span_vecs.unsqueeze(-2).expand(num_batch, num_spans, num_spans, dim_vector)
        right = span_vecs.unsqueeze(-3).expand(num_batch, num_spans, num_spans, dim_vector)

        tmp = [left, right]

        if self.span_product:
            tmp.append(left * right)

        if self.dim_distance_embedding > 0:
            span_dist = self.distance_function(span_begin, span_end)
            span_dist = bucket_values(span_dist, num_total_buckets=self.num_distance_buckets)
            tmp.append(self.distance_embeddings(span_dist))

        return torch.cat(tmp, -1)

    def get_product_embedding(self, span_vecs):
        num_batch, num_spans, dim_vector = span_vecs.size()
        left = span_vecs.unsqueeze(-2).expand(num_batch, num_spans, num_spans, dim_vector)
        right = span_vecs.unsqueeze(-3).expand(num_batch, num_spans, num_spans, dim_vector)
        return left * right

    def get_distance_embedding(self, span_begin, span_end):
        span_dist = self.distance_function(span_begin, span_end)
        span_dist = bucket_values(span_dist, num_total_buckets=self.num_distance_buckets)
        return self.distance_embeddings(span_dist)
