import torch
import torch.nn as nn


def create_pair_scorer(dim_input, dim_output, config, span_pair_generator):
    scorer_type = config.get('scorer_type', 'opt-ff-pairs')

    if scorer_type == 'ff-pairs':
        return FFpairs(dim_input, dim_output, config, span_pair_generator)
    elif scorer_type == 'opt-ff-pairs':
        return OptFFpairs(dim_input, dim_output, config, span_pair_generator)
    elif scorer_type == 'dot-pairs':
        return DotPairs(dim_input, dim_output, config, span_pair_generator)
    else:
        raise BaseException('no such pair scorer:', scorer_type)


class FFpairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(FFpairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.span_pair_generator = span_pair_generator
        self.scorer = nn.Sequential(
            nn.Linear(self.span_pair_generator.dim_output, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, dim_output)
        )

    def forward(self, span_vecs, span_begin, span_end):
        pairs = self.span_pair_generator(span_vecs, span_begin, span_end)
        return self.scorer(pairs)


class OptFFpairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(OptFFpairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.span_pair_generator = span_pair_generator
        self.left = nn.Linear(dim_input, hidden_dim)
        self.right = nn.Linear(dim_input, hidden_dim)
        self.prod = nn.Linear(dim_input, hidden_dim)
        self.dist = nn.Linear(span_pair_generator.dim_distance_embedding, hidden_dim)
        self.dp1 = nn.Dropout(hidden_dp)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dp2 = nn.Dropout(hidden_dp)
        self.out = nn.Linear(hidden_dim, dim_output)

    def forward(self, span_vecs, span_begin, span_end):
        p = self.span_pair_generator.get_product_embedding(span_vecs)
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)

        h = self.left(span_vecs).unsqueeze(-2) + self.right(span_vecs).unsqueeze(-3) + self.prod(p) + self.dist(d)
        h = self.dp1(torch.relu(h))
        h = self.layer2(h)
        h = self.dp2(torch.relu(h))
        return self.out(h)


class DotPairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(DotPairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.left = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.right = nn.Sequential(
            nn.Linear(dim_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dp),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, span_vecs, span_begin, span_end):
        l = self.left(span_vecs)  # [batch, length, dim_hidden]
        r = self.right(span_vecs)  # [batch, length, dim_hidden]
        s = torch.matmul(l, r.permute(0, 2, 1))
        return s.unsqueeze(-1)
