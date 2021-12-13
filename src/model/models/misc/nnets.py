import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CNNMaxpool(nn.Module):

    def __init__(self, dim_input, config):
        super(CNNMaxpool, self).__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(dim_input, config['dim'], k) for k in config['kernels']])
        self.dim_output = config['dim'] * len(config['kernels'])
        self.max_kernel = max(config['kernels'])

    def forward(self, inputs):
        inp = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        inp = inp.transpose(1, 2)
        outputs = []
        for cnn in self.cnns:
            maxpool, _ = torch.max(cnn(inp), -1)
            outputs.append(maxpool)
        outputs = torch.cat(outputs, -1)
        result = outputs.view(inputs.size(0), inputs.size(1), -1)
        return result


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def create_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'glu':
        return nn.GLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise BaseException('no such activation function:', name)


class Wrapper1(nn.Module):

    def __init__(self, label, module, dim_output=None):
        super(Wrapper1, self).__init__()
        self.label = label
        self.module = module
        self.dim_output = module.dim_output if dim_output is None else dim_output

    def forward(self, inputs):
        outputs = self.module(inputs)
        norm_inputs = inputs.norm().item()
        norm_outputs = outputs.norm().item()
        print('forward {}: {} / {} = {}'.format(self.label, norm_outputs, norm_inputs, norm_outputs / norm_inputs))
        return outputs


class ResLayerX(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayerX, self).__init__()
        self.layer = Wrapper1('res', FeedForward(dim_input, config['layer']))
        self.out = nn.Linear(self.layer.dim_output, dim_input)

    def forward(self, tensor):
        return tensor + self.out(self.layer(tensor))


class ResLayer(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayer, self).__init__()
        self.dp = nn.Dropout(config['dropout'])
        self.input = nn.Linear(dim_input, config['dim'])
        self.fnc = create_activation_function(config['actfnc'])
        self.output = nn.Linear(config['dim'], dim_input)

    def forward(self, tensor):
        h = self.dp(tensor)
        h = self.input(h)
        h = self.fnc(h)
        h = self.output(h)
        return tensor + h


class FeedForward(nn.Module):

    def __init__(self, dim_input, config):
        super(FeedForward, self).__init__()
        self.dim_output = dim_input
        self.layers = []

        if 'type' not in config:
            self.create_default(config)
        elif config['type'] == 'ffnn':
            self.create_ffnn(config)
        elif config['type'] == 'res':
            self.create_res(config)
        elif config['type'] == 'resnet':
            self.create_resnet(config)
        elif config['type'] == 'glu':
            self.create_glu(config)
        else:
            raise BaseException('no such type: ', config['type'])

        self.layers = nn.Sequential(*self.layers)

    def create_default(self, config):
        if config['ln']:
            self.layers.append(LayerNorm(self.dim_output))
        if config['dropout'] != 0.0:
            self.layers.append(nn.Dropout(config['dropout']))

    def create_ffnn(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, dim))
            if 'actfnc' in config:
                self.layers.append(create_activation_function(config['actfnc']))
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def create_glu(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, 2 * dim))
            self.layers.append(nn.GLU())
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def create_res(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayerX(self.dim_output, config))

    def create_resnet(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayer(self.dim_output, config))

    def forward(self, tensor):
        return self.layers(tensor)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


seq2seq_modules = {}


def seq2seq_register(name, factory):
    print('register', name, factory)
    seq2seq_modules[name] = factory


def seq2seq_create(dim_input, config):
    if config['type'] == 'lstm' or config['type'] == 'gru':
        return Seq2seq(dim_input, config)
    elif config['type'] in seq2seq_modules:
        return seq2seq_modules[config['type']](dim_input, config)
    else:
        raise BaseException('no such type', config['type'])


class Seq2seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2seq, self).__init__()
        if 'i_dp' in config:
            self.idp = nn.Dropout(config['i_dp'])
        else:
            self.idp = nn.Sequential()

        if config['type'] == 'lstm':
            self.rnn = nn.LSTM(dim_input, config['dim'], bidirectional=True, num_layers=config['layers'],
                               dropout=config['dropout'], batch_first=True)
        elif config['type'] == 'gru':
            self.rnn = nn.GRU(dim_input, config['dim'], bidirectional=True, num_layers=config['layers'],
                              dropout=config['dropout'], batch_first=True)

        self.dim_output = config['dim'] * 2

        self.concat_input_output = config['concat_input_output']
        if self.concat_input_output:
            self.dim_output += dim_input

        self.dim = self.dim_output

    def forward(self, inputs, seqlens, indices=None):
        # print('inputs:', inputs.size())
        inputs = self.idp(inputs)
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens.cpu(), batch_first=True)
        # packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        if self.concat_input_output:
            outputs = torch.cat((outputs, inputs), -1)

        return outputs


class Seq2Seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2Seq, self).__init__()
        self.module = seq2seq_create(dim_input, config)
        self.dim_output = self.module.dim_output

    def forward(self, inputs, seqlens, indices=None):
        return self.module(inputs, seqlens, indices)
