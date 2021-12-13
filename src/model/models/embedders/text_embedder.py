import numpy as np
import torch
import torch.nn as nn
from transformers import *

from model.data.embeddings_loader import load_wordembeddings_words, load_wordembeddings_with_random_unknowns, \
    load_wordembeddings
from model.models.misc.nnets import CNNMaxpool
from model.training import settings


class TextFieldEmbedderTokens(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderTokens, self).__init__()
        self.dictionary = dictionaries[config['dict']]
        self.dim = config['dim']
        self.embed = nn.Embedding(self.dictionary.size, self.dim)
        self.dropout = nn.Dropout(config['dropout'], inplace=True)
        self.normalize = 'norm' in config
        self.freeze = config.get('freeze', False)

        if 'embed_file' in config:
            self.init_unknown = config['init_unknown']
            self.init_random = config['init_random']
            self.backoff_to_lowercase = config['backoff_to_lowercase']

            self.load_embeddings(config['embed_file'])
        else:
            print('WARNING: training word vectors from scratch')

        nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
        print('norms: min={} max={} avg={}'.format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))

    def load_all_wordvecs(self, filename):
        print('LOADING ALL WORDVECS')
        words = load_wordembeddings_words(filename)
        for word in words:
            self.dictionary.add(word)
        self.load_embeddings(filename)
        print('DONE')

    def load_embeddings(self, filename):
        if self.init_random:
            embeddings = load_wordembeddings_with_random_unknowns(filename, accept=self.dictionary.word2idx,
                                                                  dim=self.dim,
                                                                  backoff_to_lowercase=self.backoff_to_lowercase)
        else:
            unknown_vec = np.ones((self.dim)) / np.sqrt(self.dim) if self.init_unknown else None

            word_vectors = load_wordembeddings(filename, accept=self.dictionary.word2idx, dim=self.dim,
                                               out_of_voc_vector=unknown_vec)
            if self.normalize:
                norms = np.einsum('ij,ij->i', word_vectors, word_vectors)
                np.sqrt(norms, norms)
                norms += 1e-8
                word_vectors /= norms[:, np.newaxis]

            embeddings = torch.from_numpy(word_vectors)

        device = next(self.embed.parameters()).device
        self.embed = nn.Embedding(self.dictionary.size, self.dim).to(device)
        self.embed.weight.data.copy_(embeddings)
        self.embed.weight.requires_grad = not self.freeze

    def forward(self, inputs):
        return self.dropout(self.embed(inputs))


class TextFieldEmbedderCharacters(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderCharacters, self).__init__()
        self.embedder = TextFieldEmbedderTokens(dictionaries, config['embedder'])
        self.padding = self.embedder.dictionary.lookup('PADDING')
        self.seq2vec = CNNMaxpool(self.embedder.dim, config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.dim_output = self.seq2vec.dim_output
        self.min_word_length = self.seq2vec.max_kernel
        # self.min_word_length = 50
        print('TextFieldEmbedderCharacters:', self.min_word_length)

    def forward(self, characters):
        char_vec = self.embedder(characters)
        # print('char_embed', char_vec.size(), char_vec.sum().item())
        char_vec = self.seq2vec(char_vec)
        return self.dropout(torch.relu(char_vec))


class TextEmbedder(nn.Module):
    def __init__(self, dictionaries, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.dim_output = 0
        if 'char_embedder' in config:
            self.char_embedder = TextFieldEmbedderCharacters(dictionaries, config['char_embedder'])
            self.dim_output += self.char_embedder.dim_output
        if 'text_field_embedder' in config:
            self.word_embedder = TextFieldEmbedderTokens(dictionaries, config['text_field_embedder'])
            self.dim_output += self.word_embedder.dim
        if 'bert_embedder' in config:
            self.bert_embedder = WrapperBERT(dictionaries, config['bert_embedder'])
            self.dim_output += self.bert_embedder.dim_output

    def forward(self, characters, tokens, texts=None):
        outputs = []
        if 'char_embedder' in self.config:
            outputs.append(self.char_embedder(characters))
        if 'text_field_embedder' in self.config:
            outputs.append(self.word_embedder(tokens))
        if 'bert_embedder' in self.config:
            outputs.append(self.bert_embedder(texts))

        return torch.cat(outputs, -1)


def mysentsplitter(tokens, maxlen):
    sentences = []
    begin = 0
    while begin < len(tokens):
        if len(tokens) - begin < maxlen:
            end = len(tokens)
        else:
            end = begin + maxlen
            while end > begin and tokens[end - 1] != '.':
                end -= 1
            if begin == end:
                # print('FAILED TO SPLIT INTO SENTENCES:', tokens[begin:])
                end = begin + maxlen
        sentences.append(tokens[begin:end])
        begin = end
    return sentences


def myencode(tokenizer, orig_tokens):
    bert_tokens = []
    orig_to_tok_map = []

    bert_tokens.append('[CLS]')
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append('[SEP]')

    return bert_tokens, orig_to_tok_map


def pad_tensors(instances):
    maxlen = max([x.size()[1] for x in instances])
    out = []
    for instance in instances:
        if instance.size()[1] < maxlen:
            instance = torch.cat(
                (instance, torch.zeros(1, maxlen - instance.size()[1], instance.size()[2]).to(settings.device)), 1)
        out.append(instance)
    return torch.cat(out, 0)


class CombineConcat(nn.Module):
    def __init__(self):
        super(CombineConcat, self).__init__()

    def forward(self, list_of_tensors):
        return torch.cat(list_of_tensors, -1)


class WrapperBERT(nn.Module):

    def __init__(self, dictionaries, config):
        super(WrapperBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.layers = config['layers']
        self.max_bert_length = config['max_length']

        if config['combine'] == 'concat':
            self.out = CombineConcat()
            self.dim_output = 768 * len(self.layers)
        else:
            raise BaseException('no such module')

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        instances = []
        for text in texts:
            reps = [list() for _ in self.layers]

            for sentence in mysentsplitter(text, self.max_bert_length):
                tokens, orig_to_tok_map = myencode(self.tokenizer, sentence)

                input_ids = torch.LongTensor(self.tokenizer.encode(tokens)).unsqueeze(0).to(settings.device)
                outputs = self.model(input_ids)
                all_hidden_states, all_attentions = outputs[-2:]

                indices = torch.LongTensor(orig_to_tok_map).to(settings.device)
                for rep, l in zip(reps, self.layers):
                    rep.append(torch.index_select(all_attentions[l].detach(), 1, indices))

            instances.append([torch.cat(rep, 1) for rep in reps])
        # transpose
        instances = list(map(list, zip(*instances)))
        # pad layers
        instances = [pad_tensors(x).detach() for x in instances]
        output = self.out(instances)
        return output
