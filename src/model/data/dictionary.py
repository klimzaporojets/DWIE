import gzip
import json

import numpy as np


def load_dictionary(config, path, name):
    type = config['type']
    filename = config['filename']
    filename = filename if filename.startswith('/') else '{}/{}'.format(path, filename)

    if type == 'word2vec':
        # print('init {} with {}'.format(name, filename))
        dictionary = Dictionary(filename)
    elif type == 'spirit':
        dictionary = Dictionary()
        dictionary.load_spirit_dictionary(filename, config['threshold'])
    elif type == 'vocab':
        dictionary = Dictionary()
        dictionary.load_wordpiece_vocab(filename)
    elif type == 'json':
        dictionary = Dictionary()
        dictionary.load_json(filename)
    else:
        raise BaseException('no such type', type)

    return dictionary


def create_dictionaries(config, training):
    path = config['path']

    print('Loading dictionaries (training={})'.format(training))

    if 'dictionaries' in config:
        dictionaries = {}
        for name, dict_config in config['dictionaries'].items():
            if training:
                if 'init' in dict_config:
                    dictionary = load_dictionary(dict_config['init'], path, name)
                    # print('init {}: size={}'.format(name, dictionary.size))
                else:
                    # print('init {} (blank)'.format(name))
                    dictionary = Dictionary()
            else:
                dictionary = load_dictionary(dict_config, path, name)
                print('load {}: size={}'.format(name, dictionary.size))

            dictionary.prefix = dict_config['prefix'] if 'prefix' in dict_config else ''

            if 'rewriter' in dict_config:
                if dict_config['rewriter'] == 'lowercase':
                    dictionary.rewriter = lambda t: t.lower()
                elif dict_config['rewriter'] == 'none':
                    # print('rewriter: none')
                    pass
                else:
                    raise BaseException('no such rewriter', dict_config['rewriter'])

            if 'append' in dict_config:
                for x in dict_config['append']:
                    idx = dictionary.add(x)
                    # print('   add token', x, '->', idx)

            if 'unknown' in dict_config:
                dictionary.set_unknown_token(dict_config['unknown'])

            if 'debug' in dict_config:
                dictionary.debug = dict_config['debug']

            if 'update' in dict_config:
                dictionary.update = dict_config['update']

            if not training:
                dictionary.update = False

            # print('   update:', dictionary.update)
            # print('   debug:', dictionary.debug)

            dictionaries[name] = dictionary

        return dictionaries
    else:
        print('WARNING: using wikipedia dictionary')
        words = Dictionary()
        entities = Dictionary()

        words.set_unknown_token('UNKNOWN')
        words.load_spirit_dictionary('data/tokens.dict', 5)
        entities.set_unknown_token('UNKNOWN')
        entities.load_spirit_dictionary('data/entities.dict', 5)
        return {
            'words': words,
            'entities': entities
        }


def load_word2vec_text(filename):
    with gzip.open(filename, 'r') as f:
        vec_n, vec_size = map(int, f.readline().split())

        word2idx = {}
        matrix = np.zeros((vec_n, vec_size), dtype=np.float32)

        for n in range(vec_n):
            line = f.readline().split()
            word = line[0].decode('utf-8')
            vec = np.array([float(x) for x in line[1:]], dtype=np.float32)
            word2idx[word] = n
            matrix[n] = vec
        return word2idx, matrix, vec_n


class Dictionary:

    def __init__(self, filename=None):
        self.rewriter = lambda t: t
        self.debug = False
        self.token_unknown = -1
        self.update = True
        self.prefix = ''
        self.tmp_unknown = None

        self.clear()

        if filename is not None:
            print('loading %s' % filename)
            self.load_embedding_matrix_text(filename)
            print('done.')
            print()

    def clear(self):
        self.word2idx = {}
        self.matrix = False
        self.size = 0
        self.out_of_voc = 0
        self.oov = set()

        if self.tmp_unknown is not None:
            self.token_unknown = self.lookup(self.tmp_unknown)

    def load_spirit_dictionary(self, filename, threshold_doc_freq=0):
        self.update = True
        with open(filename) as file:
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 3:
                    df, tf, term = data
                    if int(df) >= threshold_doc_freq:
                        self.lookup(term)
        self.update = False

    def load_wordpiece_vocab(self, filename):
        self.update = True
        with open(filename) as file:
            for line in file:
                term, _ = line.split('\t')
                self.lookup(term)
        self.update = False

    def load_json(self, filename):
        with open(filename) as file:
            data = json.load(file)
            if isinstance(data, (list,)):
                for idx, word in enumerate(data):
                    if self.lookup(word) != idx:
                        print('WARNING: invalid dictionary')
            else:
                for word, idx in data.items():
                    if self.lookup(word) != idx:
                        print('WARNING: invalid dictionary')

    def load_embedding_matrix_text(self, filename):
        self.word2idx, self.matrix, self.size = load_word2vec_text(filename)
        self.update = False

    def lookup(self, token):
        token = self.prefix + self.rewriter(token)
        if not token in self.word2idx:
            if self.update:
                self.word2idx[token] = self.size
                self.size += 1
            else:
                if self.debug:
                    print('oov: \'{}\' -> {}'.format(token, self.token_unknown))
                self.out_of_voc += 1
                return self.token_unknown
        return self.word2idx[token]

    def add(self, token):
        if not token in self.word2idx:
            self.word2idx[token] = self.size
            self.size += 1
        return self.word2idx[token]

    def set_unknown_token(self, unknown_token):
        self.tmp_unknown = unknown_token
        self.token_unknown = self.word2idx[self.prefix + unknown_token]
        print(self.get(self.token_unknown), '->', self.token_unknown)

    def write(self, filename):
        import json
        with open(filename, 'w') as file:
            json.dump(self.word2idx, file)

    def get(self, index):
        for word, idx in self.word2idx.items():
            if idx == index:
                return word
        return None

    def tolist(self):
        list = [None] * self.size
        for word, idx in self.word2idx.items():
            list[idx] = word
        return list
