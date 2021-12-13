import json
import os
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model.data.tokenizer import TokenizerCPN


class DatasetCPN(Dataset):

    def __init__(self, name, config, dictionaries):
        self.name = name
        self.tokenize = config['tokenize']
        self.tag = config['tag']
        self.dict_words = dictionaries['words']
        self.dict_characters = dictionaries['characters']
        self.dict_tags = dictionaries['tags-y']
        self.dict_relations = dictionaries['relations']
        self.dict_entities = dictionaries.get('entities', None)
        self.shuffle_candidates = config['shuffle_candidates']

        if self.tokenize:
            self.tokenizer = TokenizerCPN()
            # self.tokenizer = TokenizerSimple()
        path = config['filename']

        self.instances = []
        self.number_of_lost_mentions = 0

        print('Loading {} tokenize={} tag={}'.format(path, self.tokenize, self.tag))
        if os.path.isdir(path):
            for filename in tqdm(os.listdir(path)):
                f = os.path.join(path, filename)
                self.load_file(f)
        else:
            self.load_file(path)
        print('done.')

        print('Number of instances in {}: {}.'.format(self.name, len(self)))
        print('Number of mentions lost due to tokenization: {}'.format(self.number_of_lost_mentions))
        print('Shuffle candidates:', self.shuffle_candidates)
        self.print_histogram_of_span_length()

    def get_token_buckets(self, tokens):
        token2idx = {}
        for token in tokens:
            token = token.lower()
            if token not in token2idx:
                token2idx[token] = len(token2idx)
        return [token2idx[token.lower()] for token in tokens]

    def print_histogram_of_span_length(self):
        counter = Counter()
        total = 0
        fail = 0
        for instance in self.instances:
            for begin, end in instance['spans']:
                if begin is None or end is None:
                    fail += 1
                else:
                    counter[end - begin] += 1
                    total += 1

        print('span\tcount\trecall')
        cum = 0
        for span_length in sorted(counter.keys()):
            count = counter[span_length]
            cum += count
            print('{}\t{}\t{}'.format(span_length, count, cum / total))
        print()
        print('failed spans:', fail)

    def load_file(self, filename):
        if filename.endswith('.json'):
            self.load_json(filename)
        elif filename.endswith('.jsonl'):
            self.load_jsonl(filename)
        else:
            raise BaseException('unknown file type:', filename)

    def load_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            if self.tag in data['tags']:
                self.instances.append(self.convert(data))

    def load_jsonl(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                data = json.loads(line.rstrip())
                if self.tag in data['tags']:
                    self.instances.append(self.convert(data))

    def convert(self, data):
        identifier = data['id']
        mentions = data['mentions']
        concepts = data['concepts']

        if self.tokenize:
            tokens = self.tokenizer.tokenize(data['content'])
            begin = [token['offset'] for token in tokens]
            end = [token['offset'] + token['length'] for token in tokens]
            tokens = [token['token'] for token in tokens]
        else:
            tokens = data['tokenization']['tokens']
            begin = data['tokenization']['begin']
            end = data['tokenization']['end']

        if len(tokens) == 0:
            print('WARNING: dropping empty document')
            return

        begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
        end_to_index = {pos: idx for idx, pos in enumerate(end)}

        # this makes life easier
        for concept in concepts:
            concept['mentions'] = []
            if 'link' in concept and concept['link'] is None:
                concept['link'] = 'NILL'
        for mention in mentions:
            if mention['concept'] >= len(concepts):
                raise BaseException('invalid mention concept', mention['concept'], 'in doc', identifier)
            concept = concepts[mention['concept']]
            mention['concept'] = concept
            mention['token_begin'] = begin_to_index.get(mention['begin'], None)
            mention['token_end'] = end_to_index.get(mention['end'], None)
            if mention['token_begin'] is None or mention['token_end'] is None:
                self.number_of_lost_mentions += 1
            concept['mentions'].append(mention)

            if 'candidates' in mention:
                mention['candidates'].append('NILL')
                if self.shuffle_candidates:
                    random.shuffle(mention['candidates'])
            if 'link' in mention and mention['link'] is None:
                mention['link'] = 'NILL'

        token_indices = self.get_token_indices(tokens)
        character_indices = self.get_character_indices(tokens)
        spans = [(mention['token_begin'], mention['token_end']) for mention in data['mentions']]
        gold_clusters = [[(mention['token_begin'], mention['token_end']) for mention in concept['mentions']] for concept
                         in concepts]

        # linker candidates
        linker_candidates = self.get_linker_candidates(data)
        linker_targets = self.get_linker_targets(data)
        linker_gold = self.get_linker_gold(data)

        # TODO: rename variables to be more clear
        return {
            'id': identifier,
            'content': data['content'],
            'begin': torch.IntTensor(begin),
            'end': torch.IntTensor(end),
            'tokens': torch.LongTensor(token_indices),
            'characters': character_indices,
            'tokens-indices': torch.LongTensor(self.get_token_buckets(tokens)),
            'spans': spans,
            'gold_clusters': gold_clusters,
            'gold_tags_indices': self.get_span_tags(mentions),
            'text': tokens,
            'clusters': torch.IntTensor([mention['concept']['concept'] for mention in mentions]),
            'relations2': self.get_relations(data),
            'num_concepts': len(concepts),
            'linker_candidates': linker_candidates,
            'linker_targets': linker_targets,
            'linker_gold': linker_gold
        }

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def get_token_indices(self, tokens):
        return [self.dict_words.lookup(token) for token in tokens]

    def get_character_indices(self, tokens):
        output = []
        for token in tokens:
            token = '<' + token + '>'
            output.append([self.dict_characters.lookup(c) for c in token])
        return output

    def get_span_tags(self, mentions):
        spans = []
        for mention in mentions:
            if mention['token_begin'] is not None and mention['token_end'] is not None:
                spans.extend([(mention['token_begin'], mention['token_end'], self.dict_tags.lookup(tag)) for tag in
                              mention['concept']['tags']])
        return spans

    def get_relations(self, data):
        return [(relation['s'], relation['o'], self.dict_relations.lookup(relation['p'])) for relation in
                data['relations']]

    def get_linker_candidates(self, data):
        # no linking for span: empty candidate list

        if 'annotation::links' in data['tags']:
            candidates = []

            for mention in data['mentions']:
                if is_link_trainable(mention):
                    candidates.append([self.dict_entities.add(c) for c in mention['candidates']])
                else:
                    candidates.append([])
        else:
            candidates = [[] for mention in data['mentions']]

        return candidates

    def get_linker_targets(self, data):
        if 'annotation::links' in data['tags']:
            targets = []

            for mention in data['mentions']:
                if is_link_trainable(mention):
                    if mention['link'] in mention['candidates']:
                        index = mention['candidates'].index(mention['link'])
                    else:
                        index = mention['candidates'].index('NILL')
                else:
                    index = 0

                targets.append(index)
        else:
            # no linking annotation
            targets = [0 for _ in data['mentions']]

        return targets

    def get_linker_gold(self, data):
        gold = []
        for mention in data['mentions']:
            if 'link' in mention:
                gold.append((mention['token_begin'], mention['token_end'], mention['link']))
        return gold


def is_link_trainable(mention):
    if mention['token_begin'] is None or mention['token_end'] is None:
        return False
    return 'candidates' in mention and 'link' in mention
