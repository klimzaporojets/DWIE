import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils


def collate_character(batch, maxlen, padding, min_word_len=0):
    seqlens = [len(x) for x in batch]
    max_word_len = max([len(w) for sentence in batch for w in sentence])
    maxlen = min(maxlen, max_word_len)
    maxlen = max(maxlen, min_word_len)

    output = torch.LongTensor(len(batch), max(seqlens), maxlen)
    output[:, :, :] = padding
    for i, sentence in enumerate(batch):
        for pos, token in enumerate(sentence):
            token_len = len(token)
            if token_len < maxlen:
                output[i, pos, :len(token)] = torch.from_numpy(np.array(token, dtype=np.long))
            else:
                output[i, pos, :] = torch.from_numpy(np.array(token[0:maxlen], dtype=np.long))
    return output


def main_collate(model, batch, device):
    batch.sort(key=lambda x: x['tokens'].size()[0], reverse=True)

    sequence_lengths = torch.LongTensor([x['tokens'].size()[0] for x in batch])
    characters = collate_character([x['characters'] for x in batch], 50, model.embedder.char_embedder.padding,
                                   min_word_len=model.embedder.char_embedder.min_word_length)
    tokens = rnn_utils.pad_sequence([x['tokens'] for x in batch], batch_first=True)
    last_idx = max([len(x['tokens']) for x in batch]) - 1
    indices = rnn_utils.pad_sequence([x['tokens-indices'] for x in batch], batch_first=True, padding_value=last_idx)

    inputs = {
        'tokens': tokens.to(device),
        'characters': characters.to(device),
        'sequence_lengths': sequence_lengths.to(device),
        'token_indices': indices.to(device)
    }

    gold_spans = [[(m[0], m[1]) for m in x['spans']] for x in batch]

    if 'gold_clusters' in batch[0]:
        gold_clusters = [x['gold_clusters'] for x in batch]
    else:
        # TODO: move to cpn utility .py (or remove)
        gold_clusters = []
        for spans, m2c in zip(gold_spans, [x['mention2concept'] for x in batch]):
            clusters = [list() for _ in range(m2c[0])]
            for mention, concept in zip(m2c[3], m2c[2]):
                clusters[concept].append(spans[mention])
            gold_clusters.append(clusters)

    metadata = {
        'identifiers': [x['id'] for x in batch],
        'tokens': [x['text'] for x in batch],
        'content': [x['content'] for x in batch],
        'begin': [x['begin'] for x in batch],
        'end': [x['end'] for x in batch]
    }
    metadata['gold_tags_indices'] = [x['gold_tags_indices'] for x in batch]
    metadata['gold_spans'] = gold_spans
    metadata['gold_m2i'] = [x['clusters'] for x in batch]

    relations = {
        'gold_spans': gold_spans,
        'gold_m2i': [x['clusters'] for x in batch],
        'gold_clusters2': gold_clusters
    }

    if 'relations' in batch[0]:
        # old: remove the dimension
        relations['gold_relations'] = [x['relations'][1] for x in batch]
        relations['num_concepts'] = [x['relations'][0][0] for x in batch]
    else:
        relations['gold_relations'] = [x['relations2'] for x in batch]
        relations['num_concepts'] = [x['num_concepts'] for x in batch]

    return {
        'inputs': inputs,
        'relations': relations,
        'metadata': metadata
    }
