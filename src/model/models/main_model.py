import torch
import torch.nn as nn

from model.data.predictions_serializer import convert_to_json
from model.models.att_prop import ModuleAttentionProp
from model.models.coref_loss import LossCoref
from model.models.coref_scorer import ModuleCorefScorer, ModuleCorefBasicScorer
from model.models.embedders.span_embedder import create_span_embedder, SpanPairs
from model.models.embedders.text_embedder import TextEmbedder
from model.models.misc.collate import main_collate
from model.models.misc.misc import batched_index_select
from model.models.misc.nnets import Seq2Seq
from model.models.ner import TaskNER, create_all_spans
from model.models.pruner import MentionPruner
from model.models.rel_scorer import ModuleRelScorer, ModuleRelScorerX, ModuleRelBasicScorer
from model.models.rel_loss import create_task_relations
from model.training import settings


def create_spanprop(model, config):
    if 'spanprop' in config:
        sp_type = config['spanprop']['type']

        if sp_type == 'attprop':
            return ModuleAttentionProp(model.span_extractor.dim_output, model.span_pruner.scorer,
                                       model.span_pair_generator, config['spanprop'])
        else:
            raise BaseException('no such spanprop:', sp_type)
    else:
        return None


def create_corefprop(model, config):
    cp_type = config['corefprop']['type']

    if cp_type == 'none':
        return None
    elif cp_type == 'basic':
        return ModuleCorefBasicScorer(model.span_extractor.dim_output, model.span_pruner.scorer,
                                      model.span_pair_generator,
                                      config['corefprop'])
    elif cp_type == 'default':
        return ModuleCorefScorer(model.span_extractor.dim_output, model.span_pruner.scorer, model.span_pair_generator,
                                 config['corefprop'])
    else:
        raise BaseException('no such corefprop:', cp_type)


def create_relprop(model, config):
    rp_type = config['relprop']['type']

    if rp_type == 'none':
        return None
    elif rp_type == 'basic':
        return ModuleRelBasicScorer(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                                    config['relprop'])
    elif rp_type == 'default':
        return ModuleRelScorer(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                               config['relprop'])
    elif rp_type == 'default-x':
        return ModuleRelScorerX(model.span_extractor.dim_output, model.span_pair_generator, model.relation_labels,
                                config['relprop'])
    else:
        raise BaseException('no such relprop:', rp_type)


class MainModel(nn.Module):

    def __init__(self, dictionaries, config):
        super(MainModel, self).__init__()
        self.random_embed_dim = config['random_embed_dim']
        self.max_span_length = config['max_span_length']
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.rel_after_coref = config['rel_after_coref']
        self.debug_memory = False

        self.embedder = TextEmbedder(dictionaries, config['text_embedder'])
        self.emb_dropout = nn.Dropout(config['lexical_dropout'])
        self.seq2seq = Seq2Seq(self.embedder.dim_output + self.random_embed_dim, config['seq2seq'])

        self.span_extractor = create_span_embedder(self.seq2seq.dim_output, self.max_span_length,
                                                   config['span-extractor'])

        self.span_pruner = MentionPruner(self.span_extractor.dim_output, self.max_span_length, config['pruner'])

        self.span_pair_generator = SpanPairs(self.span_extractor.dim_output, config['span-pairs'])

        self.span_prop = create_spanprop(self, config)

        self.coref_scorer = create_corefprop(self, config)

        self.relation_labels = dictionaries['relations'].tolist()
        self.rel_scorer = create_relprop(self, config)

        self.coref_task = LossCoref('coref', config['coref'])

        self.ner_task = TaskNER('tags', self.span_extractor.dim_output, dictionaries['tags-y'], config['ner'])
        self.relation_task = create_task_relations('rels', config['relations'], self.relation_labels)

        if not self.span_pruner.sort_after_pruning and self.pairs.requires_sorted_spans:
            raise BaseException('ERROR: spans MUST be sorted')

    def collate_func(self, datasets, device):
        return lambda x: main_collate(self, x, device)

    def end_epoch(self, dataset_name):
        self.span_pruner.end_epoch(dataset_name)

    def forward(self, inputs, relations, metadata, metrics=[]):
        output = {}

        sequence_lengths = inputs['sequence_lengths']

        if self.debug_memory:
            print('START', sequence_lengths)
            print('(none)  ', torch.cuda.memory_allocated(0) / 1024 / 1024)

        # MODEL MODULES
        embeddings = self.embedder(inputs['characters'], inputs['tokens'], texts=metadata['tokens'])

        embeddings = self.emb_dropout(embeddings)

        if self.random_embed_dim > 0:
            rand_embedding = torch.FloatTensor(embeddings.size(0), embeddings.size(1), self.random_embed_dim).to(
                embeddings.device).normal_(std=4.0)
            rand_embedding = batched_index_select(rand_embedding, inputs['token_indices'])
            embeddings = torch.cat((embeddings, rand_embedding), -1)

        hidden = self.seq2seq(embeddings, sequence_lengths, inputs['token_indices']).contiguous()

        # create span
        span_begin, span_end = create_all_spans(hidden.size(0), hidden.size(1), self.max_span_length)
        span_begin, span_end = span_begin.to(settings.device), span_end.to(settings.device)
        span_mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float()

        # extract span embeddings
        span_vecs = self.span_extractor(hidden, span_begin, span_end, self.max_span_length)

        all_spans = {
            'span_vecs': span_vecs,
            'span_begin': span_begin,
            'span_end': span_end,
            'span_mask': span_mask
        }

        # prune spans
        obj_pruner, all_spans, filtered_spans = self.span_pruner(all_spans, metadata['gold_spans'], sequence_lengths)
        pred_spans = filtered_spans['spans']
        gold_spans = metadata['gold_spans']

        if self.debug_memory:
            print('(pruner)', torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## spanprop (no extra labels)
        if self.span_prop is not None:
            all_spans, filtered_spans = self.span_prop(
                all_spans,
                filtered_spans,
                sequence_lengths
            )

        ## coref
        if self.coref_task.enabled:
            coref_all, coref_filtered, coref_scores = self.coref_scorer(
                all_spans,
                filtered_spans,
                sequence_lengths
            )

        else:
            coref_all = all_spans
            coref_filtered = filtered_spans
            coref_scores = None
            coref_targets = None

        if not self.rel_after_coref:
            coref_all = all_spans
            coref_filtered = filtered_spans

        if self.debug_memory:
            print('(coref) ', torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## relations
        if self.relation_task.enabled:
            relation_all, relation_filtered, relation_scores = self.rel_scorer(
                coref_all,
                coref_filtered,
                sequence_lengths
            )

        else:
            relation_all = coref_all
            relation_filtered = coref_filtered
            relation_scores = None
            relation_targets = None

        if self.debug_memory:
            print('(rels)  ', torch.cuda.memory_allocated(0) / 1024 / 1024)

        # LOSS FUNCTIONS

        ## ner
        ner_obj, output['tags'] = self.ner_task(
            relation_all,
            sequence_lengths,
            metadata['gold_tags_indices']
        )

        ner_spans = [list(set([(begin, end - 1) for begin, end, _ in spans])) for spans in
                     output['tags']['pred']]

        ## coref
        coref_obj, output['coref'] = self.coref_task(
            coref_scores,
            gold_m2i=metadata['gold_m2i'],
            pred_spans=pred_spans,
            gold_spans=gold_spans,
            predict=True,
            pruner_spans=relation_filtered['enabled_spans'],
            ner_spans=ner_spans
        )

        ## relations
        rel_obj, output['rels'] = self.relation_task(
            relation_filtered,
            relation_scores,
            relations,
            output['coref'],
            predict=not self.training
        )

        for m in metrics:
            if m.task in output:
                m.update2(output[m.task], metadata)

        if self.debug_memory:
            print('(loss)  ', torch.cuda.memory_allocated(0) / 1024 / 1024)

        return obj_pruner + coref_obj + ner_obj + rel_obj, output

    def predict(self, inputs, relations, metadata, metrics=[]):
        loss, output = self.forward(inputs, relations, metadata, metrics)
        return loss, self.decode(metadata, output)

    def create_metrics(self):
        return self.coref_task.create_metrics() + self.ner_task.create_metrics() + self.relation_task.create_metrics()

    def write_model(self, filename):
        return

    def load_model(self, filename, config, to_cpu=False):
        return

    def decode(self, metadata, outputs):
        predictions = []

        for identifier, content, begin, end, ner, coref, concept_rels, span_rels in zip(metadata['identifiers'],
                                                                                        metadata['content'],
                                                                                        metadata['begin'],
                                                                                        metadata['end'],
                                                                                        outputs['tags']['pred'],
                                                                                        outputs['coref']['pred'],
                                                                                        outputs['rels']['pred'],
                                                                                        outputs['rels'][
                                                                                            'span-rel-pred']):
            predictions.append(
                convert_to_json(identifier, content, begin.tolist(), end.tolist(), ner, coref, concept_rels, span_rels,
                                singletons=self.coref_task.singletons))

        return predictions
