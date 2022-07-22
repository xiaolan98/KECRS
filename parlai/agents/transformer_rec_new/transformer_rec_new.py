# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.utils import warn_once, padded_tensor, round_sigfigs
from parlai.core.utils import padded_3d
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from .modules import TransformerMemNetModel, TransformerCopyNetModel

from parlai.core.torch_agent import Batch, Output
import torch.nn.functional as F

import torch
import re
import os
import math
import pickle as pkl
import random


warn_once(
    "Public release transformer models are currently in beta. The name of "
    "command line options may change or disappear before a stable release. We "
    "welcome your feedback. Please file feedback as issues at "
    "https://github.com/facebookresearch/ParlAI/issues/new"
)


def add_common_cmdline_args(argparser):
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument('-hid', '--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
    argparser.add_argument('--dropout', type=float, default=0.0,
                           help='Dropout used in Vaswani 2017.')
    argparser.add_argument('--attention-dropout', type=float, default=0.0,
                           help='Dropout used after attention softmax.')
    argparser.add_argument('--relu-dropout', type=float, default=0.0,
                           help='Dropout used after ReLU. From tensor2tensor.')
    argparser.add_argument('--n-heads', type=int, default=2,
                           help='Number of multihead attention heads')
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument('--n-positions', type=int, default=None, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')


class TransformerRecGeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        agent.add_argument("-ne", "--n-entity", type=int)
        agent.add_argument("-nr", "--n-relation", type=int)
        agent.add_argument("-dim", "--dim", type=int, default=128)
        agent.add_argument("-hop", "--n-hop", type=int, default=2)
        agent.add_argument("-kgew", "--kge-weight", type=float, default=1)
        agent.add_argument("-l2w", "--l2-weight", type=float, default=2.5e-6)
        agent.add_argument("-nmem", "--n-memory", type=int, default=32)
        agent.add_argument(
            "-ium", "--item-update-mode", type=str, default="plus_transform"
        )
        agent.add_argument("-uah", "--using-all-hops", type=bool, default=True)
        agent.add_argument('--memory-attention', type=str, default='sqrt',
                           choices=['cosine', 'dot', 'sqrt'],
                           help='similarity for basic attention mechanism'
                                'when using transformer to encode memories')
        agent.add_argument('--share-encoders', type='bool', default=True)
        agent.add_argument('--learn-embeddings', type='bool', default=True,
                           help='learn embeddings')
        agent.add_argument('--embedding-type', type=str, default='random',
                           help='embeddings type')
        agent.add_argument('--model-type', type=str, default='copy_net',
                           choices=['copy_net', 'memory_net', 'transformer'],
                           help='Using memory net or just copy net')
        agent.add_argument('--infuse-loss', type=bool, default=False,
                           help='Whether to use infusion loss')
        agent.add_argument('--fake-copy', type=bool, default=False,
                           help='fake copy network')
        agent.add_argument('--max-length', type=int, default=0,
                           help='Max Length of Generation')
        agent.add_argument('--match', type=bool, default=False,
                           help='Whether to change entity into number')
        agent.add_argument('--same-dim', type=bool, default=False,
                           help='Word embedding dim equal to KG embedding')
        agent.add_argument('--output-suffix', type=str, default=".txt")
        agent.add_argument('--meta-path', type=bool, default=False,
                           help='Add meta path into context')
        agent.add_argument('--embedding-meta', type=str, default='context_voc',
                           choices=['context_voc', 'kg', 'init'],
                           help='The init embedding using in meta-path encoder')
        agent.add_argument('--bag-of-entity', type=bool, default=False,
                           help='Bag of entity loss')
        agent.add_argument('--bag-of-entity-positive-only', type=bool, default=True,
                           help='Bag of entity loss')
        agent.add_argument('--bag-of-entity-voc-embedding', type=bool, default=False,
                           help='Using the user representation to match the vocabulary')
        agent.add_argument('--replace-movie', type=bool, default=True,
                           help='replace the movie by the movie in recommendation system when generating text')

        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerRecGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        self.entity2id = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "entity2entity_id4.pkl"), "rb")
        )
        self.kg = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "movie_kg4.pkl"), "rb")
        )
        self.id2entity = {}
        for e in self.entity2id:
            self.id2entity[self.entity2id[e]] = e
        self.kg_mask = None
        self.kg_mapping = None
        self.dialogue_mask = None
        self.match = opt['match']
        self.model_type = opt['model_type']
        self.infusion_loss = opt['infuse_loss']
        self.same_dim = opt['same_dim']
        self.valid_output = []
        self.valid_input = []
        self.valid_ground_truth = []
        self.max_length = opt["max_length"] if opt['max_length'] > 0 else None
        self.output_suffix = opt["output_suffix"]
        self.meta_path = opt["meta_path"]
        self.embedding_meta = opt["embedding_meta"]
        self.bag_of_entity = opt["bag_of_entity"]
        self.bag_of_entity_positive = opt["bag_of_entity_positive_only"]
        if shared:
            self.kg_voc_criterion = shared['kg_voc_criterion']  # loss in the view of vocabulary
            self.voc_kg_criterion = shared['voc_kg_criterion']  # loss in the view of kg
            self.bag_of_entity_criterion = shared['bag_of_entity_criterion']
        if self.model_type == "memory_net":
            abstracts =pkl.load(
                open(os.path.join(opt["datapath"], "crs", "entity_overview4.pkl"), "rb")
            )
            self.abstracts = [abstract.lower().strip() for abstract in abstracts]
        super().__init__(opt, shared)

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['kg_voc_criterion'] = self.kg_voc_criterion
        shared['voc_kg_criterion'] = self.voc_kg_criterion
        shared['bag_of_entity_criterion'] = self.bag_of_entity_criterion
        return shared

    def build_mask(self):
        kg_mask = torch.zeros(len(self.dict.tok2ind), dtype=torch.uint8)
        kg_mapping = torch.zeros(len(self.dict.tok2ind), dtype=torch.long)
        if self.match:
            entity_set = set([str(self.entity2id[e]) for e in self.entity2id])
        else:
            entity_set = set([str(e) for e in self.entity2id])
        counter1 = 0
        counter2 = 0
        for token in self.dict.tok2ind:
            if token in entity_set:
                kg_mask[self.dict.tok2ind[token]] = 1
                counter1 += 1
                kg_mapping[self.dict.tok2ind[token]] = self.entity2id[token]
        print(counter1, counter2)
        self.kg_mask = kg_mask
        self.kg_mapping = kg_mapping

    def build_model(self, states=None):
        self.dict.add_token('__split__')
        self.dict.freq['__split__'] += 10
        print("ahhhh", len(self.entity2id),
              len(set([e for e in self.entity2id]) & set(self.dict.tok2ind.keys())))
        for entity in self.entity2id:
            if self.match:
                self.dict.add_token(str(self.entity2id[entity]))
                self.dict.freq[str(self.entity2id[entity])] += 10
            else:
                self.dict.add_token(str(entity))
                self.dict.freq[str(entity)] += 10
        self.build_mask()
        if self.model_type == 'copy_net':
            self.model = TransformerCopyNetModel(self.opt, self.dict, self.kg_mask.cuda(), None, self.kg_mapping.cuda())
        elif self.model_type == "memory_net":
            abstract_list = [self._vectorize_text(abstract, False, False, 1024) for abstract in self.abstracts]
            abstract_mask = torch.ones((len(abstract_list), len(self.dict)), dtype=torch.uint8)
            for i, abstract in enumerate(abstract_list):
                for word_id in abstract:
                    if word_id == self.dict.tok2ind[self.dict.default_unk]:
                        continue
                    abstract_mask[i][word_id] = 0
            abstract_vec, _ = padded_tensor(abstract_list, self.NULL_IDX, self.use_cuda)
            self.model = TransformerMemNetModel(
                self.opt, self.dict, abstract_vec, abstract_mask.cuda(), self.kg_mask.cuda(), self.dialogue_mask.cuda()
            )
        else:
            self.model = TransformerCopyNetModel(self.opt, self.dict, self.kg_mask.cuda(), None, self.kg_mapping.cuda())
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            m['loss'] = self.metrics['loss']
            m['kg_voc_loss'] = self.metrics['kg_voc_loss']
            m['bag_of_entity_loss'] = self.metrics['bag_of_entity_loss']
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['nll_loss'] = self.metrics['nll_loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['nll_loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['num_pre'] > 0:
            m['accuracy'] = self.metrics["accuracy"] / self.metrics["num_pre"]
        if not self.is_training:
            m['dist1'], m['dist2'], m['dist3'], m['dist4'], m['dist5'] = self.distinct_metrics()
            with open("./test_output_"+self.model_type+self.output_suffix, "w", encoding="utf-8") as f:
                for output in self.valid_output:
                    f.write(output + "\n")
            with open("./test_input_"+self.model_type+self.output_suffix, "w", encoding="utf-8") as f:
                for output in self.valid_input:
                    f.write(output+"\n")
            with open("./test_ground_truth_"+self.model_type+self.output_suffix, "w", encoding="utf-8") as f:
                for output in self.valid_ground_truth:
                    f.write(output+"\n")
            self.valid_output = []
        if self.metrics["recall@count"] > 0:
            m["recall"] = self.metrics["recall"] / self.metrics["recall@count"]
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def get_one_hop_neighbor(self, root_list):
        one_hop_neighbor = []
        for root in root_list:
            for relation, tail in self.kg[root]:
                if relation in [0, 1, 7, 8, 14, 15]:
                    one_hop_neighbor.append(tail)
        return one_hop_neighbor

    def vectorize(self, obs, history, **kwargs):
        kwargs['add_start'] = False
        kwargs['add_end'] = True
        super().vectorize(obs, history, **kwargs)

        if "text" not in obs:
            return obs
        # match movies and entities
        input_match = list(map(int, obs['label_candidates'][1].split()))
        entities_match = list(map(int, obs['label_candidates'][3].split()))
        label_movies = list(map(int, obs['label_candidates'][2].split()))
        if self.infusion_loss:
            voc_label = [0] * len(self.dict.tok2ind)
            voc_mask = 0
            for i in entities_match + input_match:
                if self.match:
                    voc_label[self.dict.tok2ind[str(i)]] = 1
                else:
                    voc_label[self.dict.tok2ind[self.id2entity[i]]] = 1
            if len(input_match + entities_match) > 0:
                voc_mask = 1
            obs["voc_label"] = voc_label
            obs["voc_mask"] = voc_mask
        if self.bag_of_entity:
            one_hop_neighbor = self.get_one_hop_neighbor(label_movies)
            bag_of_entity_label = [0] * self.opt['n_entity']
            bag_of_entity_mask = 0
            if len(one_hop_neighbor) > 0:
                bag_of_entity_mask = 1
            for one_hop in one_hop_neighbor:
                bag_of_entity_label[one_hop] = 1
            obs["bag_of_entity_label"] = bag_of_entity_label
            obs["bag_of_entity_mask"] = bag_of_entity_mask
        obs["movies"] = input_match + entities_match
        obs["label_movie"] = label_movies
        return obs

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]
        valid_inds, exs = zip(*valid_obs)
        # MOVIE ENTITIES
        movies = None
        if any('movies' in ex for ex in exs):
            movies = [ex.get('movies', []) for ex in exs]
        # label movie
        label_movie = None
        if any('label_movie' in ex for ex in exs):
            label_movie = [ex.get('label_movie', []) for ex in exs]
        if self.infusion_loss:
            # label of vocabulary
            voc_label = None
            if any('voc_label' in ex for ex in exs):
                voc_label = [ex.get('voc_label', []) for ex in exs]
            voc_mask = None
            if any('voc_mask' in ex for ex in exs):
                voc_mask = [ex.get('voc_mask', None) for ex in exs]
            batch.voc_label = torch.tensor(voc_label)
            batch.voc_mask = torch.tensor(voc_mask)
        if self.bag_of_entity:
            bag_of_entity_label = None
            if any('bag_of_entity_label' in ex for ex in exs):
                bag_of_entity_label = [ex.get('bag_of_entity_label', []) for ex in exs]
            batch.bag_of_entity_label = torch.tensor(bag_of_entity_label).float().cuda()
            bag_of_entity_mask = None
            if any('bag_of_entity_mask' in ex for ex in exs):
                bag_of_entity_mask = [ex.get('bag_of_entity_mask', []) for ex in exs]
            batch.bag_of_entity_label = torch.tensor(bag_of_entity_label).float().cuda()
            batch.bag_of_entity_mask = torch.tensor(bag_of_entity_mask).cuda()
        batch.movies = movies
        batch.label_movie = label_movie
        return batch

    def _model_input(self, batch):
        if self.model_type == 'copy_net' or self.model_type == 'transformer':
            return batch.text_vec,
        else:
            return batch.text_vec, batch.abstract_m, batch.abstract_mask

    def build_criterion(self):
        """
        Constructs the loss function. By default torch.nn.CrossEntropyLoss.
        The criterion function should be set to self.criterion.

        If overridden, this model should (1) handle calling cuda and (2)
        produce a sum that can be used for a per-token loss.
        """
        if self.model_type == 'copy_net':
            if self.opt['fake_copy']:
                self.criterion = torch.nn.CrossEntropyLoss(
                    ignore_index=self.NULL_IDX, reduction='sum'
                )
            else:
                self.criterion = torch.nn.NLLLoss(
                    ignore_index=self.NULL_IDX, reduction='sum'
                )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, reduction='sum'
            )
        self.kg_voc_criterion = torch.nn.MSELoss(reduction='none')
        self.voc_kg_criterion = torch.nn.MSELoss(reduction='none')
        self.bag_of_entity_criterion = torch.nn.MSELoss(reduction='none')
        if self.use_cuda:
            self.criterion.cuda()
            self.kg_voc_criterion.cuda()
            self.bag_of_entity_criterion.cuda()

    def _dummy_batch(self, batchsize, maxlen):
        """
        Creates a dummy batch. This is used to preinitialize the cuda buffer,
        or otherwise force a null backward pass after an OOM.
        """
        return Batch(
            text_vec=torch.zeros(batchsize, maxlen).long().cuda(),
            # abstract_m=torch.ones(batchsize, 5, maxlen).long().cuda(),
            label_vec=torch.zeros(batchsize, maxlen).long().cuda(),
            voc_label=torch.zeros(batchsize, len(self.dict.tok2ind)).long().cuda() if self.infusion_loss else None,
            voc_mask=torch.zeros(batchsize).long().cuda() if self.infusion_loss else None,
            bag_of_entity_label=torch.zeros(batchsize, self.opt['n_entity']).cuda() if self.bag_of_entity else None,
            bag_of_entity_mask=torch.zeros(batchsize).long().cuda() if self.bag_of_entity else None,
        )

    def compute_loss(self, batch, return_output=False):
        """
        Computes and returns the loss for the given batch. Easily overridable for
        customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        if self.infusion_loss:
            # calculate kg-vocabulary loss
            if self.same_dim:
                kg_voc_score = F.linear(self.model.user_representation,
                                        self.model.embeddings.weight, self.model.kg_voc.bias)
            else:
                kg_voc_score = F.linear(self.model.dim_align(self.model.user_representation),
                                        self.model.embeddings.weight, self.model.kg_voc.bias)

            kg_voc_loss = \
                torch.sum(self.kg_voc_criterion(kg_voc_score, batch.voc_label.cuda().float()), dim=-1) * batch.voc_mask.cuda()
            kg_voc_loss = torch.mean(kg_voc_loss)

        if self.model_type == 'copy_net' or self.model_type == 'transformer':
            if self.bag_of_entity:
                bag_scores, scores, preds, *_ = model_output
            else:
                scores, preds, *_ = model_output
        else:
            scores, preds, weights, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        gen_loss = self.criterion(score_view, batch.label_vec.view(-1))
        # knowledge_loss = self.criterion(weights, batch.knowledge_label)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum().item()
        correct = ((batch.label_vec == preds) * notnull).sum().item()
        self.metrics['correct_tokens'] += correct
        self.metrics['nll_loss'] += gen_loss.item()
        if self.infusion_loss:
            self.metrics['kg_voc_loss'] += kg_voc_loss.item()
        # self.metrics['knowledge_loss'] += knowledge_loss.item()
        if self.bag_of_entity:
            bag_mask = batch.label_vec.ne(self.NULL_IDX).unsqueeze(-1).repeat(1, 1, bag_scores.size(-1))
            bag_scores = bag_scores * bag_mask
            bag_scores_sum = torch.sum(bag_scores, dim=1)
            if self.bag_of_entity_positive:
                bag_of_entity_loss = torch.sum(
                    -torch.log(F.sigmoid(bag_scores_sum)) * batch.bag_of_entity_label, dim=-1
                ) * batch.bag_of_entity_mask
                bag_of_entity_loss = torch.mean(bag_of_entity_loss)
            else:
                bag_of_entity_loss = torch.sum(
                    self.bag_of_entity_criterion(bag_scores_sum, batch.bag_of_entity_label), dim=-1
                ) * batch.bag_of_entity_mask
                # bag_of_entity_loss = torch.sum(bag_of_entity_loss) / torch.sum(batch.bag_of_entity_mask)
                bag_of_entity_loss = torch.mean(bag_of_entity_loss)
            self.metrics['bag_of_entity_loss'] += bag_of_entity_loss.item()
        self.metrics['num_tokens'] += target_tokens
        gen_loss /= target_tokens  # average loss per token
        if self.infusion_loss:
            if self.bag_of_entity:
                loss = gen_loss + kg_voc_loss * 0.025 + bag_of_entity_loss * 1.5
            else:
                loss = gen_loss + kg_voc_loss * 0.025
        else:
            if self.bag_of_entity:
                loss = gen_loss + 1.5 * bag_of_entity_loss
            else:
                loss = gen_loss

        if return_output:
            # return loss + 0.1 * knowledge_loss, model_output
            return loss, model_output
        else:
            # return loss + 0.1 * knowledge_loss
            return loss

    def distinct_metrics(self):
        outs = [line.strip().split(" ") for line in self.valid_output]

        # outputs is a list which contains several sentences, each sentence contains several words
        unigram_count = 0
        bigram_count = 0
        trigram_count = 0
        quadragram_count = 0
        quintagram_count = 0
        unigram_set = set()
        bigram_set = set()
        trigram_set = set()
        quadragram_set = set()
        quintagram_set = set()
        for sen in outs:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start+1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen)-2):
                trg = str(sen[start]) + ' ' + str(sen[start+1]) + ' ' + str(sen[start+2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(sen)-3):
                quadg = str(sen[start]) + ' ' + str(sen[start+1]) + \
                        ' ' + str(sen[start+2]) + ' ' + str(sen[start+3])
                quadragram_count += 1
                quadragram_set.add(quadg)
            for start in range(len(sen)-4):
                quing = str(sen[start]) + ' ' + str(sen[start+1]) + ' ' + \
                        str(sen[start+2]) + ' ' + str(sen[start+3]) + ' ' + str(sen[start+4])
                quintagram_count += 1
                quintagram_set.add(quing)
        dist1 = len(unigram_set) / len(outs)  # unigram_count
        dist2 = len(bigram_set) / len(outs)  # bigram_count
        dist3 = len(trigram_set)/len(outs)  # trigram_count
        dist4 = len(quadragram_set)/len(outs)  # quadragram_count
        dist5 = len(quintagram_set)/len(outs)  # quintagram_count
        return dist1, dist2, dist3, dist4, dist5

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if getattr(batch, 'movies', None):
            assert hasattr(self.model, 'recommend_model')
            self.model.user_representation, self.model.nodes_feature = \
                self.model.recommend_model.kg_movie_score(batch.movies)
            self.model.user_representation = self.model.user_representation.detach()
            self.model.nodes_feature = self.model.nodes_feature.detach()
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.metrics['loss'] += loss.item()
            self.backward(loss)
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None
        if getattr(batch, 'movies', None):
            assert hasattr(self.model, 'recommend_model')
            self.model.user_representation, self.model.nodes_feature = \
                self.model.recommend_model.kg_movie_score(batch.movies)
            self.model.user_representation = self.model.user_representation.detach()
            self.model.movie_scores = F.linear(self.model.user_representation,
                                               self.model.nodes_feature[:self.model.recommend_model.movie_entity_num],
                                               self.model.recommend_model.output.bias)
            self.model.movie_scores = self.model.movie_scores.detach()
            # self.model.nodes_feature = self.model.nodes_feature.detach()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        elif self.beam_size == 1:
            # greedy decode
            if self.bag_of_entity:
                _, _, preds, *_ = self.model(*self._model_input(batch), bsz=bsz, maxlen=self.max_length)
            else:
                _, preds, *_ = self.model(*self._model_input(batch), bsz=bsz, maxlen=self.max_length)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        cand_choices = None
        # TODO: abstract out the scoring here

        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._model_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        # self.valid_output.extend(preds.detach().cpu().tolist())
        self.valid_output.extend(text)
        input_text = [self._v2t(p).replace("__null__", "") for p in batch.text_vec] if batch.text_vec is not None else None
        label_text = [self._v2t(p).replace("__null__", "") for p in batch.label_vec] if batch.label_vec is not None else None
        self.valid_input.extend(input_text)
        self.valid_ground_truth.extend(label_text)
        return Output(text, cand_choices)
