# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import pickle as pkl
from collections import OrderedDict
import numpy as np

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.core.utils import neginf
from parlai.agents.kecrs.modules import KECRS


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _build_encoder(opt, dictionary, embedding=None, padding_idx=None, reduction=True,
                   n_positions=1024):
    return TransformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        reduction=reduction,
        n_positions=n_positions,
    )


def _build_decoder(opt, dictionary, embedding=None, padding_idx=None,
                   n_positions=1024):
    return TransformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
    )


def get_reduction(tensor, mask, dim=1):
    if dim == 1:
        mask_ = mask.type_as(tensor).unsqueeze(-1).repeat(1, 1, tensor.size(-1))
    elif dim == 2:
        mask_ = mask.type_as(tensor).unsqueeze(-1).repeat(1, 1, 1, tensor.size(-1))
    else:
        raise RuntimeError
    tensor_ = tensor * mask_
    divisor = mask.type_as(tensor).sum(dim=dim).unsqueeze(-1).clamp(min=1)
    output = tensor_.sum(dim=dim) / divisor
    return output


class TransformerMemNetModel(TorchGeneratorModel):
    """Model which takes context, memories, candidates and encodes them"""
    def __init__(self, opt, dictionary, abstract_vec, abstract_mask, kg_mask, dialogue_mask):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.opt = opt
        self.abstract_vec = abstract_vec
        self.abstract_mask = abstract_mask
        self.kg_mask = kg_mask
        self.dialogue_mask = dialogue_mask
        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.dict = dictionary
        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.context_encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )

        if opt.get('share_encoders'):
            self.memory_encoder = TransformerResponseWrapper(
                self.context_encoder, self.context_encoder.out_dim,
            )
        else:
            self.memory_encoder = _build_encoder(
                opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
                n_positions=n_positions,
            )

        self.attender = BasicAttention(dim=2, attn=opt['memory_attention'])
        self.encoder = self.encode_context_memory
        self.decoder = _build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx, n_positions=n_positions
        )
        self.copy_net_abstract = nn.Linear(opt['embedding_size'], len(dictionary))
        self.copy_net_kg = nn.Linear(opt['embedding_size'], len(dictionary))
        self.dim_align = nn.Linear(opt['dim'], opt['embedding_size'])
        self.position_embeddings = nn.Embedding(n_positions, opt['embedding_size'])
        create_position_codes(
            n_positions, opt['embedding_size'], out=self.position_embeddings.weight
        )
        self.position = torch.tensor([i for i in range(n_positions)], dtype=torch.int64, requires_grad=False).cuda()
        entity2entity_id = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "entity2entity_id4.pkl"), "rb")
        )
        kg = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "movie_kg4.pkl"), "rb")
        )
        opt["n_entity"] = len(set(entity2entity_id.values()))
        self.recommend_model = KECRS(opt, opt["n_entity"], 0, opt['dim'], 0, kg)
        state_dict = torch.load('kecrs/kecrs')['model']
        # state_dict = torch.load('../parlai/tasks/crs/kecrs/kecrs')['model']
        self.recommend_model.load_state_dict(state_dict)

    def encode_context_memory(self, context_w, memories_w):
        # [batch, d]
        if context_w is None:
            # it's possible that only candidates were passed into the
            # forward function, return None here for LHS representation
            return None, None

        context_h, context_mask = self.context_encoder(context_w)
        context_h_mean = get_reduction(context_h, context_mask, dim=1)

        if memories_w is None:
            return [], context_h

        if memories_w.dim() == 3:
            oldshape = memories_w.shape
            memories_w = memories_w.reshape(oldshape[0] * oldshape[1], oldshape[2])
        else:
            oldshape = None
        memories_h, memories_mask = self.memory_encoder(memories_w)
        if oldshape is not None:
            memories_h = memories_h.reshape(oldshape[0], oldshape[1], -1, memories_h.size(-1))
            memories_mask = memories_mask.reshape(oldshape[0], oldshape[1], -1)
            assert memories_h.size(2) == oldshape[2]
        memories_h_mean = get_reduction(memories_h, memories_mask, dim=2)
        # memories_h_mean = memories_h_mean.view(oldshape[0], -1, memories_h_mean.size(-1))

        context_h_mean = context_h_mean.unsqueeze(1)
        context_h_mean, weights = self.attender(context_h_mean, memories_h_mean)
        scores, index = weights.max(dim=2, keepdim=True)  # index/ score -> (batch_size * 1 * 1)
        index_1 = index.unsqueeze(-1).repeat(1, 1, memories_h.size(2), memories_h.size(3))
        memories_h_out = torch.gather(memories_h, dim=1, index=index_1).squeeze(1)
        index_2 = index.repeat(1, 1, memories_mask.size(-1))
        memories_mask_out = torch.gather(memories_mask, dim=1, index=index_2).squeeze(1)
        context_memory = torch.cat([context_h, memories_h_out], dim=1)
        context_memory_mask = torch.cat([context_mask, memories_mask_out], dim=1)
        # context_memory_mask_ = context_memory_mask.detach_().cpu().numpy()

        return (context_memory, context_memory_mask), weights.squeeze(1), context_h_mean

    def output(self, decoder_output, mask, context_h_mean):
        score1 = F.linear(decoder_output, self.embeddings.weight).masked_fill(self.dialogue_mask.bool(), -1000000.0)
        # score1 = F.linear(decoder_output, self.embeddings.weight)
        # position = self.position[:decoder_output.size(1)].unsqueeze(0).repeat(decoder_output.size(0), 1)
        # position_ = self.position_embeddings(position)
        # context_h_mean = context_h_mean.unsqueeze(1).repeat(1, decoder_output.size(1), 1) + position_
        # # context_h_mean = context_h_mean.unsqueeze(1).repeat(1, decoder_output.size(1), 1)
        # score2 = self.copy_net_abstract(context_h_mean).masked_fill(mask.unsqueeze(1).bool(), -1000000.0)
        copy_latent = self.dim_align(self.user_representation.unsqueeze(1).repeat(1, decoder_output.size(1), 1))
        score2 = self.copy_net_kg(copy_latent).masked_fill(self.kg_mask.bool(), -1000000.0)
        # score3 = self.copy_net_kg(context_h_mean.unsqueeze(1).repeat(1, decoder_output.size(1), 1)).\
        #     masked_fill(self.kg_mask.bool(), -1000000.0)
        dict_len = score1.size(2)
        # score = F.log_softmax(torch.cat([score1, score2], dim=2), 2)
        # score = F.softmax(torch.cat([score1, score2, score3], dim=2), 2)
        score = F.softmax(torch.cat([score1, score2], dim=2), 2)
        score1 = score[:, :, :dict_len]
        score2 = score[:, :, dict_len:]
        # score1 = F.softmax(score1, dim=2)
        # score2 = F.softmax(score2, dim=2)
        score_total = score1 + score2
        score_logits = torch.log(score_total + 10**-10)
        # Only for debug
        sentence_score1 = score1.max(dim=2)[0].cpu().detach().numpy()
        sentence_score2 = score2.max(dim=2)[0].cpu().detach().numpy()
        sentence_score = score_total.max(dim=2)[0].cpu().detach().numpy()
        sentence_pred1 = score1.max(dim=2)[1].cpu().detach().numpy()
        sentence_pred2 = score2.max(dim=2)[1].cpu().detach().numpy()
        sentence_pred = score_total.max(dim=2)[1].cpu().detach().numpy()
        sentence_pred_w1 = []
        sentence_pred_w2 = []
        sentence_pred_w = []
        for sentence in score1.max(dim=2)[1].cpu().detach().numpy():
            temp = []
            for word in sentence:
                temp.append(self.dict.ind2tok[word])
            sentence_pred_w1.append(temp)
        for sentence in score2.max(dim=2)[1].cpu().detach().numpy():
            temp = []
            for word in sentence:
                temp.append(self.dict.ind2tok[word])
            sentence_pred_w2.append(temp)
        for sentence in score_total.max(dim=2)[1].cpu().detach().numpy():
            temp = []
            for word in sentence:
                temp.append(self.dict.ind2tok[word])
            sentence_pred_w.append(temp)
        mask_word = []
        for b in mask.cpu().detach().numpy():
            temp = []
            for i, v in enumerate(b):
                if v == 0:
                    temp.append(self.dict.ind2tok[i])
            mask_word.append(temp)

        return score_logits
        # return score_total

    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        if ys is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states, weights, context_h_mean = prev_enc if prev_enc is not None else self.encoder(*xs[:2])
        mask = xs[2]
        _, index = weights.max(dim=1, keepdim=True)
        index_1 = index.unsqueeze(-1).repeat(1, 1, mask.size(2))
        mask = torch.gather(mask, dim=1, index=index_1).squeeze(1)
        num = mask.nonzero()
        if ys is not None:
            # use teacher forcing
            scores, preds = self.decode_forced((encoder_states, mask, context_h_mean), ys)
        else:
            scores, preds = self.decode_greedy(
                (encoder_states, mask, context_h_mean),
                bsz,
                maxlen or self.longest_label,
            )

        return scores, preds, weights, encoder_states

    def decode_greedy(self, encoder_states, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        encoder_states, mask, context_h_mean = encoder_states
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)
            scores = scores[:, -1:, :]
            scores = self.output(scores, mask, context_h_mean)
            _, preds = scores.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        encoder_states, mask, context_h_mean = encoder_states
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent, mask, context_h_mean)
        _, preds = logits.max(dim=2)
        return logits, preds


class TransformerCopyNetModel(TorchGeneratorModel):
    """Model which takes context, memories, candidates and encodes them"""
    def __init__(self, opt, dictionary, kg_mask, dialogue_mask, kg_mapping):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.opt = opt
        self.kg_mask = kg_mask
        self.kg_mapping = kg_mapping
        self.dialogue_mask = dialogue_mask
        # set up embeddings
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.dict = dictionary
        if not opt.get('learn_embeddings'):
            self.embeddings.weight.requires_grad = False

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.context_encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )

        self.encoder = self.context_encoder
        self.decoder = _build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx, n_positions=n_positions
        )
        self.copy_net_dialogue = nn.Linear(opt['embedding_size'], len(dictionary))
        self.copy_net_kg = nn.Linear(opt['embedding_size'], len(dictionary))
        if opt['infuse_loss']:
            self.kg_voc = nn.Linear(opt['embedding_size'], len(dictionary))
        self.meta_path = opt["meta_path"]
        self.embedding_meta = opt["embedding_meta"]
        self.bag_of_entity = opt["bag_of_entity"]
        self.using_voc_embedding = opt['bag_of_entity_voc_embedding']
        self.dim_align = nn.Linear(opt['dim'], opt['embedding_size'])
        self.replace_movie = opt["replace_movie"]
        entity2entity_id = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "entity2entity_id4.pkl"), "rb")
        )
        self.id2entity = {}
        for e in entity2entity_id:
            self.id2entity[entity2entity_id[e]] = e
        kg = pkl.load(
            open(os.path.join(opt["datapath"], "crs", "movie_kg4.pkl"), "rb")
        )
        opt["n_entity"] = len(set(entity2entity_id.values()))
        opt["n_relation"] = 17
        opt["item_num"] = 6730
        self.recommend_model = KECRS(opt, opt["n_entity"], 16, opt['dim'], 0, kg, num_bases=10, return_all=True)
        state_dict = torch.load('kecrs/kecrs')['model']
        # state_dict = torch.load('../parlai/tasks/crs/kecrs/kecrs')['model']
        self.recommend_model.load_state_dict(state_dict)
        if self.bag_of_entity:
            if self.using_voc_embedding:
                self.bag_dim_align = nn.Linear(opt['embedding_size']+opt['dim'], opt['dim'])
            else:
                self.bag_dim_align = nn.Linear(opt['embedding_size'], opt['dim'])
            self.bag_output = nn.Linear(opt['dim'], opt["n_entity"])

    def output(self, decoder_output, test=False):
        if self.opt['model_type'] == 'copy_net':
            if self.opt['fake_copy']:
                score1 = F.linear(decoder_output, self.embeddings.weight)
                copy_latent = self.dim_align(self.user_representation.unsqueeze(1).repeat(1, decoder_output.size(1), 1))
                score2 = self.copy_net_kg(copy_latent) * self.kg_mask.unsqueeze(0).unsqueeze(0)
                if test:
                    ratio = 2
                    score2 = abs(score2)
                else:
                    ratio = 1
                score_logits = score1 + score2 * ratio
            else:
                score1 = F.linear(decoder_output, self.embeddings.weight).masked_fill(self.dialogue_mask.bool(), -1000000.0)
                copy_latent = self.dim_align(self.user_representation.unsqueeze(1).repeat(1, decoder_output.size(1), 1))
                score2 = self.copy_net_kg(copy_latent).masked_fill(self.kg_mask.bool(), -1000000.0)
                score1 = F.softmax(score1, dim=2)
                score2 = F.softmax(score2, dim=2)
                score_total = score1 + score2
                score_logits = torch.log(score_total + 10**-10)
        else:
            score_logits = F.linear(decoder_output, self.embeddings.weight)

        if self.bag_of_entity:
            if self.using_voc_embedding:
                concat_feature = torch.cat(
                    [self.user_representation.unsqueeze(1).repeat(1, decoder_output.size(1), 1), decoder_output],
                    dim=-1
                )
                aligned_features = self.bag_dim_align(concat_feature)

                entity_score = F.softmax(F.linear(aligned_features, self.nodes_feature, self.bag_output.bias), dim=-1)
            else:
                aligned_features = self.bag_dim_align(decoder_output)
                entity_score = F.softmax(F.linear(aligned_features, self.nodes_feature, self.bag_output.bias), dim=-1)
            return score_logits, entity_score
        else:
            return score_logits

    def forward(self, *xs, ys=None, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        if ys is not None:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)
        if ys is not None:
            # use teacher forcing
            if self.bag_of_entity:
                bag_scores, scores, preds = self.decode_forced(encoder_states, ys)
            else:
                scores, preds = self.decode_forced(encoder_states, ys)
        else:
            scores, preds = self.decode_greedy(
                encoder_states,
                bsz,
                maxlen or self.longest_label,
            )
            if self.bag_of_entity:
                bag_scores = None
        if self.bag_of_entity:
            return bag_scores, scores, preds, encoder_states
        else:
            return scores, preds, encoder_states

    def decode_greedy(self, encoder_states, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        test_mode = False
        if bsz == 1:
            test_mode = True
        if test_mode:
            _, top_k_movie = torch.topk(self.movie_scores, k=10, dim=1)
            top_k_movie = [self.dict.tok2ind[self.id2entity[i.item()]] for i in top_k_movie.squeeze(0)]
            top_k_counter = 0
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)
            scores = scores[:, -1:, :]
            if self.bag_of_entity:
                scores, bag_scores = self.output(scores, True)
                bag_scores_ = torch.index_select(bag_scores, -1, self.kg_mapping) * self.kg_mask
                scores = F.softmax(scores, dim=-1) + bag_scores_ * 0.1
            else:
                scores = self.output(scores, True)
            _, preds = scores.max(dim=-1)
            logits.append(scores)
            if test_mode and preds[0].item() == self.dict.tok2ind['@']:
                if top_k_counter >= len(top_k_movie):
                    print(1)
                    break
                preds = torch.cat([preds, torch.tensor([[top_k_movie[top_k_counter]]]).cuda()], dim=1)
                top_k_counter += 1
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        if self.bag_of_entity:
            logits, bag_scores = self.output(latent)
        else:
            logits = self.output(latent)
        _, preds = logits.max(dim=2)
        if self.bag_of_entity:
            return bag_scores, logits, preds
        else:
            return logits, preds
        # return loss_logits, preds


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


class TransformerResponseWrapper(nn.Module):
    """Transformer response rapper. Pushes input through transformer and MLP"""
    def __init__(self, transformer, hdim):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, dim)
        )

    def forward(self, *args):
        transformer_out = self.transformer(*args)
        if type(transformer_out) == tuple and len(transformer_out) == 2:
            return self.mlp(transformer_out[0]), transformer_out[1]
        else:
            return self.mlp(transformer_out)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions: Size of the position embeddings matrix.
    """
    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
        n_positions=1024
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            assert False
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        mask = input != self.padding_idx
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.type_as(tensor).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(
            n_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class BasicAttention(nn.Module):
    def __init__(self, dim=1, attn='cosine'):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim

    def forward(self, xs, ys):
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        l2 = self.softmax(l1)
        lhs_emb = torch.bmm(l2, ys)
        # add back the query
        lhs_emb = lhs_emb.add(xs)

        return lhs_emb.squeeze(self.dim - 1), l2


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights_ = attn_weights.detach().cpu().numpy()
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
