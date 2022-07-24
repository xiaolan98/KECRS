import time
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
import warnings

warnings.filterwarnings('ignore')


def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        N = h.shape[0]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, h)


def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            edge_list.append((entity, entity, 391))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 391:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)


class KECRS(nn.Module):
    def __init__(
            self,
            opt,
            n_entity,
            n_relation,
            dim,
            n_hop,
            kg,
            num_bases=None,
            return_all=False
    ):
        super(KECRS, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        # self.dim = 64
        self.n_hop = n_hop
        self.genre_num = 19
        self.cast_num = 5295
        self.movie_entity_num = opt["item_num"]
        self.task = opt["task"]

        # self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        # nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        # -------- baseline setting ---------------------------
        self.criterion = nn.CrossEntropyLoss()
        self.kg_criterion = nn.MSELoss()
        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        self.kg = kg
        self.return_all = return_all
        # print(self.n_relation)
        # self.edge_idx, self.edge_type, self.self_loop, self.n_relation = self.flatten_graph(self.kg, True)
        # # self.edge_idx, self.edge_type = self.flatten_graph(self.kg)
        # print(self.n_relation)
        # self.rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation, num_bases=num_bases)
        # -------- baseline setting end -----------------------

        # -------- different loss setting ---------------------
        # self.output = nn.Linear(self.dim, n_entity - self.genre_num - self.cast_num)
        self.output = nn.Linear(self.dim, self.movie_entity_num)
        # self.output1 = nn.Linear(self.dim, len(movie_ids))
        # self.output2 = nn.Linear(self.dim, len(movie_ids))
        # self.pair_criterion = nn.CosineEmbeddingLoss(margin=0.5)
        # -------- different loss setting end -----------------

        # -------- Movie kg RGCN setting ----------------------
        self.edge_idx, self.edge_type = self.flatten_graph(self.kg, False, False)
        print(self.edge_idx.shape, self.edge_type.shape)
        # 2 RGCN layers
        # self.node_feature_kg = nn.Parameter(torch.randn((n_entity, self.dim)), requires_grad=True)
        # self.kg_rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation ,num_bases=num_bases)
        # self.kg_rgcn2 = RGCNConv(self.dim, self.dim, self.n_relation, num_bases=num_bases)
        # self.kg_node_fea_transformation = nn.Linear(self.dim * 2, self.dim)
        # 1 RGCN layers
        self.kg_rgcn = RGCNConv(self.n_entity, self.dim, self.n_relation, num_bases=num_bases)

    def forward(
            self,
            seed_sets: list,
            labels: torch.LongTensor,
    ):
        u_emb, nodes_features = self.kg_movie_score(seed_sets)

        scores = F.linear(u_emb, nodes_features, self.output.bias)
        base_loss = self.criterion(scores, labels)
        loss = base_loss

        return dict(scores=scores.detach(), base_loss=base_loss, loss=loss)

    def kg_movie_score(self, seed_sets):
        # 2 RGCN layers --> High GPU memory consuming
        # nodes_features1 = self.kg_rgcn(self.node_feature_kg, self.edge_idx, self.edge_type)
        # nodes_features2 = self.kg_rgcn2(nodes_features1, self.edge_idx, self.edge_type)
        # nodes_features = torch.cat([nodes_features1, nodes_features2], dim=1)
        # nodes_features = self.kg_node_fea_transformation(nodes_features)
        # 1 RGCN layers --> comparable performance
        nodes_features = self.kg_rgcn(None, self.edge_idx, self.edge_type)
        # ----------- self-attention ------------------------------------
        user_representation_list = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                continue
            # seed_set = [self.movie_ids_.index(seed)for seed in seed_set]
            user_representation = nodes_features[seed_set]
            user_representation = self.self_attn(user_representation)
            user_representation_list.append(user_representation)
        if self.return_all:
            return torch.stack(user_representation_list), nodes_features
        else:
            return torch.stack(user_representation_list), nodes_features[:self.movie_entity_num]

    def flatten_graph(self, kg, return_relation=False, kg_modified=True):
        edge_list = []
        kg_m = defaultdict(list)
        for entity in kg:
            for relation_and_tail in kg[entity]:
                relation = relation_and_tail[0]
                if relation == 16 and self.task == "crs":
                    continue
                tail = relation_and_tail[1]
                edge_list.append((entity, tail, relation))
                edge_list.append((tail, entity, relation))
        relation_cnt = defaultdict(int)
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        if return_relation:
            relation_idx = {}
            for h, t, r in edge_list:
                if relation_cnt[r] > 1000 and r not in relation_idx:
                    relation_idx[r] = len(relation_idx)
            edge_list = [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000]
        if kg_modified:
            for (h, t, r) in edge_list:
                kg_m[h].append((r, t))
            self.kg = kg_m
        edge_list = list(set(edge_list))
        edge_list_tensor = torch.LongTensor(edge_list).cuda()
        edge_idx = edge_list_tensor[:, :2].t()
        edge_type = edge_list_tensor[:, 2]
        if return_relation:
            return edge_idx, edge_type, len(relation_idx)
        else:
            return edge_idx, edge_type


