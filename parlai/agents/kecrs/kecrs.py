import copy
import os
import pickle as pkl
import re
import random
from collections import defaultdict
import math
import numpy as np
import torch

from parlai.core.torch_agent import Output, TorchAgent, Batch
from parlai.core.utils import round_sigfigs

from .modules import KECRS


class KECRSAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(KECRSAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("Arguments")
        agent.add_argument("-dim", "--dim", type=int, default=128)
        agent.add_argument("-n_relation", "--n_relation", type=int, default=16)
        agent.add_argument("-n_entity", "--n_entity", type=int, default=0)
        agent.add_argument("-kgew", "--kge-weight", type=float, default=1)
        agent.add_argument("-l2w", "--l2-weight", type=float, default=2.5e-5)
        agent.add_argument("-nmem", "--n-memory", type=int, default=32)
        agent.add_argument(
            "-ium", "--item-update-mode", type=str, default="plus_transform"
        )
        agent.add_argument("-uah", "--using-all-hops", type=bool, default=True)
        agent.add_argument(
            "-lr", "--learningrate", type=float, default=3e-3, help="learning rate"
        )
        agent.add_argument("-nb", "--num-bases", type=int, default=10)
        KECRSAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        init_model, is_finetune = self._get_init_model(opt, shared)

        self.id = "KECRSAgent"

        self.task = opt.get("task", "crs")

        if not shared:
            # set up model from scratch
            if self.task == "crs":
                self.kg = pkl.load(
                    open(os.path.join(opt["datapath"], "crs", "movie_kg4.pkl"), "rb")
                )
                entity2entityId = pkl.load(
                    open(os.path.join(opt["datapath"], "crs", "entity2entity_id4.pkl"), "rb")
                )

                opt["n_entity"] = len(set(entity2entityId.values()))
                # print(opt["n_entity"])
                opt["n_relation"] = 16
                opt["item_num"] = 6730
            else:
                assert self.task == "duRecDial"
                self.kg = pkl.load(
                    open(os.path.join(opt["datapath"], "DuRecDial_new", "kg.pkl"), "rb")
                )
                entity2entityId = pkl.load(
                    open(os.path.join(opt["datapath"], "DuRecDial_new", "entity2id.pkl"), "rb")
                )
                relation2id = pkl.load(
                    open(os.path.join(opt["datapath"], "DuRecDial_new", "relation2id.pkl"), "rb")
                )
                item2id = pkl.load(
                    open(os.path.join(opt["datapath"], "DuRecDial_new", "item2id.pkl"), "rb")
                )

                opt["n_entity"] = len(set(entity2entityId.values()))
                opt["n_relation"] = len(set(relation2id.values()))
                opt["item_num"] = len(set(item2id.values()))
            # encoder captures the input text
            self.model = KECRS(
                opt,
                n_entity=opt["n_entity"],
                n_relation=opt["n_relation"],
                dim=opt["dim"],
                n_hop=1,
                kg=self.kg,
                num_bases=opt["num_bases"],
            )
            if init_model is not None:
                # load model parameters if available
                print("[ Loading existing model params from {} ]" "".format(init_model))
                states = self.load(init_model)
                if "number_training_updates" in states:
                    self._number_training_updates = states["number_training_updates"]

            if self.use_cuda:
                self.model.cuda()
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                opt["learningrate"],
                weight_decay=opt["l2_weight"]
            )

        elif "kbrd" in shared:
            # copy initialized data from shared table
            self.model = shared["kbrd"]
            self.kg = shared["kg"]
            self.optimizer = shared["optimizer"]

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        m["num_tokens"] = self.counts["num_tokens"]
        m["num_batches"] = self.counts["num_batches"]
        if m["num_batches"] == 0:
            m["loss"] = self.metrics["loss"]
            m["base_loss"] = self.metrics["base_loss"]
        else:
            m["loss"] = self.metrics["loss"] / m["num_batches"]
            m["base_loss"] = self.metrics["base_loss"] / m["num_batches"]
        if m["num_tokens"] == 0:
            m["acc"] = self.metrics["acc"]
            m["auc"] = self.metrics["auc"]
        else:
            m["acc"] = self.metrics["acc"] / m["num_tokens"]
            m["auc"] = self.metrics["auc"] / m["num_tokens"]
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall") and self.counts[x] > 200:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
            if x.startswith("precision") and self.counts[x] > 0:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
            if x.startswith("ndcg") and self.counts[x] > 0:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["kbrd"] = self.model
        shared["kg"] = self.kg
        shared["optimizer"] = self.optimizer
        return shared

    def vectorize(self, obs, history, **kwargs):
        if "text" not in obs:
            return obs

        if "labels" in obs:
            label_type = "labels"
        elif "eval_labels" in obs:
            label_type = "eval_labels"
        else:
            label_type = None
        if label_type is None:
            return obs

        # mentioned movies
        input_match = list(map(int, obs['label_candidates'][1].split()))
        labels_match = list(map(int, obs['label_candidates'][2].split()))
        entities_match = list(map(int, obs['label_candidates'][3].split()))

        if not labels_match:
            del obs["text"], obs[label_type]
            return obs

        input_vec = torch.zeros(self.model.n_entity)
        labels_vec = torch.zeros(self.model.n_entity, dtype=torch.long)
        input_vec[input_match] = 1
        input_vec[entities_match] = 1
        labels_vec[labels_match] = 1
        obs["text_vec"] = input_vec
        obs[label_type + "_vec"] = labels_vec

        # turn no.
        obs["turn"] = len(input_match)
        return obs

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]
        if len(valid_obs) == 0:
            return Batch()
        valid_inds, exs = zip(*valid_obs)
        # MOVIE ENTITIES
        turn = None
        if any('turn' in ex for ex in exs):
            turn = [ex.get('turn', []) for ex in exs]
        batch.turn = turn
        return batch

    def train_step(self, batch):
        self.model.train()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)
        seed_sets = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            labels[i] = movieIdx
            seed_sets.append(seed_set)
        if self.use_cuda:
            labels = labels.cuda()
        assert len(seed_sets) == len(labels)
        return_dict = self.model(seed_sets, labels)
        outputs = return_dict["scores"].cpu()
        pre_score, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(bs):
            target_idx = labels[b].item()
            pred_list = pred_idx[b][:50].tolist()
            self.metrics["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.counts[f"recall@1"] += 1
            self.counts[f"recall@10"] += 1
            self.counts[f"recall@50"] += 1
        # L1_reg = 0
        # for param in self.model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        loss = return_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()

        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1
        self._number_training_updates += 1

    def eval_step(self, batch):
        if batch.text_vec is None:
            return
        self.model.eval()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)
        text = []

        # create subgraph for propagation
        seed_sets = []
        turns = []
        positive_set_list = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            labels[i] = movieIdx
            # labels[i] = self.movie_ids.index(movieIdx)
            positive_set = batch.label_vec[b].nonzero().squeeze(dim=1).tolist()
            # positive_set = [self.movie_ids.index(pos_item) for pos_item in positive_set]
            positive_set_list.append(set(positive_set))
            seed_sets.append(seed_set)
            turns.append(batch.turn[b])

        if self.use_cuda:
            labels = labels.cuda()

        assert len(seed_sets) == len(labels)
        return_dict = self.model(seed_sets, labels)

        loss = return_dict["loss"]

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()
        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1

        outputs = return_dict["scores"].cpu()
        # outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        pred_score, pred_idx = torch.topk(outputs, k=100, dim=1)
        counter = 0
        for b in range(bs):
            target_idx = labels[b].item()
            pred_list = pred_idx[b][:50].tolist()
            temp_seed_set = []
            for seed in seed_sets[b]:
                try:
                    temp_seed_set.append(seed)
                    # temp_seed_set.append(self.movie_ids.index(seed))
                except ValueError:
                    pass
            self.metrics["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics["recall@5"] += int(target_idx in pred_idx[b][:5].tolist())
            self.metrics["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.counts[f"recall@1"] += 1
            self.counts[f"recall@5"] += 1
            self.counts[f"recall@10"] += 1
            self.counts[f"recall@50"] += 1
            # NDCG
            if target_idx in pred_idx[b][:5].tolist():
                self.metrics["ndcg@5"] += 1 / math.log(pred_idx[b][:5].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@5"] += 1
            if target_idx in pred_idx[b][:10].tolist():
                self.metrics["ndcg@10"] += 1 / math.log(pred_idx[b][:10].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@10"] += 1
            if target_idx in pred_idx[b][:20].tolist():
                self.metrics["ndcg@20"] += 1 / math.log(pred_idx[b][:20].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@20"] += 1
            if target_idx in pred_idx[b][:50].tolist():
                self.metrics["ndcg@50"] += 1 / math.log(pred_idx[b][:50].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@50"] += 1
            if counter == 0:
                target_id_set = positive_set_list[b]
                counter = len(target_id_set) - 1
                # self.metrics["recall@1_real"] += len(target_id_set & set(pred_idx[b][:1].tolist())) \
                #                                  / len(target_id_set)
                # self.metrics["recall@10_real"] += len(target_id_set & set(pred_idx[b][:10].tolist())) \
                #                                   / len(target_id_set)
                # self.metrics["recall@50_real"] += len(target_id_set & set(pred_idx[b][:50].tolist())) \
                #                                   / len(target_id_set)
                # self.counts[f"recall@1_real"] += 1
                # self.counts[f"recall@10_real"] += 1
                # self.counts[f"recall@50_real"] += 1
                self.metrics["precision@1"] += len(target_id_set & set(pred_idx[b][:1].tolist()))
                self.metrics["precision@3"] += len(target_id_set & set(pred_idx[b][:3].tolist())) / 3
                self.metrics["precision@5"] += len(target_id_set & set(pred_idx[b][:5].tolist())) / 5
                self.metrics["precision@10"] += len(target_id_set & set(pred_idx[b][:10].tolist())) / 10
                self.metrics["precision@50"] += len(target_id_set & set(pred_idx[b][:50].tolist())) / 50
                self.counts[f"precision@1"] += 1
                self.counts[f"precision@3"] += 1
                self.counts[f"precision@5"] += 1
                self.counts[f"precision@10"] += 1
                self.counts[f"precision@50"] += 1
            else:
                counter -= 1
