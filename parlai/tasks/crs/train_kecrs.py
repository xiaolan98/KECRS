#!/data/qibin/anaconda3/envs/alchemy/bin/python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""
import sys
sys.path.append("../../../")
import json
from parlai.scripts.train_model import TrainLoop, setup_args, _maybe_load_eval_world, run_eval
from parlai.core.agents import create_agent
from collections import defaultdict


def train_once(base_path_):
    parser = setup_args()
    parser.set_defaults(
        task="crs",
        # datatype='test',
        dict_tokenizer="split",
        model="kecrs",
        dict_file=base_path_+"/tmp",
        model_file=base_path_+"/kecrs",
        fp16=True,
        batchsize=64,
        n_entity=0,
        n_relation=0,
        dim=200,
        validation_metric="recall@50",
        validation_metric_mode='max',
        validation_every_n_epochs=0.2,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        tensorboard_metrics="loss,base_loss,kge_loss,l2_loss,"
                            "acc,auc,recall@1,recall@10,recall@50,recall@1@same,recall@10@same,recall@50@same",
        sub_graph=True,
        hop=3,
        L=1,
        vocab_dim=100
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()


def average_results(base_path_, times_):
    output_scores = defaultdict(float)
    for i_ in range(times_):
        test_results_file_path = base_path_ + str(i_) + "/kecrs.test"
        test_results = open(test_results_file_path, 'r').readline().strip()[5:].replace("\'", "\"")
        test_results_dict = json.loads(test_results)
        for key in test_results_dict:
            output_scores[key] += test_results_dict[key]
    for key in output_scores:
        output_scores[key] = round(output_scores[key] / 3, 5)
    print(json.dumps(output_scores))


def test_model(base_path_):
    parser = setup_args()
    parser.set_defaults(
        task="crs",
        datatype='test',
        dict_tokenizer="split",
        model="kecrs",
        dict_file=base_path_ + "/tmp",
        model_file=base_path_ + "/kecrs",
        fp16=True,
        batchsize=64,
        n_entity=0,
        n_relation=0,
        dim=200,
        validation_metric="recall@50",
        validation_metric_mode='max',
        validation_every_n_epochs=0.2,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        sub_graph=True,
        hop=3,
        L=1,
        vocab_dim=100
    )
    opt = parser.parse_args()
    agent = create_agent(opt)
    test_world = _maybe_load_eval_world(agent, opt, 'test')
    t_report = run_eval(test_world, opt, 'test', -1, write_log=True)
    print(1)


if __name__ == "__main__":
    # times = 1
    # base_path = "kecrs_"
    # for i in range(times):
    #     train_once(base_path + str(i))
    # print(base_path)
    # average_results(base_path, times)
    train_once("kecrs")
    test_model("kecrs")
