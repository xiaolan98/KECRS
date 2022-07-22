#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train a model using parlai's standard training loop.

For documentation, see parlai.scripts.train_model.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("../../../")
from parlai.scripts.train_model import TrainLoop, setup_args, _maybe_load_eval_world, run_eval
from parlai.core.agents import create_agent

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='crs',
        model='transformer_rec_new/generator',
        # datatype="test",
        model_file=
        'kecrs/transformer_rec',
        dict_file='kecrs/kecrs.dict',
        output_suffix='kecrs.txt',
        bag_of_entity_positive_only=True,
        bag_of_entity_voc_embedding=True,
        model_type='transformer',
        fake_copy=True,
        infuse_loss=True,
        bag_of_entity=True,
        delimiter="__split__",
        max_length=20,
        dict_tokenizer='nltk',
        dict_lower=True,
        batchsize=32,
        dim=200,
        embedding_size=300,
        truncate=256,
        dropout=0.1,
        relu_dropout=0.1,
        validation_metric='dist4',
        validation_metric_mode='max',
        validation_every_n_epochs=3,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,ffn_size,embedding_size,n_layers,learningrate,model_file",
        tensorboard_metrics="loss,ppl,nll_loss,token_acc,bleu, kg_voc_loss, accuracy, "
                            "dist1, dist2, dist3, dist4, dist5, bag_of_entity_loss",
        dict_minfreq=3,
        # gpu=1,
    )
    opt = parser.parse_args()
    TrainLoop(opt).train()
    # agent = create_agent(opt)
    # test_world = _maybe_load_eval_world(agent, opt, 'test')
    # t_report = run_eval(test_world, opt, 'test', -1, write_log=True)
