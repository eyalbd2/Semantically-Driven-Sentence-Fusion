# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAE config.
"""

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

dataset = "ptb"
num_epochs = 100
hidden_size = 256
dec_dropout_in = 0.
enc_dropout_in = 0.
enc_dropout_out = 0.
batch_size = 32
embed_dim = 256

latent_dims = 32

lr_decay_hparams = {
    "init_lr": 0.001,
    "threshold": 2,
    "decay_factor": 0.5,
    "max_decay": 5
}


relu_dropout = 0.2
embedding_dropout = 0.2
attention_dropout = 0.2
residual_dropout = 0.2
num_blocks = 3

decoder_hparams = {
    "type": "transformer"
}

enc_cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": 1. - enc_dropout_out},
    "num_layers": 1
}

enc_emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    "dropout_rate": enc_dropout_in,
    'initializer' : {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': embed_dim**-0.5,
        },
    }
}

dec_emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    "dropout_rate": dec_dropout_in,
    'initializer' : {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': embed_dim**-0.5,
        },
    }
}

# due to the residual connection, the embed_dim should be equal to hidden_size
trans_hparams = {
    'output_layer_bias': False,
    'embedding_dropout': embedding_dropout,
    'residual_dropout': residual_dropout,
    'num_blocks': num_blocks,
    'dim': hidden_size,
    'position_embedder_hparams': {
        'dim': hidden_size,
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'multihead_attention': {
        'dropout_rate': attention_dropout,
        'num_heads': 8,
        'num_units': hidden_size,
        'output_dim': hidden_size
    },
    'poswise_feedforward': {
        'name': 'fnn',
        'layers': [
            {
                'type': 'Dense',
                'kwargs': {
                    'name': 'conv1',
                    'units': hidden_size*4,
                    'activation': 'relu',
                    'use_bias': True,
                },
            },
            {
                'type': 'Dropout',
                'kwargs': {
                    'rate': relu_dropout,
                }
            },
            {
                'type': 'Dense',
                'kwargs': {
                    'name': 'conv2',
                    'units': hidden_size,
                    'use_bias': True,
                    }
            }
        ],
    }
}

# KL annealing
kl_anneal_hparams = {
    "warm_up": 10,
    "start": 0.1
}

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": './simple-examples/data/ptb.train.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": './simple-examples/data/ptb.valid.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": './simple-examples/data/ptb.test.txt',
        "vocab_file": './simple-examples/data/vocab.txt'
    }
}

opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
    }
}
