
import pandas as pd
import numpy as np
import pickle
import os
import texar as tx


dcoder_config = {
    'dim': 768,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 768
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}

loss_label_confidence = 0.9
random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768


opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}


lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 10000,
}

bos_token_id = 101
eos_token_id = 102

in_domain = "wiki"
data_dir = "data_frames/"
inner_data_dir = "gen+type+conn_large/"
run_mode = "train_and_evaluate"
model_dir = "./models/AugAuxBert/" + in_domain + "/"


batch_size = 100
max_train_steps = 10
display_steps = 100
checkpoint_steps = 1000
eval_steps = 100
max_decoding_length = 60
max_seq_length_src = 60
max_seq_length_tgt = 60
total_epochs = 10

is_distributed = False


train_out_file = "data/" + inner_data_dir + in_domain + "/train.tf_record"
eval_out_file = "data/" + inner_data_dir + in_domain + "/eval.tf_record"
test_out_file = "data/" + inner_data_dir + in_domain + "/test.tf_record"
test_cross_out_file = "data/" + inner_data_dir + in_domain + "/test_cross.tf_record"

bert_pretrain_dir = "./uncased_L-12_H-768_A-12"

con_str2idx_dict_path = "data/dictionaries/con_str2idx.pickle"
type2idx_dict_path = "data/dictionaries/type2idx.pickle"


def get_train_data_size(df_path):
    pickle_in = open(df_path, "rb")
    df = pickle.load(pickle_in)
    discourse_type = df["discourse_type"].values
    return len(discourse_type)


def get_data_size(df_path):
    try:
        pickle_in = open(df_path, "rb")
        df = pickle.load(pickle_in)
        try:
            discourse_type = df["discourse_type"].values
        except:
            discourse_type = df["discourse_type"]
    except:
        df = pd.read_csv(df_path, sep='\t')
        discourse_type = df.discourse_type.values
    return len(discourse_type)


train_df_path = data_dir + in_domain + "/Balanced-large-data/train.pickle"
train_size = get_data_size(train_df_path)
num_train_batches_in_epoch = int(np.ceil(train_size/batch_size))

dev_df_path = data_dir + in_domain + "/Balanced/dev.pickle"
dev_size = get_data_size(dev_df_path)
num_eval_batches_in_epoch = int(np.ceil(dev_size/batch_size))

test_df_path = data_dir + in_domain + "/Balanced/test.pickle"
test_size = get_data_size(test_df_path)
num_test_batches_in_epoch = int(np.ceil(test_size/batch_size))

