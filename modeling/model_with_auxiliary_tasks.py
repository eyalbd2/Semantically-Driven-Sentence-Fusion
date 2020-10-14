
import sys
import os
import csv
import collections
import importlib
import tensorflow as tf

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']
if not 'configs' in sys.path:
  sys.path += ['configs']
if not 'preprocesses' in sys.path:
  sys.path += ['preprocesses']

from config_with_auxiliary_tasks import *
from preprocess_with_auxiliary_tasks import file_based_input_fn_builder
import texar as tx
from examples.bert.utils import data_utils, model_utils, tokenization
from examples.bert import config_classifier as config_downstream
from texar.utils import transformer_utils
from examples.transformer.bleu_tool import bleu_wrapper
from examples.transformer.utils import data_utils, utils


train_dataset = file_based_input_fn_builder(
            input_file=train_out_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=True,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})

eval_dataset = file_based_input_fn_builder(
            input_file=eval_out_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt=max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})


test_dataset = file_based_input_fn_builder(
            input_file=test_out_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})


bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

vocab_size = len(tokenizer.vocab)

src_input_ids = tf.placeholder(tf.int64, shape=(None, None))
src_segment_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_input_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_segment_ids = tf.placeholder(tf.int64, shape=(None, None))


batch_size = tf.shape(src_input_ids)[0]

src_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                     axis=1)
tgt_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(tgt_input_ids, 0)),
                     axis=1)

labels = tf.placeholder(tf.int64, shape=(None, None))
type_label = tf.placeholder(tf.int64, shape=(None))
conn_label = tf.placeholder(tf.int64, shape=(None))
is_target = tf.to_float(tf.not_equal(labels, 0))

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

iterator = tx.data.FeedableDataIterator({'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})

batch = iterator.get_next()

# Encoder - Bert model
print("Intializing the Bert Encoder Graph")
with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(src_input_ids)

        # Creates segment embeddings for each type of tokens.
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(src_segment_ids)

        input_embeds = word_embeds + segment_embeds

        # The BERT model (a TransformerEncoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        # 'encoder_output' is used for a generation task
        encoder_output = encoder(input_embeds, src_input_length)
        # Builds layers for downstream classification, which is also initialized
        # with BERT pre-trained checkpoint.
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(encoder_output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            # 'output' is used for classification task
            output = tf.layers.dropout(bert_sent_output, rate=0.1, training=tx.global_mode_train())
            dense = tf.keras.layers.Dense(256, activation='relu')(output)
            type_logits = tf.keras.layers.Dense(13)(dense)
            conn_logits = tf.keras.layers.Dense(71)(dense)


print("loading the bert pretrained weights")
# Loads pretrained BERT model parameters
init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
model_utils.init_bert_checkpoint(init_checkpoint)

tgt_embedding = tf.concat(
    [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)

decoder = tx.modules.TransformerDecoder(embedding=tgt_embedding, hparams=dcoder_config)

# For training
outputs = decoder(  
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    inputs=embedder(tgt_input_ids),
    sequence_length=tgt_input_length,
    decoding_strategy='train_greedy',
    mode=tf.estimator.ModeKeys.TRAIN
)

# Type Classification Loss
type_probabilities = tf.nn.softmax(type_logits, axis=-1)
type_log_probs = tf.nn.log_softmax(type_logits, axis=-1)
type_one_hot_labels = tf.one_hot(type_label, depth=13, dtype=tf.float32)
type_per_example_loss = -tf.reduce_sum(type_one_hot_labels * type_log_probs, axis=-1)
type_cls_loss = tf.reduce_mean(type_per_example_loss)

# Connective string Classification Loss
conn_probabilities = tf.nn.softmax(conn_logits, axis=-1)
conn_log_probs = tf.nn.log_softmax(conn_logits, axis=-1)
conn_one_hot_labels = tf.one_hot(conn_label, depth=71, dtype=tf.float32)
conn_per_example_loss = -tf.reduce_sum(conn_one_hot_labels * conn_log_probs, axis=-1)
conn_cls_loss = tf.reduce_mean(conn_per_example_loss)

# Generation Loss
mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, loss_label_confidence)
mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

type_const_cls = 0.2

conn_const_cls = 0.2

total_loss = mle_loss + (type_const_cls*type_cls_loss) + (conn_const_cls*conn_cls_loss)

tvars = tf.trainable_variables()

non_bert_vars = [var for var in tvars if 'bert' not in var.name]
bert_vars = [var for var in tvars if 'bert' in var.name]
enc_vars = [var for var in bert_vars if "/encoder/" in var.name]
pooler_vars = [var for var in tvars if "/pooler/" in var.name]
last_layers_enc = [var for var in enc_vars if "/layer_11/" in var.name] + \
                  [var for var in enc_vars if "/layer_10/" in var.name]

if type_const_cls == 0 and conn_const_cls == 0:
    trainable = non_bert_vars + last_layers_enc
else:
    trainable = non_bert_vars + pooler_vars + last_layers_enc


train_op = tx.core.get_train_op(
        total_loss,
        learning_rate=learning_rate,
        variables=trainable,
        global_step=global_step,
        hparams=opt)

tf.summary.scalar('lr', learning_rate)
tf.summary.scalar('mle_loss', mle_loss)
tf.summary.scalar('type_cls_loss', type_cls_loss)
tf.summary.scalar('conn_cls_loss', conn_cls_loss)
tf.summary.scalar('total_loss', total_loss)
summary_merged = tf.summary.merge_all()

saver_load = tf.train.Saver([v for v in tf.all_variables() if 'OptimizeLoss' not in v.name])
saver = tf.train.Saver(max_to_keep=5)
best_results = {'score': 0, 'epoch': -1}

start_tokens = tf.fill([tx.utils.get_batch_size(src_input_ids)],
                       bos_token_id)
predictions = decoder(
    memory=encoder_output,   
    memory_sequence_length=src_input_length,
    decoding_strategy='infer_greedy',
    beam_width=beam_width,
    alpha=alpha,
    start_tokens=start_tokens,
    end_token=eos_token_id,
    max_decoding_length=60,
    mode=tf.estimator.ModeKeys.PREDICT
)
if beam_width <= 1:
    inferred_ids = predictions[0].sample_id
else:
    # Uses the best sample by beam search
    inferred_ids = predictions['sample_id'][:, :, 0]



