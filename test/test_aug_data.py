import sys
import tensorflow as tf
import pickle
import numpy as np
from tensor2tensor.utils import sari_hook
import collections
import os

if not 'texar_repo' in sys.path:
    sys.path += ['texar_repo']
if not 'configs' in sys.path:
    sys.path += ['configs']
if not 'modeling' in sys.path:
    sys.path += ['modeling']

import texar as tx
from config_aug_data import *
from model_aug_data import *
from examples.transformer.bleu_tool import bleu_tokenize


def exact_wrapper(ref_filename, hyp_filename):
    """Compute EXACT for two files (reference and hypothesis translation)."""
    ref_lines = open(ref_filename, encoding='utf-8').read().splitlines()
    hyp_lines = open(hyp_filename, encoding='utf-8').read().splitlines()
    assert len(ref_lines) == len(hyp_lines)
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    correct_cnt = 0
    for i in range(len(ref_tokens)):
        if hyp_tokens[i] == ref_tokens[i]:
            correct_cnt += 1
    exact = correct_cnt / len(ref_tokens)
    return exact


def _test(sess):
    references, hypotheses = [], []
    fetches = {
        'inferred_ids': inferred_ids,
        'type_log_probs': type_log_probs,
        'conn_log_probs': conn_log_probs,
    }
    correct_conns = 0
    correct_types = 0
    total_conns = 0
    total_types = 0
    pred_conns = []
    gt_conns = []
    pred_types = []
    gt_types = []

    num_batches_in_epoch = num_test_batches_in_epoch

    for test_batch in range(num_batches_in_epoch):

        try:
            if test_batch % 50 == 0:
                print("Batch", test_batch + 1, "out of", num_batches_in_epoch, "test batches")

            feed_dict = {
                iterator.handle: iterator.get_handle(sess, 'test'),
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            op = sess.run([batch], feed_dict)
            feed_dict = {
                src_input_ids: op[0]['src_input_ids'],
                src_segment_ids: op[0]['src_segment_ids'],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
            labels = op[0]['tgt_labels']
            type_labels = op[0]['type_label']
            conn_labels = op[0]['conn_label']
            src_sentences = op[0]['src_input_ids']

            hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
            references.extend(r.tolist() for r in labels)
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

            cur_type_log_probs = fetches_['type_log_probs']
            cur_type_preds = np.argmax(cur_type_log_probs, axis=1)
            cur_conn_log_probs = fetches_['conn_log_probs']
            cur_conn_preds = np.argmax(cur_conn_log_probs, axis=1)
            for i in range(len(cur_type_preds)):
                pred_types.append(cur_type_preds[i])
                gt_types.append(type_labels[i])
                total_types += 1
                if cur_type_preds[i] == type_labels[i]:
                    correct_types += 1
            for i in range(len(cur_conn_preds)):
                pred_conns.append(cur_conn_preds[i])
                gt_conns.append(conn_labels[i])
                total_conns += 1
                if cur_conn_preds[i] == conn_labels[i]:
                    correct_conns += 1
        except tf.errors.OutOfRangeError:
            break

    # Writes results to files to evaluate EXACT, MR-EXACT, SARI, and MR-SARI
    fname = os.path.join(model_dir, 'tmp.test')
    gt_type_fname = os.path.join(model_dir, 'gt-type-test.pkl')
    pred_type_fname = os.path.join(model_dir, 'pred-type-test.pkl')
    gt_conns_fname = os.path.join(model_dir, 'gt-conns-test.pkl')
    pred_conns_fname = os.path.join(model_dir, 'pred-conns-test.pkl')

    with open(pred_type_fname, 'wb') as f:
        pickle.dump(pred_types, f)
    with open(gt_type_fname, 'wb') as f:
        pickle.dump(gt_types, f)
    with open(pred_conns_fname, 'wb') as f:
        pickle.dump(pred_conns, f)
    with open(gt_conns_fname, 'wb') as f:
        pickle.dump(gt_conns, f)

    hypotheses = tx.utils.str_join(hypotheses)
    references = tx.utils.str_join(references)
    hyp_fn, ref_fn = tx.utils.write_paired_text(
        hypotheses, references, fname, mode='s')
    eval_exact = exact_wrapper(ref_fn, hyp_fn)
    eval_exact = 100. * eval_exact
    eval_types_acc = 100 * (correct_types / total_types)
    eval_conns_acc = 100 * (correct_conns / total_conns)
    print('EXACT score %.4f, type accuracy %.2f, conns accuracy %.2f' % (eval_exact, eval_types_acc, eval_conns_acc))


tx.utils.maybe_create_dir(model_dir)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    print('Begin running with test mode')

    if tf.train.latest_checkpoint(model_dir) is not None:
        print('Restore latest checkpoint in %s' % model_dir)
        # saver_load.restore(sess, tf.train.latest_checkpoint(model_dir))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    iterator.initialize_dataset(sess)
    iterator.restart_dataset(sess, 'test')
    _test(sess)
