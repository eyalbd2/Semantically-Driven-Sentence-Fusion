
import os
import sys
import tensorflow as tf
import numpy as np
from tensor2tensor.utils import sari_hook

if not 'texar_repo' in sys.path:
    sys.path += ['texar_repo']
if not 'configs' in sys.path:
    sys.path += ['configs']
if not 'modeling' in sys.path:
    sys.path += ['modeling']

import texar as tx
from config_with_auxiliary_tasks import *
from model_with_auxiliary_tasks import *
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
    exact = correct_cnt/len(ref_tokens)
    return exact


def _train_epoch(sess, epoch, step, smry_writer):

    fetches = {
        'step': global_step,
        'train_op': train_op,
        'smry': summary_merged,
        'mle_loss': mle_loss,
        'type_cls_loss': type_cls_loss,
        'conn_cls_loss': conn_cls_loss,
        'total_loss': total_loss,
    }

    print("------ Epoch number", epoch + 1, "out of", total_epochs, "epochs ------")
    for train_batch in range(num_train_batches_in_epoch):
        try:
            feed_dict = {
                iterator.handle: iterator.get_handle(sess, 'train'),
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }
            op = sess.run([batch], feed_dict)

            feed_dict = {
                src_input_ids: op[0]['src_input_ids'],
                src_segment_ids: op[0]['src_segment_ids'],
                tgt_input_ids: op[0]['tgt_input_ids'],
                labels: op[0]['tgt_labels'],
                type_label: op[0]['type_label'],
                conn_label: op[0]['conn_label'],
                learning_rate: utils.get_lr(step, lr),
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)
            step, m_loss, t_loss, c_loss = fetches_['step'], fetches_['mle_loss'], fetches_['type_cls_loss'], fetches_['conn_cls_loss']

            if step and step % display_steps == 0:
                logger.info('batch: %d/%d, mle_loss: %.4f, type_cls_loss: %.4f, conn_cls_loss: %.4f', train_batch,
                            num_train_batches_in_epoch, m_loss, t_loss, c_loss)
                print('batch: %d/%d, mle_loss: %.4f, type_cls_loss: %.4f, conn_cls_loss: %.4f' % (train_batch+1,
                                                                                          num_train_batches_in_epoch,
                                                                                          m_loss, t_loss, c_loss))
                smry_writer.add_summary(fetches_['smry'], global_step=step)

        except tf.errors.OutOfRangeError:
            break

    model_path = model_dir + "/model_" + str(step) + ".ckpt"
    logger.info('saving model to %s', model_path)
    print('saving model to %s' % model_path)
    saver.save(sess, model_path)
    print("---EVAL---")
    _eval_epoch(sess, epoch, mode='eval')

    return step


def _eval_epoch(sess, epoch, mode):
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
    for eval_batch in range(num_eval_batches_in_epoch):

        try:
            print("Batch", eval_batch + 1, "out of", num_eval_batches_in_epoch, "eval batches")
            feed_dict = {
                iterator.handle: iterator.get_handle(sess, 'eval'),
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
            # references, hypotheses, source_ids = [], [], []
            # source_ids.extend(h.tolist()[1:] for h in feed_dict[src_input_ids])
            hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
            references.extend(r.tolist() for r in labels)
            # source_ids = utils.list_strip_eos(source_ids, eos_token_id)
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)
            cur_type_log_probs = fetches_['type_log_probs']
            cur_type_preds = np.argmax(cur_type_log_probs, axis=1)
            cur_conn_log_probs = fetches_['conn_log_probs']
            cur_conn_preds = np.argmax(cur_conn_log_probs, axis=1)
            for i in range(len(cur_type_preds)):
                total_types += 1
                if cur_type_preds[i] == type_labels[i]:
                    correct_types += 1
            for i in range(len(cur_conn_preds)):
                total_conns += 1
                if cur_conn_preds[i] == conn_labels[i]:
                    correct_conns += 1

        except tf.errors.OutOfRangeError:
            break

    if mode == 'eval':
        # Writes results to files to evaluate BLEU
        # For 'eval' mode, the BLEU is based on token ids (rather than
        # text tokens) and serves only as a surrogate metric to monitor
        # the training process
        fname = os.path.join(model_dir, 'tmp.eval')

        hypotheses = tx.utils.str_join(hypotheses)
        references = tx.utils.str_join(references)
        hyp_fn, ref_fn = tx.utils.write_paired_text(
            hypotheses, references, fname, mode='s')
        eval_exact = exact_wrapper(ref_fn, hyp_fn)
        eval_exact = 100. * eval_exact
        eval_types_acc = 100*(correct_types/total_types)
        eval_conns_acc = 100*(correct_conns/total_conns)
        logger.info('epoch: %d, exact score %.4f, type accuracy %.2f, conn accuracy %.2f', epoch + 1, eval_exact, eval_types_acc, eval_conns_acc)
        print('epoch: %d, exact score %.4f, type accuracy %.2f, conn accuracy %.2f' % (epoch + 1, eval_exact, eval_types_acc, eval_conns_acc))


        if eval_exact > best_results['score']:
            logger.info('epoch: %d, best exact: %.4f', epoch + 1, eval_exact)
            best_results['score'] = eval_exact
            best_results['epoch'] = epoch
            model_path = os.path.join(model_dir, 'best-model_type_conn.ckpt')

            logger.info('saving model to %s', model_path)
            print('saving model to %s' % model_path)
            saver.save(sess, model_path)


restore_dir = model_dir
tx.utils.maybe_create_dir(model_dir)
logging_file = os.path.join(model_dir, "logging.txt")
logger = utils.get_logger(logging_file)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    smry_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    if run_mode == 'train_and_evaluate':
        logger.info('Begin running with train_and_evaluate mode')

        if tf.train.latest_checkpoint(restore_dir) is not None:
            logger.info('Restore latest checkpoint in %s' % restore_dir)
            saver_load.restore(sess, tf.train.latest_checkpoint(restore_dir))
        else:
            logger.info('Didnt find checkpoint to restore from %s' % restore_dir)

        iterator.initialize_dataset(sess)

        step = 0
        for epoch in range(total_epochs):
            iterator.restart_dataset(sess, 'train')
            step = _train_epoch(sess, epoch, step, smry_writer)

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))
