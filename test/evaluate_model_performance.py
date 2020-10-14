
import sys
import os
import pandas as pd
import argparse

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']
if not 'sari' in sys.path:
  sys.path += ['sari']
if not 'preprocesses' in sys.path:
  sys.path += ['preprocesses']
if not 'modeling' in sys.path:
  sys.path += ['modeling']

import sari_utils
from examples.bert.utils import tokenization
from examples.transformer.bleu_tool import bleu_tokenize
import texar as tx


def load_data_from_file(fpath_trg_class):
    df = pd.read_pickle(fpath_trg_class)
    return df


def process_example_for_comparison(utterance):
    for char in ['-', '.', ',', "'", "$", "&", "#", "/", "+", "_", ":", "@"]:
        utterance = utterance.replace(" {} ".format(char), "{}".format(char)).\
            replace("{} ".format(char), "{}".format(char)).replace(" {}".format(char), "{}".format(char)).\
            replace("{}".format(char), " {} ".format(char))
    utterance = utterance.replace(" \\\\ ", " \ \ ")

    return utterance


def get_test_examples(df_path, tokenizer):
    data_dict = load_data_from_file(df_path)
    text_src = data_dict['defused_sent']
    exact_text_trg = data_dict['exact_fused_sent']
    many_text_trg = data_dict['many_fused_sents']

    src_text = []
    src_lines = []
    for src in text_src:
        src_line = tokenization.convert_to_unicode(src)
        tokens = tokenizer.tokenize(src_line)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_text.append(ids)
        src_lines.append(src_line)

    trg_text = []
    trg_lines = []

    for tgt in many_text_trg:
        cur_trg_text = []
        cur_trg_lines = []
        for cur_tgt in tgt:
            tgt_line = tokenization.convert_to_unicode(cur_tgt)
            tokens = tokenizer.tokenize(tgt_line)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            cur_trg_text.append(ids)
            cur_trg_lines.append(tgt_line)
        trg_text.append(cur_trg_text)
        trg_lines.append(cur_trg_lines)

    return src_text, src_lines, trg_text, trg_lines


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_file_path",
        default='models/AugAuxBert/wiki/',
        type=str,
        required=False,
        help="Path to model the is evaluated.",
    )
    parser.add_argument(
        "--mr_pkl_file_path",
        default='data_frames/wiki/Balanced-multi-ref/test.pickle',
        type=str,
        required=False,
        help="Path to pickle file of the MR test data.",
    )
    args = parser.parse_args()

    bert_pretrain_dir = "./uncased_L-12_H-768_A-12"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

    ref_filename = args.eval_file_path + "tmp.test.tgt"
    hyp_filename = args.eval_file_path + "tmp.test.src"

    ref_lines = open(ref_filename, encoding='utf-8').read().splitlines()
    hyp_lines = open(hyp_filename, encoding='utf-8').read().splitlines()
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]

    ref_strings = []
    for i in range(len(ref_tokens)):
        cur_ref_int = []
        for j in range(len(ref_tokens[i])):
            cur_ref_int.append(int(ref_tokens[i][j]))
        cur_ref_str = tokenizer.convert_ids_to_tokens(cur_ref_int)
        cur_ref_str = tx.utils.str_join(cur_ref_str)
        hwords_ref = cur_ref_str.replace(" ##", "")
        ref_strings.append(hwords_ref)

    hyp_strings = []
    for i in range(len(hyp_tokens)):
        cur_hyp_int = []
        for j in range(len(hyp_tokens[i])):
            cur_hyp_int.append(int(hyp_tokens[i][j]))
        cur_hyp_str = tokenizer.convert_ids_to_tokens(cur_hyp_int)
        cur_hyp_str = tx.utils.str_join(cur_hyp_str)
        hwords_hyp = cur_hyp_str.replace(" ##", "")
        hyp_strings.append(hwords_hyp)

    tokenizer = tokenization.FullTokenizer(
            vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
            do_lower_case=True)

    test_text_src, test_lines_src, test_text_trg, test_lines_trg = get_test_examples(args.mr_pkl_file_path, tokenizer)

    # calculate EXACT, SARI - using hyp_strings, ref_string, test_lines_src, test_lines_trg
    exact_cnt, mr_exact_cnt = 0, 0
    accum_sari, accum_mr_sari = 0, 0
    for i in range(len(hyp_strings)):
        cur_src = test_lines_src[i]
        cur_hyp = hyp_strings[i]
        cur_ref = ref_strings[i]

        # Prepare hyps and refs for comparison
        cur_proc_hyp = process_example_for_comparison(cur_hyp)
        cur_proc_ref = process_example_for_comparison(cur_ref)

        cur_src_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cur_src))
        cur_hyp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cur_proc_hyp))
        cur_ref_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cur_proc_ref))

        cur_sari, cur_avg_keep_score, cur_avg_addition_score, cur_avg_deletion_score = \
            sari_utils.get_sari_score(cur_src_ids, cur_hyp_ids, [cur_ref_ids], beta_for_deletion=1)
        accum_sari += cur_sari
        max_sari_per_mr_example = cur_sari

        if cur_hyp == cur_ref:
            exact_cnt += 1

        proc_hyp = process_example_for_comparison(cur_hyp)
        for cur_trg in test_lines_trg[i]:
            cur_proc_trg = process_example_for_comparison(cur_trg)
            cur_trg_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cur_proc_trg))
            cur_sari, cur_avg_keep_score, cur_avg_addition_score, cur_avg_deletion_score = \
                sari_utils.get_sari_score(cur_src_ids, cur_hyp_ids, [cur_trg_ids], beta_for_deletion=1)

            if cur_sari > max_sari_per_mr_example:
                max_sari_per_mr_example = cur_sari

            if proc_hyp == cur_proc_trg:
                mr_exact_cnt += 1
                break
        accum_mr_sari += max_sari_per_mr_example

    exact = 100 * exact_cnt / len(hyp_strings)
    mr_exact = 100 * mr_exact_cnt / len(hyp_strings)
    sari = 100 * accum_sari / len(hyp_strings)
    mr_sari = 100 * accum_mr_sari / len(hyp_strings)

    print("--- Results ---")
    print("     EXACT - {}".format(exact))
    print("     MR-EXACT - {}".format(mr_exact))
    print("     SARI - {}".format(sari))
    print("     MR-SARI - {}".format(mr_sari))


if __name__ == "__main__":
    main()

