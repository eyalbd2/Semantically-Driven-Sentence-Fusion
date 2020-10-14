
import sys
import tensorflow as tf
import os
import csv
import collections
import pandas as pd
import pickle
import numpy as np

if not 'texar_repo' in sys.path:
  sys.path += ['texar_repo']
if not 'configs' in sys.path:
  sys.path += ['configs']

from config_with_auxiliary_tasks import *
from examples.bert.utils import data_utils, model_utils, tokenization
from examples.transformer.utils import data_utils, utils


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, type_label, conn_label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.src_txt = text_a
        self.tgt_txt = text_b
        self.type_label = type_label
        self.conn_label = conn_label

        
class InputFeatures():
    """A single set of features of data."""

    def __init__(self, src_input_ids, src_input_mask, src_segment_ids, tgt_input_ids, tgt_input_mask, tgt_labels,
                 type_label, conn_label):
        self.src_input_ids = src_input_ids
        self.src_input_mask = src_input_mask
        self.src_segment_ids = src_segment_ids
        self.tgt_input_ids = tgt_input_ids
        self.tgt_input_mask = tgt_input_mask 
        self.tgt_labels = tgt_labels
        self.type_label = type_label
        self.conn_label = conn_label
        
       
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                lines.append(line)
        return lines


    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\n", quotechar=quotechar)
            lines = []
            i = 0
            for line in reader:
                lines.append(line)
        return lines
      
      
class FusionProcessor(DataProcessor):
    """Processor for the Sentence Fusion data set."""

    def get_train_examples(self, data_dir):
        train = self.load_data_from_file(train_df_path, type2idx_dict_path, con_str2idx_dict_path)
        train_text_src = train['src_sentence'].tolist()
        train_text_src = np.array(train_text_src, dtype=object)[:, np.newaxis]
        train_text_trg = train['trg_sentence'].tolist()
        train_text_trg = np.array(train_text_trg, dtype=object)[:, np.newaxis]
        train_type_label = train['discourse_type'].tolist()
        train_conn_label = train['connective_string'].tolist()
        train_examples = self._create_examples(train_text_src, train_text_trg, train_type_label, train_conn_label,
                                               "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        dev = self.load_data_from_file(dev_df_path, type2idx_dict_path, con_str2idx_dict_path)
        dev_text_src = dev['src_sentence'].tolist()
        dev_text_src = np.array(dev_text_src, dtype=object)[:, np.newaxis]
        dev_text_trg = dev['trg_sentence'].tolist()
        dev_text_trg = np.array(dev_text_trg, dtype=object)[:, np.newaxis]
        dev_type_label = dev['discourse_type'].tolist()
        dev_conn_label = dev['connective_string'].tolist()
        dev_examples = self._create_examples(dev_text_src, dev_text_trg, dev_type_label, dev_conn_label, "dev")
        return dev_examples

    def get_test_examples(self, data_dir):
        test = self.load_data_from_file(test_df_path, type2idx_dict_path, con_str2idx_dict_path)
        test_text_src = test['src_sentence'].tolist()
        test_text_src = np.array(test_text_src, dtype=object)[:, np.newaxis]
        test_text_trg = test['trg_sentence'].tolist()
        test_text_trg = np.array(test_text_trg, dtype=object)[:, np.newaxis]
        test_type_label = test['discourse_type'].tolist()
        test_conn_label = test['connective_string'].tolist()
        test_examples = self._create_examples(test_text_src, test_text_trg, test_type_label, test_conn_label, "test")
        return test_examples

    def load_data_from_file(self, pickle_path, type2idx_dict_path, con_str2idx_dict_path):
        '''Loads source and target data and filters out too lengthy samples.
        fpath_src: source file path. string.
        fpath_trg: target file path. string.
        Returns
        sents1: list of source sents
        sents2: list of target classes
        '''
        pickle_in = open(type2idx_dict_path, "rb")
        type2idx_dict = pickle.load(pickle_in)
        pickle_in = open(con_str2idx_dict_path, "rb")
        con_str2idx_dict = pickle.load(pickle_in)
        pickle_in = open(pickle_path, "rb")
        df = pickle.load(pickle_in)

        data = {}
        try:
            data["src_sentence"] = df["defused_sent"].values
            data["trg_sentence"] = df["fused_sent"].values

            discourse_type = df["discourse_type"].values
            discourse_connective = df["connective_string"].values
            data["discourse_type"], data["connective_string"] = [], []
            for disc_type, conn in zip(discourse_type, discourse_connective):
                data["discourse_type"].append(type2idx_dict[disc_type])
                if (not isinstance(conn, str)) and (not isinstance(conn, int)):
                    data["connective_string"].append(con_str2idx_dict['None'])
                else:
                    data["connective_string"].append(con_str2idx_dict[conn])

        except:
            data["src_sentence"] = df["defused_sent"]
            data["trg_sentence"] = df["fused_sent"]

            discourse_type = df["discourse_type"]
            discourse_connective = df["connective_string"]
            data["discourse_type"], data["connective_string"] = [], []
            for disc_type, conn in zip(discourse_type, discourse_connective):
                data["discourse_type"].append(type2idx_dict[disc_type])
                if (not isinstance(conn, str)) and (not isinstance(conn, int)):
                    data["connective_string"].append(con_str2idx_dict['None'])
                else:
                    data["connective_string"].append(con_str2idx_dict[conn])

        return pd.DataFrame.from_dict(data)

    def _create_examples(self, src_lines, tgt_lines, type_labels, conn_labels, set_type):
        examples = []
        for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            guid = "%s-%s" % (set_type, i)
            src_lines = tokenization.convert_to_unicode(" ".join(src))
            tgt_lines = tokenization.convert_to_unicode(" ".join(tgt))
            type_label = type_labels[i]
            conn_label = conn_labels[i]

            examples.append(InputExample(guid=guid, text_a=src_lines, text_b=tgt_lines, type_label=type_label,
                                         conn_label=conn_label))
        return examples
  
  
def file_based_convert_examples_to_features(examples, max_seq_length_src, max_seq_length_tgt, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if (ex_index+1) % 1000 == 0:
            print("------------processed..{}...examples".format(ex_index))
          
        feature = convert_single_example(ex_index, example, max_seq_length_src, max_seq_length_tgt, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["src_input_ids"] = create_int_feature(feature.src_input_ids)
        features["src_input_mask"] = create_int_feature(feature.src_input_mask)
        features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)

        features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
        features["tgt_input_mask"] = create_int_feature(feature.tgt_input_mask)
        features['tgt_labels'] = create_int_feature(feature.tgt_labels)
        features['type_label'] = create_int_feature([feature.type_label])
        features['conn_label'] = create_int_feature([feature.conn_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def convert_single_example(ex_index, example, max_seq_length_src, max_seq_length_tgt, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    """
    tokens_a = tokenizer.tokenize(example.src_txt)
    tokens_b = tokenizer.tokenize(example.tgt_txt)
    type_label = example.type_label
    conn_label = example.conn_label

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    if len(tokens_a) > max_seq_length_src - 2:
            tokens_a = tokens_a[0:(max_seq_length_src - 2)]
    
    if len(tokens_b) > max_seq_length_tgt - 2:
            tokens_b = tokens_b[0:(max_seq_length_tgt - 2)]

    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]")
    segment_ids_src.append(0)
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")
    segment_ids_src.append(0)

    tokens_tgt = []
    segment_ids_tgt = []
    tokens_tgt.append("[CLS]")
    for token in tokens_b:
        tokens_tgt.append(token)
    tokens_tgt.append("[SEP]")

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
    input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)

    labels_tgt = input_ids_tgt[1:]
    
    # Add begining and end token
    input_ids_tgt = input_ids_tgt[:-1] 
    
    input_mask_src = [1] * len(input_ids_src)

    input_mask_tgt = [1] * len(input_ids_tgt)
    
    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)

    while len(input_ids_tgt) < max_seq_length_tgt:
        input_ids_tgt.append(0)
        input_mask_tgt.append(0)
        segment_ids_tgt.append(0)
        labels_tgt.append(0)

    feature = InputFeatures(src_input_ids=input_ids_src, src_input_mask=input_mask_src, src_segment_ids=segment_ids_src,
                            tgt_input_ids=input_ids_tgt, tgt_input_mask=input_mask_tgt, tgt_labels=labels_tgt,
                            type_label=type_label, conn_label=conn_label)
    return feature


def file_based_input_fn_builder(input_file, max_seq_length_src, max_seq_length_tgt, is_training,
                                drop_remainder, is_distributed=False):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "src_input_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_input_mask": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_segment_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "tgt_input_ids": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_input_mask": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_labels": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "type_label": tf.FixedLenFeature([], tf.int64),
        "conn_label": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        print(example)
        print(example.keys())

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:

            if is_distributed:
                import horovod.tensorflow as hvd
                tf.logging.info('distributed mode is enabled.'
                                'size:{} rank:{}'.format(hvd.size(), hvd.rank()))
                # https://github.com/uber/horovod/issues/223
                d = d.shard(hvd.size(), hvd.rank())

                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size//hvd.size(),
                        drop_remainder=drop_remainder))
            else:
                tf.logging.info('distributed mode is not enabled.')
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))

        else:
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

        return d
    return input_fn
  
  
def get_dataset(processor,
                tokenizer,
                data_dir,
                max_seq_length_src,
                max_seq_length_tgt,
                batch_size,
                mode,
                output_dir,
                is_distributed=False):
    """
    Args:
        processor: Data Preprocessor, must have get_lables,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        batch_size: mini-batch size.
        model: `train`, `eval` or `test`.
        output_dir: The directory to save the TFRecords in.
    """

    if mode == 'train':
        train_examples = processor.get_train_examples(data_dir)
        train_file = os.path.join(output_dir, "train.tf_record")
        
        file_based_convert_examples_to_features(
            train_examples, max_seq_length_src, max_seq_length_tgt,
            tokenizer, train_file)
        dataset = file_based_input_fn_builder(
            input_file=train_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt=max_seq_length_tgt,
            is_training=True,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'eval':
        eval_examples = processor.get_dev_examples(data_dir)
        eval_file = os.path.join(output_dir, "eval.tf_record")
        
        file_based_convert_examples_to_features(
            eval_examples, max_seq_length_src, max_seq_length_tgt,
            tokenizer, eval_file)
        dataset = file_based_input_fn_builder(
            input_file=eval_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt=max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'test':
        test_examples = processor.get_test_examples(data_dir)
        test_file = os.path.join(output_dir, "test.tf_record")

        file_based_convert_examples_to_features(
            test_examples, max_seq_length_src,max_seq_length_tgt,
            tokenizer, test_file)
        dataset = file_based_input_fn_builder(
            input_file=test_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt=max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})

    return dataset


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

    vocab_size = len(tokenizer.vocab)
    out_dir = "data/" + inner_data_dir + in_domain
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    processor = FusionProcessor()
    train_dataset = get_dataset(processor, tokenizer, data_dir, max_seq_length_src, max_seq_length_tgt, batch_size,
                                'train', out_dir)
    eval_dataset = get_dataset(processor, tokenizer, data_dir, max_seq_length_src, max_seq_length_tgt, batch_size,
                               'eval', out_dir)
    test_dataset = get_dataset(processor, tokenizer, data_dir, max_seq_length_src, max_seq_length_tgt, batch_size,
                               'test', out_dir)

