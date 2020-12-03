import os
from glob import glob
import tensorflow as tf
import tokenization


class DataProcessor:
    def __init__(self, batch_size=16, buffer_size=10000, max_len=256, tokenizer=None):
        self.max_len = max_len
        val_path = 'data/cnn_dailymail/val.tfrecord'
        # test_path = '/content/drive/MyDrive/DLNLP/Final_Project/data/gigaword/test.tfrecord'

        # Create a description of the features.
        feature_description = {
            'article': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'summary': tf.io.FixedLenFeature([], tf.string, default_value='')
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        if tokenizer is None:
            gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
            self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
                                                             do_lower_case=True)
        else:
            self.bert_tokenizer = tokenizer

        self.target_vocab_size = len(self.bert_tokenizer.vocab)

        train_dataset = tf.data.Dataset.list_files('data/cnn_dailymail/train_shards/*.tfrecord')
        train_dataset = tf.data.TFRecordDataset(train_dataset).map(_parse_function)
        train_dataset = train_dataset.map(self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.filter(self.filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size)
        self.train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.TFRecordDataset(val_path).map(_parse_function)
        val_dataset = val_dataset.map(self.tf_encode)
        val_dataset = val_dataset.filter(self.filter_max_length)
        val_dataset = val_dataset.cache()
        self.val_dataset = val_dataset.padded_batch(batch_size)

    def encode(self, article, summary):
        article_tokens = self.bert_tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + self.bert_tokenizer.tokenize(article.numpy()) + ['[SEP]'])
        summary_tokens = self.bert_tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + self.bert_tokenizer.tokenize(summary.numpy()) + ['[SEP]'])

        return article_tokens, summary_tokens

    def tf_encode(self, example):
        result_ar, result_sum = tf.py_function(self.encode, [example['article'], example['summary']],
                                               [tf.int64, tf.int64])
        result_ar.set_shape([None])
        result_sum.set_shape([None])

        return result_ar, result_sum

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self.max_len,
                              tf.size(y) <= self.max_len)
